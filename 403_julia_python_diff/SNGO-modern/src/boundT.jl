"""
    Bound Tightening (FBBT / OBBT)

Ported from boundT.jl (~1611 lines). Contains:
  - FBBT: Feasibility-Based Bound Tightening (custom forward/backward propagation)
  - OBBT: Optimization-Based Bound Tightening (LP-based)
  - Reduced cost BT
  - Interval arithmetic helpers
  - Stochastic bound sync wrappers
"""

# ─────────────────────────────────────────────────────────────────────
# Affine backward propagation
# ─────────────────────────────────────────────────────────────────────

"""
    AffineBackward!(aff, lb, ub) -> Symbol

Backward propagation on a linear expression aff within [lb, ub].
Tightens variable bounds based on constraint feasibility.
Returns :feasible, :infeasible, or :updated.
"""
function AffineBackward!(aff::AffExprData, P::ModelWrapper, lb::Float64, ub::Float64)
    for j in 1:length(aff.vars)
        alpha = aff.coeffs[j]
        abs(alpha) < 1e-12 && continue

        var_col = aff.vars[j]
        xl = P.colLower[var_col]
        xu = P.colUpper[var_col]

        sum_min = aff.constant
        sum_max = aff.constant
        for k in 1:length(aff.vars)
            k == j && continue
            ck = aff.coeffs[k]
            vk = aff.vars[k]
            xlk = P.colLower[vk]
            xuk = P.colUpper[vk]
            sum_min += min(ck * xlk, ck * xuk)
            sum_max += max(ck * xlk, ck * xuk)
        end

        if alpha > 0
            # xl_trial = (lb - sum_max) / alpha
            # xu_trial = (ub - sum_min) / alpha
            if isfinite(lb)
                xl_trial = (lb - sum_max) / alpha
                if xl_trial > xl + small_bound_improve
                    P.colLower[var_col] = xl_trial
                end
            end
            if isfinite(ub)
                xu_trial = (ub - sum_min) / alpha
                if xu_trial < xu - small_bound_improve
                    P.colUpper[var_col] = xu_trial
                end
            end
        elseif alpha < 0
            if isfinite(ub)
                xl_trial = (ub - sum_min) / alpha
                if xl_trial > xl + small_bound_improve
                    P.colLower[var_col] = xl_trial
                end
            end
            if isfinite(lb)
                xu_trial = (lb - sum_max) / alpha
                if xu_trial < xu - small_bound_improve
                    P.colUpper[var_col] = xu_trial
                end
            end
        end

        # Check feasibility
        if P.colLower[var_col] > P.colUpper[var_col] + machine_error
            return :infeasible
        end
    end
    return :feasible
end

# ─────────────────────────────────────────────────────────────────────
# Linear backward (for MultiVariable groups)
# ─────────────────────────────────────────────────────────────────────

function linearBackward!(mv_lb::Vector{Float64}, mv_ub::Vector{Float64},
                         lb::Float64, ub::Float64)
    n = length(mv_lb)
    for j in 1:n
        sum_min = 0.0
        sum_max = 0.0
        for k in 1:n
            k == j && continue
            sum_min += mv_lb[k]
            sum_max += mv_ub[k]
        end

        if isfinite(lb)
            new_lb = lb - sum_max
            if new_lb > mv_lb[j] + machine_error
                mv_lb[j] = new_lb
            end
        end
        if isfinite(ub)
            new_ub = ub - sum_min
            if new_ub < mv_ub[j] - machine_error
                mv_ub[j] = new_ub
            end
        end

        if mv_lb[j] > mv_ub[j] + machine_error
            return :infeasible
        end
    end
    return :feasible
end

# ─────────────────────────────────────────────────────────────────────
# Interval arithmetic
# ─────────────────────────────────────────────────────────────────────

function Interval_cal(aff::AffExprData, P::ModelWrapper)
    sum_min = aff.constant
    sum_max = aff.constant
    for k in 1:length(aff.vars)
        xlk = P.colLower[aff.vars[k]]
        xuk = P.colUpper[aff.vars[k]]
        coeff = aff.coeffs[k]
        sum_min += min(coeff * xlk, coeff * xuk)
        sum_max += max(coeff * xlk, coeff * xuk)
    end
    return (sum_min, sum_max)
end

function Interval_cal(q::QuadExprData, P::ModelWrapper)
    sum_min, sum_max = Interval_cal(q.aff, P)
    for k in 1:length(q.qvars1)
        xlk1 = P.colLower[q.qvars1[k]]
        xuk1 = P.colUpper[q.qvars1[k]]
        xlk2 = P.colLower[q.qvars2[k]]
        xuk2 = P.colUpper[q.qvars2[k]]
        coeff = q.qcoeffs[k]
        if q.qvars1[k] == q.qvars2[k]
            if xlk1 <= 0 && 0 <= xuk1
                sum_min += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
                sum_max += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
            else
                sum_min += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
                sum_max += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
            end
        else
            sum_min += min(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
            sum_max += max(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
        end
    end
    return (sum_min, sum_max)
end

# ─────────────────────────────────────────────────────────────────────
# MultiVariable forward/backward propagation
# ─────────────────────────────────────────────────────────────────────

function multiVariableForward(mv::MultiVariable, P::ModelWrapper)
    # Compute interval bounds for a quadratic multi-variable expression
    return Interval_cal(mv.terms, P)
end

function multiVariableBackward!(mv::MultiVariable, P::ModelWrapper,
                                sum_min::Float64, sum_max::Float64,
                                target_lb::Float64, target_ub::Float64)
    # Backward propagation for individual variables within a MultiVariable group
    for varId in mv.qVarsId
        xl = P.colLower[varId]
        xu = P.colUpper[varId]

        # Compute contribution of other variables
        other_min = sum_min
        other_max = sum_max
        # This is a simplified version — full implementation would recompute
        # intervals excluding the current variable

        # Tighten based on contribution bounds
        if target_ub < other_max - machine_error
            # May be able to tighten xu
        end
        if target_lb > other_min + machine_error
            # May be able to tighten xl
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# FBBT — Feasibility-Based Bound Tightening (main function)
# ─────────────────────────────────────────────────────────────────────

"""
    fast_feasibility_reduction!(P, pr, U) -> Bool

Perform one round of FBBT on model P using preprocessing data pr.
U is the current upper bound. Returns false if infeasible.
"""
function fast_feasibility_reduction!(P::ModelWrapper, pr::PreprocessResult,
                                     U::Float64=1e20)
    feasible = true

    # Propagate through exp constraints
    for ev in pr.expVariable_list
        lvarId = ev.lvarId; nlvarId = ev.nlvarId
        op = ev.op; b = ev.b
        xl = P.colLower[nlvarId]; xu = P.colUpper[nlvarId]
        yl = P.colLower[lvarId];  yu = P.colUpper[lvarId]

        if b > 0
            xmin = exp(b * xl); xmax = exp(b * xu)
        else
            xmin = exp(b * xu); xmax = exp(b * xl)
        end

        if op == :(<=) || op == :(==)
            if yl - xmax >= machine_error
                return false
            end
            if yu - xmax >= small_bound_improve
                yu = xmax + small_bound_improve
            end
            if yl - xmin >= machine_error
                temp = log(yl) / b
                if b >= 0
                    if (temp - xl) >= small_bound_improve
                        xl = temp - small_bound_improve
                    end
                else
                    if (xu - temp) >= small_bound_improve
                        xu = temp + small_bound_improve
                    end
                end
            end
        end

        if op == :(>=) || op == :(==)
            if xmin - yu >= machine_error
                return false
            end
            if xmin - yl >= small_bound_improve
                yl = xmin - small_bound_improve
            end
            if xmax - yu >= machine_error
                temp = log(yu) / b
                if b >= 0
                    if (xu - temp) >= small_bound_improve
                        xu = temp + small_bound_improve
                    end
                else
                    if (temp - xl) >= small_bound_improve
                        xl = temp - small_bound_improve
                    end
                end
            end
        end

        P.colLower[nlvarId] = xl; P.colUpper[nlvarId] = xu
        P.colLower[lvarId] = yl;  P.colUpper[lvarId] = yu
    end

    # Propagate through log constraints
    for ev in pr.logVariable_list
        lvarId = ev.lvarId; nlvarId = ev.nlvarId
        op = ev.op; b = ev.b
        xl = P.colLower[nlvarId]; xu = P.colUpper[nlvarId]
        yl = P.colLower[lvarId];  yu = P.colUpper[lvarId]

        (b*xl <= 0 || b*xu <= 0) && continue

        xmin = min(log(b*xl), log(b*xu))
        xmax = max(log(b*xl), log(b*xu))

        if op == :(<=) || op == :(==)
            yl - xmax >= machine_error && return false
            yu - xmax >= small_bound_improve && (yu = xmax + small_bound_improve)
        end

        if op == :(>=) || op == :(==)
            xmin - yu >= machine_error && return false
            xmin - yl >= small_bound_improve && (yl = xmin - small_bound_improve)
        end

        P.colLower[nlvarId] = xl; P.colUpper[nlvarId] = xu
        P.colLower[lvarId] = yl;  P.colUpper[lvarId] = yu
    end

    # Propagate through power constraints
    for ev in pr.powerVariable_list
        lvarId = ev.lvarId; nlvarId = ev.nlvarId
        op = ev.op; b = ev.b; d = ev.d
        xl = P.colLower[nlvarId]; xu = P.colUpper[nlvarId]
        yl = P.colLower[lvarId];  yu = P.colUpper[lvarId]

        # Fractional power requires positive base
        if !isinteger(d)
            if b >= 0 && xl <= 0
                xl = 0.0
            elseif (b >= 0 && xu <= 0) || (b < 0 && xl >= 0)
                return false
            elseif b < 0 && xu >= 0
                xu = 0.0
            end
        end

        d == 0 && error("a^0 detected, please reformulate!")

        if bounded(xl) && bounded(xu)
            xmin = min((b*xl)^d, (b*xu)^d)
            xmax = max((b*xl)^d, (b*xu)^d)
        else
            xmin = -Inf; xmax = Inf
        end

        if isinteger(d) && iseven(Int(d)) && xl <= 0 && xu >= 0 && d > 0
            xmin = min(xmin, 0.0)
        end

        if op == :(<=) || op == :(==)
            yl - xmax >= machine_error && return false
            yu - xmax >= small_bound_improve && (yu = xmax + small_bound_improve)
        end

        if op == :(>=) || op == :(==)
            xmin - yu >= machine_error && return false
            xmin - yl >= small_bound_improve && (yl = xmin - small_bound_improve)
        end

        P.colLower[nlvarId] = xl; P.colUpper[nlvarId] = xu
        P.colLower[lvarId] = yl;  P.colUpper[lvarId] = yu
    end

    # Propagate through linear constraints
    for lc in P.linconstr
        status = AffineBackward!(lc.terms, P, lc.lb, lc.ub)
        if status == :infeasible
            return false
        end
    end

    # Propagate through quadratic constraints via MultiVariable decomposition
    if !isempty(P.quadconstr) && !isempty(pr.multiVariable_list)
        for i in 1:length(P.quadconstr)
            i > length(pr.multiVariable_list) && break
            con = P.quadconstr[i]
            lb = con.sense == :(<=) ? -1e20 : 0.0
            ub = con.sense == :(>=) ? 1e20 : 0.0

            mv_con = pr.multiVariable_list[i]
            mvs = mv_con.mvs
            nmw = length(mvs)

            mv_sum_min = zeros(nmw + 1)
            mv_sum_max = zeros(nmw + 1)
            for j in 1:nmw
                mv_sum_min[j], mv_sum_max[j] = multiVariableForward(mvs[j], P)
            end
            mv_sum_min[nmw+1], mv_sum_max[nmw+1] = Interval_cal(mv_con.aff, P)

            mv_lb = copy(mv_sum_min)
            mv_ub = copy(mv_sum_max)

            status = linearBackward!(mv_lb, mv_ub, lb, ub)
            if status == :infeasible
                return false
            end
        end
    end

    # Propagate through objective
    obj = P.obj
    aff = obj.aff
    for j in 1:length(aff.vars)
        alpha = aff.coeffs[j]
        abs(alpha) < 1e-12 && continue
        var_col = aff.vars[j]
        xl = P.colLower[var_col]
        xu = P.colUpper[var_col]

        sum_min = aff.constant
        for k in 1:length(aff.vars)
            k == j && continue
            xlk = P.colLower[aff.vars[k]]
            xuk = P.colUpper[aff.vars[k]]
            coeff = aff.coeffs[k]
            sum_min += min(coeff * xlk, coeff * xuk)
        end

        if alpha < 0
            xl_trial = (U - sum_min) / alpha
            if xl_trial > xl + small_bound_improve
                P.colLower[var_col] = xl_trial
            end
        elseif alpha > 0
            xu_trial = (U - sum_min) / alpha
            if xu_trial < xu - small_bound_improve
                P.colUpper[var_col] = xu_trial
            end
        end
    end

    return feasible
end

# ─────────────────────────────────────────────────────────────────────
# Stochastic FBBT — iterate across scenarios
# ─────────────────────────────────────────────────────────────────────

function Sto_fast_feasibility_reduction!(P::ModelWrapper, pr_children,
                                         Pex::ModelWrapper, prex, Rold,
                                         UB::Float64, LB::Float64,
                                         depth::Int=0, do_update::Bool=true)
    scenarios = getchildren(P)
    nscen = length(scenarios)
    feasible = true

    # FBBT on each scenario
    for (idx, scen) in enumerate(scenarios)
        if idx <= length(pr_children)
            f = fast_feasibility_reduction!(scen, pr_children[idx], UB)
            if !f
                feasible = false
                return feasible
            end
        end
    end

    # Sync first-stage bounds across scenarios
    if do_update
        updateStoFirstBounds!(P)
    end

    return feasible
end

# ─────────────────────────────────────────────────────────────────────
# OBBT — Optimization-Based Bound Tightening
# ─────────────────────────────────────────────────────────────────────

function optimality_reduction_range(P::ModelWrapper, pr::PreprocessResult,
                                    Rold::ModelWrapper, U::Float64,
                                    varsId::Vector{Int},
                                    optimizer_factory)
    !OBBT && return true
    feasible = true

    R = updaterelax(Rold, P, pr, U)

    # Relax integers for OBBT
    if hasBin
        for i in 1:R.numCols
            if R.colCat[i] == :Bin
                R.colCat[i] = :Cont
            end
        end
    end

    for varId in varsId
        xl = P.colLower[varId]
        xu = P.colUpper[varId]

        # Minimize variable
        R_min = copyModel(R)
        R_min.obj = QuadExprData(Int[], Int[], Float64[],
                                 AffExprData([varId], [1.0], 0.0))
        R_min.objSense = :Min
        status = solve_model!(R_min, optimizer_factory)

        if status == :Infeasible
            return false
        end
        if status == :Optimal
            R_obj = R_min.objVal
            if (P.colCat[varId] in (:Bin, :Int) && (R_obj - xl) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (R_obj - xl) >= machine_error)
                P.colLower[varId] = R_obj
                R.colLower[varId] = R_obj
            end
        end

        # Maximize variable
        R_max = copyModel(R)
        R_max.obj = QuadExprData(Int[], Int[], Float64[],
                                 AffExprData([varId], [1.0], 0.0))
        R_max.objSense = :Max
        status = solve_model!(R_max, optimizer_factory)

        if status == :Infeasible
            return false
        end
        if status == :Optimal
            R_obj = R_max.objVal
            if (P.colCat[varId] in (:Bin, :Int) && (xu - R_obj) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (xu - R_obj) >= machine_error)
                P.colUpper[varId] = R_obj
                R.colUpper[varId] = R_obj
            end
        end

        # Check if significant improvement → update relaxation
        if (xu - xl - P.colUpper[varId] + P.colLower[varId]) >= probing_improve
            R = updaterelax(R, P, pr, U)
        end
    end

    return feasible
end

# ─────────────────────────────────────────────────────────────────────
# Reduced cost bound tightening
# ─────────────────────────────────────────────────────────────────────

function reduced_cost_BT!(P::ModelWrapper, pr::PreprocessResult,
                          R::ModelWrapper, U::Float64, node_L::Float64)
    mu = copy(R.redCosts)
    n_reduced = 0
    isempty(mu) && return n_reduced

    for varId in 1:P.numCols
        xl = P.colLower[varId]
        xu = P.colUpper[varId]

        if mu[varId] >= 1e-4
            xu_trial = xl + (U - node_L) / mu[varId]
            if (P.colCat[varId] in (:Bin, :Int) && (xu - xu_trial) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (xu - xu_trial) >= machine_error)
                P.colUpper[varId] = xu_trial
                R.colUpper[varId] = xu_trial
                n_reduced += 1
            end
        elseif mu[varId] <= -1e-4
            xl_trial = xu + (U - node_L) / mu[varId]
            if (P.colCat[varId] in (:Bin, :Int) && (xl_trial - xl) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (xl_trial - xl) >= machine_error)
                P.colLower[varId] = xl_trial
                R.colLower[varId] = xl_trial
                n_reduced += 1
            end
        end
    end

    return n_reduced
end

# ─────────────────────────────────────────────────────────────────────
# Medium and slow feasibility reduction
# ─────────────────────────────────────────────────────────────────────

function Sto_medium_feasibility_reduction(P, pr_children, Pex, prex, Rold,
                                          UB, LB, bVarsId, optimizer_factory)
    feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, Rold, UB, LB, 0, true)
    updateExtensiveBoundsFromSto!(P, Pex)

    if feasible
        feasible = optimality_reduction_range(Pex, prex, Rold, UB, bVarsId, optimizer_factory)
        updateStoBoundsFromExtensive!(Pex, P)
    end

    return feasible
end

function Sto_slow_feasibility_reduction(P, pr_children, Pex, prex, Rold,
                                        UB, LB, bVarsId, optimizer_factory)
    feasible = true
    left_OBBT_inner = 0.0

    while left_OBBT_inner <= 0.9
        xlold = copy(P.colLower)
        xuold = copy(P.colUpper)

        feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, Rold, UB, LB)
        updateExtensiveBoundsFromSto!(P, Pex)

        if feasible
            feasible = optimality_reduction_range(Pex, prex, Rold, UB, bVarsId, optimizer_factory)
            updateStoBoundsFromExtensive!(Pex, P)
        end

        left_OBBT_inner = 1.0
        for i in 1:length(P.colLower)
            if (xuold[i] + P.colLower[i] - xlold[i] - P.colUpper[i]) > small_bound_improve
                left_OBBT_inner *= (P.colUpper[i] - P.colLower[i]) / (xuold[i] - xlold[i])
            end
        end

        !feasible && break
    end

    return feasible
end
