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
    AffineBackward!(aff, P, lb, ub) -> Symbol

Backward propagation on a linear expression aff within [lb, ub].
Tightens variable bounds based on constraint feasibility.
Returns :feasible, :infeasible, or :updated.
"""
function AffineBackward!(aff::AffExprData, P::ModelWrapper, lb::Float64, ub::Float64)
    xls = Float64[]
    xus = Float64[]
    coeffs = copy(aff.coeffs)

    for j in 1:length(aff.vars)
        push!(xls, P.colLower[aff.vars[j]])
        push!(xus, P.colUpper[aff.vars[j]])
    end
    push!(xls, aff.constant)
    push!(xus, aff.constant)
    push!(coeffs, 1.0)

    status = linearBackward!(xls, xus, lb, ub, coeffs)

    if status == :infeasible
        return :infeasible
    end

    result_status = :unupdated
    for j in 1:length(aff.vars)
        var_col = aff.vars[j]
        if P.colCat[var_col] != :Fixed
            if ((P.colCat[var_col] == :Bin || P.colCat[var_col] == :Int) &&
                (xls[j] - P.colLower[var_col]) >= small_bound_improve) ||
               (P.colCat[var_col] == :Cont && (xls[j] - P.colLower[var_col]) >= machine_error)
                P.colLower[var_col] = xls[j]
                result_status = :updated
            end
            if ((P.colCat[var_col] == :Bin || P.colCat[var_col] == :Int) &&
                (P.colUpper[var_col] - xus[j]) >= small_bound_improve) ||
               (P.colCat[var_col] == :Cont && (P.colUpper[var_col] - xus[j]) >= machine_error)
                P.colUpper[var_col] = xus[j]
                result_status = :updated
            end
        end
    end
    return result_status
end

# ─────────────────────────────────────────────────────────────────────
# Linear backward (with optional coefficients)
# ─────────────────────────────────────────────────────────────────────

function linearBackward!(xls::Vector{Float64}, xus::Vector{Float64},
                         lb::Float64, ub::Float64,
                         coeffs::Union{Vector{Float64},Nothing}=nothing)
    status = :unupdated
    coeffs_provided = coeffs !== nothing
    if !coeffs_provided
        coeffs = ones(length(xls))
    end

    if !coeffs_provided
        sum_min = sum(xls)
        sum_max = sum(xus)
    else
        sum_min = 0.0
        sum_max = 0.0
        for k in 1:length(xls)
            sum_min += min(coeffs[k]*xls[k], coeffs[k]*xus[k])
            sum_max += max(coeffs[k]*xls[k], coeffs[k]*xus[k])
        end
    end

    if sum_min >= lb && sum_max <= ub
        return :unupdated
    end

    if (abs(sum_min) < 1e6 && abs(ub) < 1e6 && (sum_min - ub) >= machine_error) ||
       (abs(sum_max) < 1e6 && abs(lb) < 1e6 && (lb - sum_max) >= machine_error)
        return :infeasible
    end

    for j in 1:length(xls)
        alpha = coeffs[j]
        abs(alpha) <= 1e-6 && continue

        xl = xls[j]
        xu = xus[j]

        sum_min_except = 0.0
        sum_max_except = 0.0
        for k in 1:length(xls)
            if k != j
                sum_min_except += min(coeffs[k]*xls[k], coeffs[k]*xus[k])
                sum_max_except += max(coeffs[k]*xls[k], coeffs[k]*xus[k])
            end
        end

        xu_trial = xu
        xl_trial = xl
        if alpha > 0
            xu_trial = (ub - sum_min_except) / alpha
            xl_trial = (lb - sum_max_except) / alpha
        elseif alpha < 0
            xl_trial = (ub - sum_min_except) / alpha
            xu_trial = (lb - sum_max_except) / alpha
        end

        if (xl_trial - xl) >= machine_error
            xls[j] = xl_trial
            status = :updated
        end
        if (xu - xu_trial) >= machine_error
            xus[j] = xu_trial
            status = :updated
        end
    end
    return status
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
# MultiVariable forward propagation (full per-variable decomposition)
# ─────────────────────────────────────────────────────────────────────

function multiVariableForward(mv::MultiVariable, P::ModelWrapper)
    terms = mv.terms
    qvars1 = terms.qvars1
    qvars2 = terms.qvars2
    qcoeffs = terms.qcoeffs
    aff = terms.aff

    sum_min, sum_max = Interval_cal(terms, P)

    # Per-variable decomposition for tighter bounds
    # square_coeff x^2 + [alpha_min, alpha_max]x + [sum_min_trial, sum_max_trial]
    varsInCon = unique([aff.vars; qvars1; qvars2])

    for varId in varsInCon
        square_coeff = 0.0
        alpha_min = 0.0
        alpha_max = 0.0
        sum_min_trial = aff.constant
        sum_max_trial = aff.constant

        for k in 1:length(aff.vars)
            if varId == aff.vars[k]
                alpha_min += aff.coeffs[k]
                alpha_max += aff.coeffs[k]
            else
                xlk = P.colLower[aff.vars[k]]
                xuk = P.colUpper[aff.vars[k]]
                coeff = aff.coeffs[k]
                sum_min_trial += min(coeff*xlk, coeff*xuk)
                sum_max_trial += max(coeff*xlk, coeff*xuk)
            end
        end

        for k in 1:length(qvars1)
            if qvars1[k] == varId && qvars2[k] == varId
                square_coeff += qcoeffs[k]
            elseif qvars1[k] != varId && qvars2[k] != varId
                xlk1 = P.colLower[qvars1[k]]; xuk1 = P.colUpper[qvars1[k]]
                xlk2 = P.colLower[qvars2[k]]; xuk2 = P.colUpper[qvars2[k]]
                coeff = qcoeffs[k]
                if qvars1[k] == qvars2[k]
                    if xlk1 <= 0 && 0 <= xuk1
                        sum_min_trial += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
                        sum_max_trial += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
                    else
                        sum_min_trial += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
                        sum_max_trial += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
                    end
                else
                    sum_min_trial += min(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
                    sum_max_trial += max(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
                end
            elseif qvars1[k] == varId
                xlk2 = P.colLower[qvars2[k]]; xuk2 = P.colUpper[qvars2[k]]
                coeff = qcoeffs[k]
                alpha_min += min(coeff*xlk2, coeff*xuk2)
                alpha_max += max(coeff*xlk2, coeff*xuk2)
            else  # qvars2[k] == varId
                xlk1 = P.colLower[qvars1[k]]; xuk1 = P.colUpper[qvars1[k]]
                coeff = qcoeffs[k]
                alpha_min += min(coeff*xlk1, coeff*xuk1)
                alpha_max += max(coeff*xlk1, coeff*xuk1)
            end
        end

        xl = P.colLower[varId]
        xu = P.colUpper[varId]

        if abs(square_coeff) == 0
            sum_min_temp = sum_min_trial + min(alpha_min*xl, alpha_min*xu, alpha_max*xl, alpha_max*xu)
            sum_max_temp = sum_max_trial + max(alpha_min*xl, alpha_min*xu, alpha_max*xl, alpha_max*xu)
            if sum_min_temp > sum_min
                sum_min = sum_min_temp
            end
            if sum_max_temp < sum_max
                sum_max = sum_max_temp
            end
        end
    end

    # Special case: a x^2 + b x
    if length(qvars1) == 1 && qvars1[1] == qvars2[1]
        a = qcoeffs[1]
        varId = qvars1[1]
        if a != 0 && aff.constant == 0 && length(aff.vars) <= 1
            b_coeff = length(aff.coeffs) == 1 ? aff.coeffs[1] : 0.0
            xl = P.colLower[varId]
            xu = P.colUpper[varId]
            s_min = min(a*xl^2 + b_coeff*xl, a*xu^2 + b_coeff*xu)
            s_max = max(a*xl^2 + b_coeff*xl, a*xu^2 + b_coeff*xu)
            vertex = -b_coeff / (2*a)
            if xl <= vertex <= xu
                s_min = min(s_min, -b_coeff^2 / a / 4.0)
                s_max = max(s_max, -b_coeff^2 / a / 4.0)
            end
            sum_min = s_min
            sum_max = s_max
        end
    end

    return (sum_min, sum_max)
end

# ─────────────────────────────────────────────────────────────────────
# MultiVariable backward propagation (full version)
# ─────────────────────────────────────────────────────────────────────

function multiVariableBackward!(mv::MultiVariable, P::ModelWrapper,
                                sum_min::Float64, sum_max::Float64,
                                lb::Float64, ub::Float64)
    if sum_min >= lb && sum_max <= ub
        return
    end

    terms = mv.terms
    qvars1 = terms.qvars1
    qvars2 = terms.qvars2
    qcoeffs = terms.qcoeffs
    aff = terms.aff

    varsInCon = unique([aff.vars; qvars1; qvars2])

    for varId in varsInCon
        square_coeff = 0.0
        alpha_min = 0.0
        alpha_max = 0.0
        s_min = aff.constant
        s_max = aff.constant

        for k in 1:length(aff.vars)
            if varId == aff.vars[k]
                alpha_min += aff.coeffs[k]
                alpha_max += aff.coeffs[k]
            else
                xlk = P.colLower[aff.vars[k]]; xuk = P.colUpper[aff.vars[k]]
                coeff = aff.coeffs[k]
                s_min += min(coeff*xlk, coeff*xuk)
                s_max += max(coeff*xlk, coeff*xuk)
            end
        end

        for k in 1:length(qvars1)
            if qvars1[k] == varId && qvars2[k] == varId
                square_coeff += qcoeffs[k]
            elseif qvars1[k] != varId && qvars2[k] != varId
                xlk1 = P.colLower[qvars1[k]]; xuk1 = P.colUpper[qvars1[k]]
                xlk2 = P.colLower[qvars2[k]]; xuk2 = P.colUpper[qvars2[k]]
                coeff = qcoeffs[k]
                if qvars1[k] == qvars2[k]
                    if xlk1 <= 0 && 0 <= xuk1
                        s_min += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
                        s_max += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1, 0)
                    else
                        s_min += min(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
                        s_max += max(coeff*xlk1*xlk1, coeff*xuk1*xuk1)
                    end
                else
                    s_min += min(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
                    s_max += max(coeff*xlk1*xlk2, coeff*xlk1*xuk2, coeff*xuk1*xlk2, coeff*xuk1*xuk2)
                end
            elseif qvars1[k] == varId
                xlk2 = P.colLower[qvars2[k]]; xuk2 = P.colUpper[qvars2[k]]
                coeff = qcoeffs[k]
                alpha_min += min(coeff*xlk2, coeff*xuk2)
                alpha_max += max(coeff*xlk2, coeff*xuk2)
            else
                xlk1 = P.colLower[qvars1[k]]; xuk1 = P.colUpper[qvars1[k]]
                coeff = qcoeffs[k]
                alpha_min += min(coeff*xlk1, coeff*xuk1)
                alpha_max += max(coeff*xlk1, coeff*xuk1)
            end
        end

        xl = P.colLower[varId]
        xu = P.colUpper[varId]
        s_min_temp = s_min
        s_max_temp = s_max

        if abs(square_coeff) > 0
            if xl <= 0 && xu >= 0
                s_min_temp += min(square_coeff*xl*xl, square_coeff*xu*xu, 0)
                s_max_temp += max(square_coeff*xl*xl, square_coeff*xu*xu, 0)
            else
                s_min_temp += min(square_coeff*xl*xl, square_coeff*xu*xu)
                s_max_temp += max(square_coeff*xl*xl, square_coeff*xu*xu)
            end
        end

        xu_trial = xu
        xl_trial = xl

        # Round 1: linear tightening
        if alpha_min > 0
            xu_trial = max((ub - s_min_temp)/alpha_min, (ub - s_min_temp)/alpha_max)
            xl_trial = min((lb - s_max_temp)/alpha_min, (lb - s_max_temp)/alpha_max)
        elseif alpha_max < 0
            xu_trial = max((lb - s_max_temp)/alpha_min, (lb - s_max_temp)/alpha_max)
            xl_trial = min((ub - s_min_temp)/alpha_min, (ub - s_min_temp)/alpha_max)
        end
        xu_trial = min(xu_trial, xu)
        xl_trial = max(xl_trial, xl)

        # Round 2: square root tightening if quadratic
        if abs(square_coeff) > 0
            s_min_temp2 = s_min
            s_max_temp2 = s_max
            s_min_temp2 += min(alpha_min*xl_trial, alpha_min*xu_trial, alpha_max*xl_trial, alpha_max*xu_trial)
            s_max_temp2 += max(alpha_min*xl_trial, alpha_min*xu_trial, alpha_max*xl_trial, alpha_max*xu_trial)
            sqrt_ub = sqrt(max((ub - s_min_temp2)/square_coeff, (lb - s_max_temp2)/square_coeff, 0))
            sqrt_lb = sqrt(max(min((ub - s_min_temp2)/square_coeff, (lb - s_max_temp2)/square_coeff), 0))
            xu_trial = min(xu_trial, sqrt_ub)
            xl_trial = max(xl_trial, -sqrt_ub)
            if xl_trial <= -sqrt_lb && xu_trial <= sqrt_lb && xu_trial >= -sqrt_lb
                xu_trial = -sqrt_lb
            elseif xu_trial >= sqrt_lb && xl_trial <= sqrt_lb && xl_trial >= -sqrt_lb
                xl_trial = sqrt_lb
            end
        end

        # Round 3: vertex-based tightening if quadratic
        if abs(square_coeff) > 0
            s_min_temp3 = s_min
            s_max_temp3 = s_max
            if alpha_min <= 0 && alpha_max >= 0
                s_min_temp3 += min(-alpha_min^2/square_coeff/4, -alpha_max^2/square_coeff/4, 0)
                s_max_temp3 += max(-alpha_min^2/square_coeff/4, -alpha_max^2/square_coeff/4, 0)
            else
                s_min_temp3 += min(-alpha_min^2/square_coeff/4, -alpha_max^2/square_coeff/4)
                s_max_temp3 += max(-alpha_min^2/square_coeff/4, -alpha_max^2/square_coeff/4)
            end
            sqrt_ub = sqrt(max((ub - s_min_temp3)/square_coeff, (lb - s_max_temp3)/square_coeff, 0))
            sqrt_lb = sqrt(max(min((ub - s_min_temp3)/square_coeff, (lb - s_max_temp3)/square_coeff), 0))
            temp_min = min(alpha_min/square_coeff/2, alpha_max/square_coeff/2)
            temp_max = max(alpha_min/square_coeff/2, alpha_max/square_coeff/2)
            xu_trial = min(xu_trial, sqrt_ub - temp_min)
            xl_trial = max(xl_trial, -sqrt_ub - temp_max)
            if (sqrt_lb - temp_max) > (-sqrt_lb - temp_min)
                if xl_trial <= (-sqrt_lb - temp_min) && xu_trial <= (sqrt_lb - temp_max) && xu_trial >= (-sqrt_lb - temp_min)
                    xu_trial = (-sqrt_lb - temp_min)
                elseif xu_trial >= (sqrt_lb - temp_max) && xl_trial <= (sqrt_lb - temp_max) && xl_trial >= (-sqrt_lb - temp_min)
                    xl_trial = (sqrt_lb - temp_max)
                end
            end
        end

        # Apply tightened bounds
        if ((P.colCat[varId] == :Bin || P.colCat[varId] == :Int) &&
            (xl_trial - xl) >= small_bound_improve) ||
           (P.colCat[varId] == :Cont && (xl_trial - xl) >= machine_error)
            P.colLower[varId] = xl_trial
        end
        if ((P.colCat[varId] == :Bin || P.colCat[varId] == :Int) &&
            (xu - xu_trial) >= small_bound_improve) ||
           (P.colCat[varId] == :Cont && (xu - xu_trial) >= machine_error)
            P.colUpper[varId] = xu_trial
        end
    end

    # Special case: a x^2 + b x
    if length(qvars1) == 1 && qvars1[1] == qvars2[1]
        a = qcoeffs[1]
        varId = qvars1[1]
        if a != 0 && aff.constant == 0 && length(aff.vars) <= 1
            b_coeff = length(aff.coeffs) == 1 ? aff.coeffs[1] : 0.0
            xl = P.colLower[varId]
            xu = P.colUpper[varId]
            sqrt_ub = sqrt(max((ub + b_coeff^2/a/4.0)/a, (lb + b_coeff^2/a/4.0)/a, 0))
            sqrt_lb = sqrt(max(min((ub + b_coeff^2/a/4.0)/a, (lb + b_coeff^2/a/4.0)/a), 0))
            temp = b_coeff / a / 2
            xu_trial = min(xu, sqrt_ub - temp)
            xl_trial = max(xl, -sqrt_ub - temp)
            if sqrt_lb > 0
                if xl_trial <= (-sqrt_lb - temp) && xu_trial <= (sqrt_lb - temp) && xu_trial >= (-sqrt_lb - temp)
                    xu_trial = (-sqrt_lb - temp)
                elseif xu_trial >= (sqrt_lb - temp) && xl_trial <= (sqrt_lb - temp) && xl_trial >= (-sqrt_lb - temp)
                    xl_trial = (sqrt_lb - temp)
                end
            end
            if ((P.colCat[varId] == :Bin || P.colCat[varId] == :Int) &&
                (xl_trial - xl) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (xl_trial - xl) >= machine_error)
                P.colLower[varId] = xl_trial
            end
            if ((P.colCat[varId] == :Bin || P.colCat[varId] == :Int) &&
                (xu - xu_trial) >= small_bound_improve) ||
               (P.colCat[varId] == :Cont && (xu - xu_trial) >= machine_error)
                P.colUpper[varId] = xu_trial
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# FBBT inner — one pass through all constraint types
# ─────────────────────────────────────────────────────────────────────

function fast_feasibility_reduction_inner!(P::ModelWrapper, pr, U::Float64=1e20)
    feasible = true

    if pr !== nothing
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
                (yl - xmax) >= machine_error && return false
                (yu - xmax) >= small_bound_improve && (yu = xmax + small_bound_improve)
                if (yl - xmin) >= machine_error
                    temp = log(yl) / b
                    if b >= 0
                        (temp - xl) >= small_bound_improve && (xl = temp - small_bound_improve)
                    else
                        (xu - temp) >= small_bound_improve && (xu = temp + small_bound_improve)
                    end
                end
            end
            if op == :(>=) || op == :(==)
                (xmin - yu) >= machine_error && return false
                (xmin - yl) >= small_bound_improve && (yl = xmin - small_bound_improve)
                if (xmax - yu) >= machine_error
                    temp = log(yu) / b
                    if b >= 0
                        (xu - temp) >= small_bound_improve && (xu = temp + small_bound_improve)
                    else
                        (temp - xl) >= small_bound_improve && (xl = temp - small_bound_improve)
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

            if b >= 0 && xl <= 0
                xl = 1e-20
            elseif (b >= 0 && xu <= 0) || (b < 0 && xl >= 0)
                return false
            elseif b < 0 && xu >= 0
                xu = -1e-20
            end

            xmin = min(log(b*xl), log(b*xu))
            xmax = max(log(b*xl), log(b*xu))

            if op == :(<=) || op == :(==)
                (yl - xmax) >= machine_error && return false
                (yu - xmax) >= small_bound_improve && (yu = xmax + small_bound_improve)
                if (yl - xmin) >= machine_error
                    temp = exp(yl) / b
                    if b >= 0
                        (temp - xl) >= small_bound_improve && (xl = temp - small_bound_improve)
                    else
                        (xu - temp) >= small_bound_improve && (xu = temp + small_bound_improve)
                    end
                end
            end
            if op == :(>=) || op == :(==)
                (xmin - yu) >= machine_error && return false
                (xmin - yl) >= small_bound_improve && (yl = xmin - small_bound_improve)
                if (xmax - yu) >= machine_error
                    temp = exp(yu) / b
                    if b >= 0
                        (xu - temp) >= small_bound_improve && (xu = temp + small_bound_improve)
                    else
                        (temp - xl) >= small_bound_improve && (xl = temp - small_bound_improve)
                    end
                end
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
                (yl - xmax) >= machine_error && return false
                (yu - xmax) >= small_bound_improve && (yu = xmax + small_bound_improve)
            end
            if op == :(>=) || op == :(==)
                (xmin - yu) >= machine_error && return false
                (xmin - yl) >= small_bound_improve && (yl = xmin - small_bound_improve)
            end
            P.colLower[nlvarId] = xl; P.colUpper[nlvarId] = xu
            P.colLower[lvarId] = yl;  P.colUpper[lvarId] = yu
        end

        # Propagate through quadratic constraints via MultiVariable
        # (distributed backward: legacy boundT.jl:1133-1170)
        if !isempty(P.quadconstr) && !isempty(pr.multiVariable_list)
            for i in 1:length(P.quadconstr)
                i > length(pr.multiVariable_list) && break
                con = P.quadconstr[i]
                lb = con.sense == :(<=) ? -1e20 : 0.0
                ub = con.sense == :(>=) ? 1e20 : 0.0

                mv_con = pr.multiVariable_list[i]
                mvs = mv_con.mvs
                nmw = length(mvs)

                # Forward: compute interval for each MV + remaining affine
                mv_sum_min = Vector{Float64}(undef, nmw + 1)
                mv_sum_max = Vector{Float64}(undef, nmw + 1)
                for j in 1:nmw
                    mv_sum_min[j], mv_sum_max[j] = multiVariableForward(mvs[j], P)
                end
                mv_sum_min[nmw+1], mv_sum_max[nmw+1] = Interval_cal(mv_con.aff, P)

                # Distribute constraint bounds across all component intervals
                mv_lb = copy(mv_sum_min)
                mv_ub = copy(mv_sum_max)
                status = linearBackward!(mv_lb, mv_ub, lb, ub)
                if status == :infeasible
                    return false
                end

                # Backward: tighten each MV with its distributed bounds
                for j in 1:nmw
                    multiVariableBackward!(mvs[j], P, mv_sum_min[j], mv_sum_max[j], mv_lb[j], mv_ub[j])
                end

                # Backward: tighten remaining affine variables
                AffineBackward!(mv_con.aff, P, mv_lb[end], mv_ub[end])
            end
        end
    end

    # Propagate through linear constraints
    for lc in P.linconstr
        status = AffineBackward!(lc.terms, P, lc.lb, lc.ub)
        if status == :infeasible
            return false
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
# FBBT — main function with outer convergence loop
# ─────────────────────────────────────────────────────────────────────

function fast_feasibility_reduction!(P::ModelWrapper, pr, U::Float64=1e20)
    n = P.numCols
    feasible = true

    if sum(P.colLower .<= P.colUpper) < n
        return false
    end

    # Binary variable fixing
    hasBin = any(c -> c == :Bin, P.colCat[1:n])
    if hasBin
        for i in 1:n
            if P.colCat[i] == :Bin
                if P.colLower[i] >= machine_error && P.colUpper[i] == 1.0
                    fixVar!(P, i, 1.0)
                elseif P.colLower[i] == 0.0 && (1.0 - P.colUpper[i]) >= machine_error
                    fixVar!(P, i, 0.0)
                elseif P.colLower[i] > machine_error && (1.0 - P.colUpper[i]) >= machine_error
                    return false
                end
            end
        end
    end

    # Iterate until convergence
    feasibility_reduced = 1e10
    while feasibility_reduced >= 1
        xlold = copy(P.colLower)
        xuold = copy(P.colUpper)
        feasible = fast_feasibility_reduction_inner!(P, pr, U)
        if !feasible
            break
        end
        feasibility_reduced = 0
        for i in 1:n
            if (P.colLower[i] - xlold[i]) >= small_bound_improve
                feasibility_reduced += 1
            end
            if (xuold[i] - P.colUpper[i]) >= small_bound_improve
                feasibility_reduced += 1
            end
        end
    end
    return feasible
end

# ─────────────────────────────────────────────────────────────────────
# Stochastic FBBT — full version with relaxation solving
# ─────────────────────────────────────────────────────────────────────

function Sto_fast_feasibility_reduction!(P::ModelWrapper, pr_children,
                                         Pex::ModelWrapper, prex, Rold,
                                         UB::Float64, LB::Float64=-1e20,
                                         ngrid::Int=0, solve_relax::Bool=true)
    nfirst = P.numCols
    scenarios = getchildren(P)
    nscen = length(scenarios)
    feasibility_reduced = 1e10
    rn = 1
    changed = trues(nscen)
    updateStoFirstBounds!(P)
    updateExtensiveBoundsFromSto!(P, Pex)

    while feasibility_reduced >= 1
        feasibility_reduced = 0
        xlold = copy(P.colLower)
        xuold = copy(P.colUpper)

        if Rold !== nothing
            R = updaterelax(Rold, Pex, prex, UB)
            hasBin = any(c -> c == :Bin, R.colCat[1:R.numCols])
            if hasBin
                for i in 1:R.numCols
                    if R.colCat[i] == :Bin
                        R.colCat[i] = :Cont
                    end
                end
            end

            if ngrid > 0
                addOuterApproximationGrid!(R, prex, ngrid)
            end
            fb = fast_feasibility_reduction!(R, nothing, UB)
            if !fb
                return false
            end
            Pex.colLower = R.colLower[1:length(Pex.colLower)]
            Pex.colUpper = R.colUpper[1:length(Pex.colUpper)]

            if solve_relax && rn == 1
                relaxed_status = solve_model!(R, nothing)  # uses default solver
                relaxed_LB = getRobjective(R, relaxed_status, LB)
                if relaxed_status == :Optimal && ((UB - relaxed_LB) <= mingap || (UB - relaxed_LB)/abs(relaxed_LB) <= mingap)
                    return false
                elseif relaxed_status == :Infeasible
                    return false
                end
                if relaxed_status == :Optimal
                    n_reduced = reduced_cost_BT!(Pex, prex, R, UB, relaxed_LB)
                end
            end
            updateStoBoundsFromExtensive!(Pex, P)
            updateStoFirstBounds!(P)
        end

        # FBBT on each scenario
        for (idx, scenario) in enumerate(scenarios)
            if changed[idx] && idx <= length(pr_children)
                fb = fast_feasibility_reduction!(scenario, pr_children[idx], 1e10)
                if !fb
                    return false
                end
            end
        end

        # Sync first-stage bounds from scenarios to master
        for (idx, scenario) in enumerate(scenarios)
            firstVarsId = scenario.ext[:firstVarsId]
            for i in 1:nfirst
                if firstVarsId[i] > 0
                    if scenario.colLower[firstVarsId[i]] > P.colLower[i]
                        P.colLower[i] = scenario.colLower[firstVarsId[i]]
                    end
                    if scenario.colUpper[firstVarsId[i]] < P.colUpper[i]
                        P.colUpper[i] = scenario.colUpper[firstVarsId[i]]
                    end
                end
            end
        end

        # Propagate master bounds back to scenarios
        for (idx, scenario) in enumerate(scenarios)
            changed[idx] = false
            for j in 1:nfirst
                varid = scenario.ext[:firstVarsId][j]
                if varid != -1
                    scenario.colLower[varid] = P.colLower[j]
                    scenario.colUpper[varid] = P.colUpper[j]
                    changed[idx] = true
                end
            end
        end

        # linearBackward! on scenario objective values
        xls = Float64[haskey(scenario.ext, :objLB) ? scenario.ext[:objLB] : -1e20 for scenario in scenarios]
        xus = Float64[haskey(scenario.ext, :objUB) ? scenario.ext[:objUB] : 1e20 for scenario in scenarios]
        status = linearBackward!(xls, xus, LB, UB)
        if status == :infeasible
            return false
        end

        for i in 1:nfirst
            if (xuold[i] + P.colLower[i] - xlold[i] - P.colUpper[i]) > small_bound_improve
                feasibility_reduced += 1
            end
        end
        updateExtensiveBoundsFromSto!(P, Pex)
        rn += 1
    end
    return true
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
    hasBin = any(c -> c == :Bin, R.colCat[1:R.numCols])
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

        # Significant improvement → update relaxation
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
