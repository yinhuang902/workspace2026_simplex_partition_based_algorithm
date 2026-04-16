"""
    Convex Relaxation Construction

Ported from relax.jl — builds the initial LP relaxation from the
preprocessed model. The relaxation replaces:
  - Bilinear terms x*y with McCormick envelopes
  - exp/log/power terms with tangent/secant linear relaxations
  - Adds UB constraint and RLT cuts
"""

function relax(P::ModelWrapper, pr::PreprocessResult, U::Float64=1e10,
               initialValue::Union{Vector{Float64}, Nothing}=nothing)
    m = copyModel(P)

    qbcId = pr.qbVarsId
    n = P.numCols

    # Set initial values
    if initialValue !== nothing
        m.colVal[1:length(initialValue)] = copy(initialValue)
    end

    # Update McCormick bounds for bilinear variables
    if isa(qbcId, Dict)
        for (key, value) in qbcId
            xid = key[1]
            yid = key[2]
            if xid > yid
                continue  # Process each pair once
            end
            bid = value[1]
            cid = value[2]
            xl = P.colLower[xid]; xu = P.colUpper[xid]
            yl = P.colLower[yid]; yu = P.colUpper[yid]

            m.colLower[bid] = min(xl*yl, xl*yu, xu*yl, xu*yu)
            m.colUpper[bid] = max(xl*yl, xl*yu, xu*yl, xu*yu)
            if xid == yid
                m.colLower[bid] = max(m.colLower[bid], 0.0)
            end
            if initialValue !== nothing && length(initialValue) == n
                m.colVal[bid] = m.colVal[xid] * m.colVal[yid]
            end
        end
    end

    # Add UB constraint: objective <= U
    ub_aff = copy(m.obj.aff)
    # objective_value <= U  (objective is just the obj_value variable)
    push!(m.linconstr, LinearConstraintData(
        copy(m.obj.aff), -Inf, U - m.obj.aff.constant))

    # Add exponential relaxation constraints
    for ev in pr.expVariable_list
        lvarId = ev.lvarId
        nlvarId = ev.nlvarId
        op = ev.op
        b = ev.b
        xl = P.colLower[nlvarId]
        xu = P.colUpper[nlvarId]

        (xu - xl) <= small_bound_improve && continue

        if op == :(<=) || op == :(==)
            # Secant: lvar <= secant line of exp
            slope = (exp(b*xu) - exp(b*xl)) / (xu - xl)
            intercept = (xu*exp(b*xl) - xl*exp(b*xu)) / (xu - xl)
            if abs(slope) <= 1e8 && abs(intercept) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -slope], -intercept)
                push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                ev.cid[1] = length(m.linconstr)
            end
        end

        if op == :(>=) || op == :(==)
            # Tangent at xl: lvar >= tangent at xl
            t_slope = b * exp(b*xl)
            t_intercept = exp(b*xl) * (1 - b*xl)
            if abs(t_slope) <= 1e8 && abs(t_intercept) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope], -t_intercept)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                ev.cid[2] = length(m.linconstr)
            end
            # Tangent at xu
            t_slope2 = b * exp(b*xu)
            t_intercept2 = exp(b*xu) * (1 - b*xu)
            if abs(t_slope2) <= 1e8 && abs(t_intercept2) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope2], -t_intercept2)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                ev.cid[3] = length(m.linconstr)
            end
        end
    end

    # Add log relaxation constraints
    for ev in pr.logVariable_list
        lvarId = ev.lvarId
        nlvarId = ev.nlvarId
        op = ev.op
        b = ev.b
        xl = P.colLower[nlvarId]
        xu = P.colUpper[nlvarId]

        (xu - xl) <= small_bound_improve && continue

        if op == :(<=) || op == :(==)
            # Tangent at xl: lvar <= 1/xl * nlvar + log(b*xl) - 1
            if abs(1/xl) <= 1e8 && abs(log(b*xl)) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -1/xl], -(log(b*xl) - 1))
                push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                ev.cid[1] = length(m.linconstr)
            end
            # Tangent at xu
            if abs(1/xu) <= 1e8 && abs(log(b*xu)) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -1/xu], -(log(b*xu) - 1))
                push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                ev.cid[2] = length(m.linconstr)
            end
        end

        if op == :(>=) || op == :(==)
            # Secant line
            slope = (log(b*xu) - log(b*xl)) / (xu - xl)
            intercept = (xu*log(b*xl) - xl*log(b*xu)) / (xu - xl)
            if abs(slope) <= 1e8 && abs(intercept) <= 1e8
                aff = AffExprData([lvarId, nlvarId], [1.0, -slope], -intercept)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                ev.cid[3] = length(m.linconstr)
            end
        end
    end

    # Add monomial relaxation constraints (d^(b*x) — treated as exp since parsemonomialcon returns ExpVariable)
    for ev in pr.monomialVariable_list
        lvarId = ev.lvarId
        nlvarId = ev.nlvarId
        op = ev.op
        b = ev.b
        ev.cid = [-1, -1, -1]
        xl = P.colLower[nlvarId]
        xu = P.colUpper[nlvarId]

        (xu - xl) <= small_bound_improve && continue

        if op == :(<=) || op == :(==)
            slope = (exp(b*xu) - exp(b*xl)) / (xu - xl)
            intercept = (xu*exp(b*xl) - xl*exp(b*xu)) / (xu - xl)
            if abs(slope) <= 1e8 && abs(intercept) <= 1e8 && bounded(xl) && bounded(xu)
                aff = AffExprData([lvarId, nlvarId], [1.0, -slope], -intercept)
                push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                ev.cid[1] = length(m.linconstr)
            end
        end
        if op == :(>=) || op == :(==)
            t_slope = b * exp(b*xl)
            t_intercept = exp(b*xl) * (1 - b*xl)
            if abs(t_slope) <= 1e8 && abs(t_intercept) <= 1e8 && bounded(xl)
                aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope], -t_intercept)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                ev.cid[2] = length(m.linconstr)
            end
            t_slope2 = b * exp(b*xu)
            t_intercept2 = exp(b*xu) * (1 - b*xu)
            if abs(t_slope2) <= 1e8 && abs(t_intercept2) <= 1e8 && bounded(xu)
                aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope2], -t_intercept2)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                ev.cid[3] = length(m.linconstr)
            end
        end
    end

    # Add power relaxation constraints: lvar op (b*x)^d
    for ev in pr.powerVariable_list
        lvarId = ev.lvarId
        nlvarId = ev.nlvarId
        op = ev.op
        b = ev.b
        d = ev.d
        ev.cid = [-1, -1, -1]
        xl = P.colLower[nlvarId]
        xu = P.colUpper[nlvarId]

        (xu - xl) <= small_bound_improve && continue

        # Fractional powers require positive base
        if positiveFrac(d) || negativeFrac(d)
            if b >= 0
                xl = max(xl, 1e-20)
            else
                xu = min(xu, -1e-20)
            end
        end

        if op == :(<=) || op == :(==)
            if positiveEven(d) || negativeFrac(d) ||
               (negativeEven(d) && xu <= 0) || (negativeEven(d) && xl >= 0) ||
               (Odd(d) && b >= 0 && xl >= 0) || (Odd(d) && b <= 0 && xu <= 0)
                # Secant overestimator
                slope = ((b*xu)^d - (b*xl)^d) / (xu - xl)
                intercept = (xu*(b*xl)^d - xl*(b*xu)^d) / (xu - xl)
                if -1e8 <= slope <= 1e8 && -1e8 <= intercept <= 1e8 && bounded(xl) && bounded(xu)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -slope], -intercept)
                    push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                    ev.cid[1] = length(m.linconstr)
                end
            elseif positiveFrac(d) ||
                   (Odd(d) && b >= 0 && xu <= 0) || (Odd(d) && b <= 0 && xl >= 0)
                # Tangent overestimators at xl and xu
                t_slope_l = b*d*(b*xl)^(d-1)
                t_int_l = -xl*b*d*(b*xl)^(d-1) + (b*xl)^d
                if abs(t_slope_l) <= 1e8 && abs(t_int_l) <= 1e8 && bounded(xl)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope_l], -t_int_l)
                    push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                end
                t_slope_u = b*d*(b*xu)^(d-1)
                t_int_u = -xu*b*d*(b*xu)^(d-1) + (b*xu)^d
                if abs(t_slope_u) <= 1e8 && abs(t_int_u) <= 1e8 && bounded(xu)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope_u], -t_int_u)
                    push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                end
            elseif positiveOdd(d) && xl < 0 && xu > 0
                # Straddling zero — incomplete in original ("to do")
            elseif (negativeEven(d) || negativeOdd(d)) && xl < 0 && xu > 0
                # Nothing to do — no valid single relaxation
            end
        end

        if op == :(>=) || op == :(==)
            if positiveFrac(d) ||
               (Odd(d) && b >= 0 && xu <= 0) || (Odd(d) && b <= 0 && xl >= 0)
                # Secant underestimator
                slope = ((b*xu)^d - (b*xl)^d) / (xu - xl)
                intercept = (xu*(b*xl)^d - xl*(b*xu)^d) / (xu - xl)
                if -1e8 <= slope <= 1e8 && -1e8 <= intercept <= 1e8 && bounded(xl) && bounded(xu)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -slope], -intercept)
                    push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                    ev.cid[1] = length(m.linconstr)
                end
            elseif positiveEven(d) || negativeFrac(d) ||
                   (negativeEven(d) && xu <= 0) || (negativeEven(d) && xl >= 0) ||
                   (Odd(d) && b >= 0 && xl >= 0) || (Odd(d) && b <= 0 && xu <= 0)
                # Tangent underestimators at xl and xu
                t_slope_l = b*d*(b*xl)^(d-1)
                t_int_l = -xl*b*d*(b*xl)^(d-1) + (b*xl)^d
                if abs(t_slope_l) <= 1e8 && abs(t_int_l) <= 1e8 && bounded(xl)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope_l], -t_int_l)
                    push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                end
                t_slope_u = b*d*(b*xu)^(d-1)
                t_int_u = -xu*b*d*(b*xu)^(d-1) + (b*xu)^d
                if abs(t_slope_u) <= 1e8 && abs(t_int_u) <= 1e8 && bounded(xu)
                    aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope_u], -t_int_u)
                    push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                end
            elseif positiveOdd(d)
                # Incomplete in original ("to do")
            elseif (negativeEven(d) || negativeOdd(d)) && xl < 0 && xu > 0
                # Nothing to do
            end
        end
    end

    # Add RLT constraints
    for rlt_con in pr.EqVconstr
        push!(m.linconstr, copy(rlt_con))
    end

    m.ext[:qbcId] = qbcId
    m.linconstrDuals = zeros(length(m.linconstr))
    m.redCosts = zeros(m.numCols)

    println("additional variables: ", m.numCols - P.numCols)
    return m
end
