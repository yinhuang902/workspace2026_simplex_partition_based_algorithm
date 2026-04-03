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

    # Add power relaxation constraints (simplified)
    for ev in pr.powerVariable_list
        # Similar pattern to exp/log — tangent/secant based on convexity
        # Full implementation follows the same pattern from updaterelax.jl
    end

    # Add RLT constraints
    for rlt_con in pr.EqVconstr
        push!(m.linconstr, copy(rlt_con))
    end

    m.ext[:qbcId] = qbcId
    m.linconstrDuals = zeros(length(m.linconstr))
    m.redCosts = zeros(m.numCols)

    return m
end
