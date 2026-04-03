"""
    Relaxation Update

Ported from updaterelax.jl — updates the LP relaxation model:
  - Updates McCormick envelope coefficients for new bounds
  - Updates exp/log/power/monomial secant/tangent cuts
  - Adds outer approximation cuts (OA)
  - Adds αBB underestimators
"""

function updaterelax(R::ModelWrapper, P::ModelWrapper, pr::PreprocessResult,
                     U::Float64=1e10, initialValue::Union{Vector{Float64},Nothing}=nothing)
    m = copyModel(R)
    n = P.numCols
    m.colLower[1:n] = copy(P.colLower)
    m.colUpper[1:n] = copy(P.colUpper)

    if initialValue !== nothing
        m.colVal[1:length(initialValue)] = copy(initialValue)
    end

    qbcId = haskey(m.ext, :qbcId) ? m.ext[:qbcId] : pr.qbVarsId

    # Update McCormick bounds for bilinear variables
    if isa(qbcId, Dict)
        for (key, value) in qbcId
            xid = key[1]; yid = key[2]
            xid > yid && continue  # each pair once
            bid = value[1]; cid = value[2]
            xl = P.colLower[xid]; xu = P.colUpper[xid]
            yl = P.colLower[yid]; yu = P.colUpper[yid]

            m.colLower[bid] = min(xl*yl, xl*yu, xu*yl, xu*yu)
            m.colUpper[bid] = max(xl*yl, xl*yu, xu*yl, xu*yu)
            xid == yid && (m.colLower[bid] = max(m.colLower[bid], 0.0))

            if initialValue !== nothing && length(initialValue) == n
                m.colVal[bid] = m.colVal[xid] * m.colVal[yid]
            end

            # Update McCormick constraint coefficients
            if cid >= 1 && cid <= length(m.linconstr)
                if xu != Inf && yu != Inf
                    _updateCon!(m.linconstr[cid], -xu*yu, Inf, xid, -yu, yid, -xu)
                end
                if cid+1 <= length(m.linconstr) && xl != -Inf && yl != -Inf
                    _updateCon!(m.linconstr[cid+1], -xl*yl, Inf, xid, -yl, yid, -xl)
                end
                if cid+2 <= length(m.linconstr) && xl != -Inf && yu != Inf
                    _updateCon!(m.linconstr[cid+2], -Inf, -xl*yu, xid, -yu, yid, -xl)
                end
                if xid != yid && cid+3 <= length(m.linconstr) && xu != Inf && yl != -Inf
                    _updateCon!(m.linconstr[cid+3], -Inf, -xu*yl, xid, -yl, yid, -xu)
                end
            end
        end
    end

    # Update UB constraint
    if !isempty(m.linconstr) && length(P.linconstr) < length(m.linconstr)
        ub_con_idx = length(P.linconstr) + 1
        if ub_con_idx <= length(m.linconstr)
            m.linconstr[ub_con_idx].ub = U - m.obj.aff.constant
        end
    end

    # Update exp relaxation
    for ev in pr.expVariable_list
        nlvarId = ev.nlvarId
        xl = P.colLower[nlvarId]; xu = P.colUpper[nlvarId]
        b = ev.b
        (xu - xl) <= small_bound_improve && continue

        if (ev.op == :(<=) || ev.op == :(==)) && ev.cid[1] != -1
            slope = (exp(b*xu) - exp(b*xl)) / (xu - xl)
            intercept = (xu*exp(b*xl) - xl*exp(b*xu)) / (xu - xl)
            if abs(slope) <= 1e8 && abs(intercept) <= 1e8 && ev.cid[1] <= length(m.linconstr)
                _updateCon_single!(m.linconstr[ev.cid[1]], -Inf, intercept, nlvarId, -slope)
            end
        end

        if (ev.op == :(>=) || ev.op == :(==))
            if ev.cid[2] != -1 && ev.cid[2] <= length(m.linconstr)
                t_slope = b * exp(b*xl)
                t_intercept = exp(b*xl) * (1 - b*xl)
                if abs(t_slope) <= 1e8 && abs(t_intercept) <= 1e8
                    _updateCon_single!(m.linconstr[ev.cid[2]], t_intercept, Inf, nlvarId, -t_slope)
                end
            end
            if ev.cid[3] != -1 && ev.cid[3] <= length(m.linconstr)
                t_slope2 = b * exp(b*xu)
                t_intercept2 = exp(b*xu) * (1 - b*xu)
                if abs(t_slope2) <= 1e8 && abs(t_intercept2) <= 1e8
                    _updateCon_single!(m.linconstr[ev.cid[3]], t_intercept2, Inf, nlvarId, -t_slope2)
                end
            end
        end
    end

    # Update log relaxation
    for ev in pr.logVariable_list
        nlvarId = ev.nlvarId
        xl = P.colLower[nlvarId]; xu = P.colUpper[nlvarId]
        b = ev.b
        (xu - xl) <= small_bound_improve && continue

        if (ev.op == :(<=) || ev.op == :(==))
            if ev.cid[1] != -1 && ev.cid[1] <= length(m.linconstr)
                if abs(1/xl) <= 1e8 && abs(log(b*xl)) <= 1e8
                    _updateCon_single!(m.linconstr[ev.cid[1]], -Inf, log(b*xl)-1, nlvarId, -1/xl)
                end
            end
            if ev.cid[2] != -1 && ev.cid[2] <= length(m.linconstr)
                if abs(1/xu) <= 1e8 && abs(log(b*xu)) <= 1e8
                    _updateCon_single!(m.linconstr[ev.cid[2]], -Inf, log(b*xu)-1, nlvarId, -1/xu)
                end
            end
        end

        if (ev.op == :(>=) || ev.op == :(==)) && ev.cid[3] != -1 && ev.cid[3] <= length(m.linconstr)
            slope = (log(b*xu) - log(b*xl)) / (xu - xl)
            intercept = (xu*log(b*xl) - xl*log(b*xu)) / (xu - xl)
            if abs(slope) <= 1e8 && abs(intercept) <= 1e8
                _updateCon_single!(m.linconstr[ev.cid[3]], intercept, Inf, nlvarId, -slope)
            end
        end
    end

    return m
end

# ─────────────────────────────────────────────────────────────────────
# Constraint update helpers
# ─────────────────────────────────────────────────────────────────────

function _updateCon!(con::LinearConstraintData, lb::Float64, ub::Float64,
                     xid::Int, xcoeff::Float64, yid::Int=0, ycoeff::Float64=0.0)
    con.lb = lb
    con.ub = ub
    if xid == yid && yid != 0
        xcoeff += ycoeff
    end

    # Find and update x coefficient
    found_x = false
    for k in 1:length(con.terms.vars)
        if con.terms.vars[k] == xid
            con.terms.coeffs[k] = xcoeff
            found_x = true
            break
        end
    end
    if !found_x
        push!(con.terms.vars, xid)
        push!(con.terms.coeffs, xcoeff)
    end

    # Find and update y coefficient
    if xid != yid && yid != 0
        found_y = false
        for k in 1:length(con.terms.vars)
            if con.terms.vars[k] == yid
                con.terms.coeffs[k] = ycoeff
                found_y = true
                break
            end
        end
        if !found_y
            push!(con.terms.vars, yid)
            push!(con.terms.coeffs, ycoeff)
        end
    end
end

function _updateCon_single!(con::LinearConstraintData, lb::Float64, ub::Float64,
                            xid::Int, xcoeff::Float64)
    con.lb = lb
    con.ub = ub
    found = false
    for k in 1:length(con.terms.vars)
        if con.terms.vars[k] == xid
            con.terms.coeffs[k] = xcoeff
            found = true
            break
        end
    end
    if !found
        push!(con.terms.vars, xid)
        push!(con.terms.coeffs, xcoeff)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Outer Approximation
# ─────────────────────────────────────────────────────────────────────

function addOuterApproximation!(m::ModelWrapper, pr::PreprocessResult)
    oldncon = length(m.linconstr)
    qbcId = haskey(m.ext, :qbcId) ? m.ext[:qbcId] : pr.qbVarsId

    # OA for x^2 terms
    if isa(qbcId, Dict)
        for (key, value) in qbcId
            xid = key[1]; yid = key[2]
            xid != yid && continue  # only for squared terms
            xid > yid && continue
            bid = value[1]
            xv = m.colVal[xid]
            bv = m.colVal[bid]
            if bv <= (xv^2 - sigma_violation)
                # w >= 2*xv*x - xv^2
                aff = AffExprData([bid, xid], [1.0, -2*xv], xv*xv)
                push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
            end
        end
    end

    # OA for exp terms
    for ev in pr.expVariable_list
        lvarId = ev.lvarId; nlvarId = ev.nlvarId
        op = ev.op; b = ev.b
        xl = m.colLower[nlvarId]; xu = m.colUpper[nlvarId]
        xv = m.colVal[nlvarId]

        if (op == :(>=) || op == :(==))
            if (xv - xl) >= machine_error && (xu - xv) >= machine_error
                t_slope = b * exp(b*xv)
                t_intercept = exp(b*xv) * (1 - b*xv)
                if abs(t_slope) <= 1e8 && abs(t_intercept) <= 1e8
                    aff = AffExprData([lvarId, nlvarId], [1.0, -t_slope], -t_intercept)
                    push!(m.linconstr, LinearConstraintData(aff, 0.0, Inf))
                end
            end
        end
    end

    # OA for log terms
    for ev in pr.logVariable_list
        lvarId = ev.lvarId; nlvarId = ev.nlvarId
        op = ev.op; b = ev.b
        xl = m.colLower[nlvarId]; xu = m.colUpper[nlvarId]
        xv = m.colVal[nlvarId]

        if (op == :(<=) || op == :(==))
            if (xv - xl) >= machine_error && (xu - xv) >= machine_error
                if abs(1/xv) <= 1e8 && abs(log(b*xv)) <= 1e8
                    aff = AffExprData([lvarId, nlvarId], [1.0, -1/xv], -(log(b*xv) - 1))
                    push!(m.linconstr, LinearConstraintData(aff, -Inf, 0.0))
                end
            end
        end
    end

    # OA for convex multivariate quadratic terms
    for mv in pr.multiVariable_convex
        if mv.pd == 1 && length(mv.qVarsId) > 1
            # Convex: add linearization at current point
            _add_convex_OA_cut!(m, mv, qbcId)
        elseif mv.pd == -1 && length(mv.qVarsId) > 1
            # Concave: add linearization at current point (with reversed inequality)
            _add_concave_OA_cut!(m, mv, qbcId)
        end
    end

    return length(m.linconstr) - oldncon
end

function _add_convex_OA_cut!(m::ModelWrapper, mv::MultiVariable, qbcId)
    terms = mv.terms
    aff_new = AffExprData(Int[], Float64[], 0.0)
    constant = 0.0
    qconstant = 0.0

    for k in 1:length(terms.qvars1)
        xid = terms.qvars1[k]; yid = terms.qvars2[k]
        xv = m.colVal[xid]; yv = m.colVal[yid]
        coeff = terms.qcoeffs[k]

        bid = 0
        if k <= length(mv.bilinearVarsId)
            bid = mv.bilinearVarsId[k]
        elseif isa(qbcId, Dict) && haskey(qbcId, (xid, yid))
            bid = qbcId[(xid, yid)][1]
        end
        bid == 0 && continue
        bv = m.colVal[bid]

        push!(aff_new.vars, bid); push!(aff_new.coeffs, coeff)
        push!(aff_new.vars, xid); push!(aff_new.coeffs, -coeff * yv)
        push!(aff_new.vars, yid); push!(aff_new.coeffs, -coeff * xv)
        constant += coeff * xv * yv
        qconstant += coeff * bv
    end

    if qconstant <= (constant - sigma_violation)
        aff_new.constant = constant
        push!(m.linconstr, LinearConstraintData(aff_new, 0.0, Inf))
    end
end

function _add_concave_OA_cut!(m::ModelWrapper, mv::MultiVariable, qbcId)
    terms = mv.terms
    aff_new = AffExprData(Int[], Float64[], 0.0)
    constant = 0.0
    qconstant = 0.0

    for k in 1:length(terms.qvars1)
        xid = terms.qvars1[k]; yid = terms.qvars2[k]
        xv = m.colVal[xid]; yv = m.colVal[yid]
        coeff = terms.qcoeffs[k]

        bid = 0
        if k <= length(mv.bilinearVarsId)
            bid = mv.bilinearVarsId[k]
        elseif isa(qbcId, Dict) && haskey(qbcId, (xid, yid))
            bid = qbcId[(xid, yid)][1]
        end
        bid == 0 && continue
        bv = m.colVal[bid]

        push!(aff_new.vars, bid); push!(aff_new.coeffs, coeff)
        push!(aff_new.vars, xid); push!(aff_new.coeffs, -coeff * yv)
        push!(aff_new.vars, yid); push!(aff_new.coeffs, -coeff * xv)
        constant += coeff * xv * yv
        qconstant += coeff * bv
    end

    if qconstant >= (constant + sigma_violation)
        aff_new.constant = constant
        push!(m.linconstr, LinearConstraintData(aff_new, -Inf, 0.0))
    end
end

function addaBB!(m::ModelWrapper, pr::PreprocessResult)
    oldncon = length(m.linconstr)
    qbcId = haskey(m.ext, :qbcId) ? m.ext[:qbcId] : pr.qbVarsId

    for mv in pr.multiVariable_aBB
        terms = mv.terms
        alpha = mv.alpha
        aff_new = AffExprData(Int[], Float64[], 0.0)
        constant = 0.0

        for k in 1:length(terms.qvars1)
            xid = terms.qvars1[k]; yid = terms.qvars2[k]
            xv = m.colVal[xid]; yv = m.colVal[yid]
            coeff = terms.qcoeffs[k]

            bid = 0
            if k <= length(mv.bilinearVarsId)
                bid = mv.bilinearVarsId[k]
            end
            bid == 0 && continue

            push!(aff_new.vars, bid); push!(aff_new.coeffs, coeff)
            push!(aff_new.vars, xid); push!(aff_new.coeffs, -coeff * yv)
            push!(aff_new.vars, yid); push!(aff_new.coeffs, -coeff * xv)
            constant += coeff * xv * yv
        end

        for k in 1:length(mv.qVarsId)
            xid = mv.qVarsId[k]
            xv = m.colVal[xid]
            coeff = alpha[k]

            bid = 0
            if isa(qbcId, Dict) && haskey(qbcId, (xid, xid))
                bid = qbcId[(xid, xid)][1]
            end
            bid == 0 && continue

            push!(aff_new.vars, bid); push!(aff_new.coeffs, coeff)
            push!(aff_new.vars, xid); push!(aff_new.coeffs, -2 * coeff * xv)
            constant += coeff * xv * xv
        end

        aff_new.constant = constant
        push!(m.linconstr, LinearConstraintData(aff_new, 0.0, Inf))
    end

    return length(m.linconstr) - oldncon
end
