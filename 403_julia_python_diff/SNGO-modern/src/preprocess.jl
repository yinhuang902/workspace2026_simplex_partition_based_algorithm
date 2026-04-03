"""
    Preprocessing for individual scenario models

Ported from Julia 0.6 preprocess.jl:
  - Builds the convex relaxation structure
  - Identifies bilinear terms and creates auxiliary variables
  - Classifies nonlinear constraints (exp, log, power, monomial)
  - Builds Q matrices for convexity detection
  - Creates RLT (Reformulation-Linearization Technique) constraints
"""

function preprocess!(mw::ModelWrapper)
    P = factorable!(mw)
    n = P.numCols

    # Squeeze quadratic constraints
    for qc in P.quadconstr
        squeeze!(qc.terms)
        # Ensure upper-triangular ordering of Q terms
        for j in 1:length(qc.terms.qvars1)
            if qc.terms.qvars1[j] > qc.terms.qvars2[j]
                qc.terms.qvars1[j], qc.terms.qvars2[j] = qc.terms.qvars2[j], qc.terms.qvars1[j]
            end
        end
        # Remove duplicate bilinear terms
        _remove_duplicate_quad_terms!(qc.terms)
    end

    # Create a simple relaxed problem
    m = ModelWrapper()
    m.numCols = P.numCols
    m.colNames = copy(P.colNames)
    m.colNamesIJulia = copy(P.colNamesIJulia)
    m.colLower = copy(P.colLower)
    m.colUpper = copy(P.colUpper)
    m.colCat = copy(P.colCat)
    m.colVal = copy(P.colVal)
    m.linconstr = [copy(c) for c in P.linconstr]
    m.obj = copy(P.obj)
    m.objSense = P.objSense
    m.linconstrDuals = zeros(length(m.linconstr))
    m.redCosts = zeros(m.numCols)

    # Identify bilinear terms and create auxiliary variables
    branchVarsId = Int[]
    qbVarsId = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    bilinearVars = Int[]

    for i in 1:length(P.quadconstr)
        con = P.quadconstr[i]
        terms = con.terms
        for j in 1:length(terms.qvars1)
            v1 = terms.qvars1[j]
            v2 = terms.qvars2[j]
            if !haskey(qbVarsId, (v1, v2))
                push!(branchVarsId, v1)
                push!(branchVarsId, v2)
                # Add bilinear auxiliary variable
                varname = "bilinear_con$(i)_$(v1)_$(v2)_$(j)"
                bid = add_variable!(m, -Inf, Inf, :Cont, varname, 0.0)
                push!(bilinearVars, bid)
                cid = length(m.linconstr) + 1

                # Add McCormick envelope constraints
                xl = P.colLower[v1]; xu = P.colUpper[v1]
                yl = P.colLower[v2]; yu = P.colUpper[v2]

                # w >= xl*y + yl*x - xl*yl  (underestimator 1)
                aff1 = AffExprData([bid, v1, v2], [1.0, -yl, -xl], -(-xl*yl))
                push!(m.linconstr, LinearConstraintData(aff1, -xl*yl, Inf))

                # w >= xu*y + yu*x - xu*yu  (underestimator 2)
                aff2 = AffExprData([bid, v1, v2], [1.0, -yu, -xu], -(-xu*yu))
                push!(m.linconstr, LinearConstraintData(aff2, -xu*yu, Inf))

                # w <= xl*y + yu*x - xl*yu  (overestimator 1)
                aff3 = AffExprData([bid, v1, v2], [1.0, -yu, -xl], -(-xl*yu))
                push!(m.linconstr, LinearConstraintData(aff3, -Inf, -xl*yu))

                # w <= xu*y + yl*x - xu*yl  (overestimator 2) — only if x != y
                if v1 != v2
                    aff4 = AffExprData([bid, v1, v2], [1.0, -yl, -xu], -(-xu*yl))
                    push!(m.linconstr, LinearConstraintData(aff4, -Inf, -xu*yl))
                end

                qbVarsId[(v1, v2)] = (bid, cid)
                qbVarsId[(v2, v1)] = (bid, cid)
            end
        end
    end
    branchVarsId = sort(unique(branchVarsId))

    # Process nonlinear constraints
    expVariable_list = Any[]
    logVariable_list = Any[]
    powerVariable_list = Any[]
    monomialVariable_list = Any[]

    for nlexpr in P.nlconstr
        if isexponentialcon(nlexpr)
            ev = parseexponentialcon(nlexpr)
            push!(expVariable_list, ev)
        elseif islogcon(nlexpr)
            ev = parselogcon(nlexpr)
            push!(logVariable_list, ev)
        elseif ispowercon(nlexpr)
            ev = parsepowercon(nlexpr)
            push!(powerVariable_list, ev)
        elseif ismonomialcon(nlexpr)
            ev = parsemonomialcon(nlexpr)
            push!(expVariable_list, ev)
        end
    end

    # Build MultiVariable structure — group connected bilinear terms
    Pcopy = copyModel(P)
    multiVariable_list = MultiVariableCon[]
    multiVariable_convex = MultiVariable[]
    multiVariable_aBB = MultiVariable[]

    for i in 1:length(Pcopy.quadconstr)
        con = Pcopy.quadconstr[i]
        terms = con.terms
        mvs = MultiVariable[]

        remain = collect(1:length(terms.qvars1))
        while length(remain) != 0
            mv = MultiVariable()
            push!(mvs, mv)

            seed = remain[1]
            push!(mv.terms.qvars1, terms.qvars1[seed])
            push!(mv.terms.qvars2, terms.qvars2[seed])
            push!(mv.terms.qcoeffs, terms.qcoeffs[seed])
            push!(mv.qVarsId, terms.qvars1[seed])
            push!(mv.qVarsId, terms.qvars2[seed])
            mv.qVarsId = sort(unique(mv.qVarsId))
            deleteat!(remain, 1)

            while length(remain) != 0
                added = Int[]
                for k in 1:length(remain)
                    j = remain[k]
                    v1 = terms.qvars1[j]
                    v2 = terms.qvars2[j]
                    if v1 in mv.qVarsId || v2 in mv.qVarsId
                        push!(mv.terms.qvars1, v1)
                        push!(mv.terms.qvars2, v2)
                        push!(mv.terms.qcoeffs, terms.qcoeffs[j])
                        push!(mv.qVarsId, v1)
                        push!(mv.qVarsId, v2)
                        mv.qVarsId = sort(unique(mv.qVarsId))
                        push!(added, k)
                    end
                end
                deleteat!(remain, added)
                if isempty(added)
                    break
                end
            end
        end

        # Associate affine terms with MultiVariables
        affcopy = copy(terms.aff)
        nonlinearIndexs = Int[]
        for j in 1:length(terms.aff.vars)
            var = terms.aff.vars[j]
            coeff = terms.aff.coeffs[j]
            idx = findfirst(mv -> var in mv.qVarsId, mvs)
            if idx !== nothing
                push!(mvs[idx].terms.aff.vars, var)
                push!(mvs[idx].terms.aff.coeffs, coeff)
            else
                push!(nonlinearIndexs, j)
            end
        end
        # Remove associated terms from the remaining affine expression
        keep = setdiff(1:length(affcopy.vars), nonlinearIndexs)
        affcopy.vars = affcopy.vars[keep]
        affcopy.coeffs = affcopy.coeffs[keep]

        # Build Q matrix & check convexity for each MultiVariable
        for mv in mvs
            C = length(mv.qVarsId)
            Q = zeros(Float64, C, C)
            for k in 1:length(mv.terms.qvars1)
                v1 = mv.terms.qvars1[k]
                v2 = mv.terms.qvars2[k]
                I = findfirst(x -> x == v1, mv.qVarsId)
                J = findfirst(x -> x == v2, mv.qVarsId)
                Q[I, J] = mv.terms.qcoeffs[k]

                if haskey(qbVarsId, (v1, v2))
                    bid = qbVarsId[(v1, v2)][1]
                    push!(mv.bilinearVarsId, bid)
                end
            end
            mv.Q = copy(Q)
            Q_sym = (Q + Q') / 2
            vals = eigvals(Q_sym)

            if all(v -> v >= 0, vals)
                mv.pd = 1
                push!(multiVariable_convex, mv)
            end
            if all(v -> v <= 0, vals)
                mv.pd = -1
                push!(multiVariable_convex, mv)
            end
            if all(v -> v == 0, vals)
                mv.pd = 0
            end

            # aBB relaxation for indefinite terms
            if mv.pd == 0
                alpha = zeros(Float64, C)
                lambda_min = minimum(vals)
                added = true

                if all(diag(Q_sym) .== 0) && length(mv.qVarsId) < 3
                    added = false
                end

                for k in 1:C
                    alpha[k] = max(0, min(-lambda_min,
                                   sum(abs.(Q_sym[k, :])) - 2 * Q_sym[k, k]))
                    varId = mv.qVarsId[k]
                    if !haskey(qbVarsId, (varId, varId))
                        added = false
                        break
                    end
                end

                if added
                    mv.alpha = alpha
                    push!(multiVariable_aBB, mv)
                end
            end
        end
        push!(multiVariable_list, MultiVariableCon(mvs, copy(affcopy)))
    end

    # Create RLT constraints
    EqVconstr = LinearConstraintData[]
    for i in 1:length(Pcopy.linconstr)
        con = Pcopy.linconstr[i]
        if con.lb == con.ub
            constant = con.lb
            for varId in branchVarsId
                newcon = copy(con)
                aff = newcon.terms
                accept = true
                for k in 1:length(aff.vars)
                    varIdinEq = aff.vars[k]
                    if haskey(qbVarsId, (varId, varIdinEq))
                        aff.vars[k] = qbVarsId[(varId, varIdinEq)][1]
                    else
                        accept = false
                        break
                    end
                end
                if accept
                    if constant != 0
                        push!(aff.vars, varId)
                        push!(aff.coeffs, -constant)
                        newcon.lb = 0.0
                        newcon.ub = 0.0
                    end
                    push!(EqVconstr, newcon)
                end
            end
        end
    end

    # Store the relaxed model in ext for later use
    m.ext[:qbcId] = qbVarsId

    pr = PreprocessResult()
    pr.branchVarsId = branchVarsId
    pr.qbVarsId = qbVarsId
    pr.EqVconstr = EqVconstr
    pr.multiVariable_list = multiVariable_list
    pr.multiVariable_convex = multiVariable_convex
    pr.multiVariable_aBB = multiVariable_aBB
    pr.expVariable_list = expVariable_list
    pr.logVariable_list = logVariable_list
    pr.powerVariable_list = powerVariable_list
    pr.monomialVariable_list = monomialVariable_list

    return pr, P
end

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

function _remove_duplicate_quad_terms!(terms::QuadExprData)
    j = 1
    while j <= length(terms.qvars1)
        v1 = terms.qvars1[j]
        v2 = terms.qvars2[j]
        dups = Int[]
        for k in (j+1):length(terms.qvars1)
            if (terms.qvars1[k] == v1 && terms.qvars2[k] == v2) ||
               (terms.qvars1[k] == v2 && terms.qvars2[k] == v1)
                push!(dups, k)
                terms.qcoeffs[j] += terms.qcoeffs[k]
            end
        end
        if !isempty(dups)
            deleteat!(terms.qvars1, dups)
            deleteat!(terms.qvars2, dups)
            deleteat!(terms.qcoeffs, dups)
        end
        j += 1
    end
end
