"""
    Extensive Form Preprocessing

Ported from preprocessex.jl — preprocesses the extensive form model.
Similar to preprocess! but handles multi-node structure where variables
and constraints span multiple scenarios.
"""

function preprocessex!(P::ModelWrapper)
    # Provide initial values
    for i in 1:length(P.colVal)
        if isnan(P.colVal[i])
            P.colVal[i] = 0.0
        end
    end

    # If objective is quadratic, move to constraints
    if length(P.obj.qvars1) > 0
        obj_val = eval_quad(P.obj, P.colVal)
        obj_col = add_variable!(P, -Inf, Inf, :Cont, "objective_value", obj_val)

        if P.objSense == :Min
            # objective_value >= quad_expr  →  0 >= quad_expr - objective_value
            new_q = copy(P.obj)
            push!(new_q.aff.vars, obj_col)
            push!(new_q.aff.coeffs, -1.0)
            add_quad_constraint!(P, new_q, :(<=))
            P.obj = QuadExprData(Int[], Int[], Float64[],
                                 AffExprData([obj_col], [1.0], 0.0))
        else
            new_q = copy(P.obj)
            push!(new_q.aff.vars, obj_col)
            push!(new_q.aff.coeffs, -1.0)
            add_quad_constraint!(P, new_q, :(>=))
            P.obj = QuadExprData(Int[], Int[], Float64[],
                                 AffExprData([obj_col], [1.0], 0.0))
        end
    end

    # Provide bounds if not defined
    for i in 1:P.numCols
        if P.colLower[i] == -Inf
            P.colLower[i] = default_lower_bound_value
        end
        if P.colUpper[i] == Inf
            P.colUpper[i] = default_upper_bound_value
        end
    end

    # Upper-triangularize quadratic constraints and remove duplicates
    for qc in P.quadconstr
        for j in 1:length(qc.terms.qvars1)
            if qc.terms.qvars1[j] > qc.terms.qvars2[j]
                qc.terms.qvars1[j], qc.terms.qvars2[j] = qc.terms.qvars2[j], qc.terms.qvars1[j]
            end
        end
        _remove_duplicate_quad_terms!(qc.terms)
    end

    # Create relaxed problem
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

    # Bilinear term identification — using Dict per column
    branchVarsId = Int[]
    qbVarsId = [Dict{Int, Int}() for _ in 1:m.numCols]
    bilinearVars = Int[]

    for i in 1:length(P.quadconstr)
        con = P.quadconstr[i]
        terms = con.terms
        for j in 1:length(terms.qvars1)
            v1 = terms.qvars1[j]
            v2 = terms.qvars2[j]
            if v1 <= length(qbVarsId) && !haskey(qbVarsId[v1], v2)
                push!(branchVarsId, v1)
                push!(branchVarsId, v2)
                varname = "bilinear_con$(i)_$(v1)_$(v2)_$(j)"
                bid = add_variable!(m, -Inf, Inf, :Cont, varname, 0.0)
                push!(bilinearVars, bid)
                # Extend qbVarsId if needed
                while length(qbVarsId) < m.numCols
                    push!(qbVarsId, Dict{Int, Int}())
                end
                qbVarsId[v1][v2] = bid
                qbVarsId[v2][v1] = bid
            end
        end
    end
    branchVarsId = sort(unique(branchVarsId))
    println("No. of nonlinear variables  ", length(branchVarsId))
    println("No. of nonlinear Terms  ", length(bilinearVars))

    # Process NL constraints
    expVariable_list = Any[]
    logVariable_list = Any[]
    powerVariable_list = Any[]
    monomialVariable_list = Any[]

    for nlexpr in P.nlconstr
        if isexponentialcon(nlexpr)
            push!(expVariable_list, parseexponentialcon(nlexpr))
        elseif islogcon(nlexpr)
            push!(logVariable_list, parselogcon(nlexpr))
        elseif ispowercon(nlexpr)
            push!(powerVariable_list, parsepowercon(nlexpr))
        elseif ismonomialcon(nlexpr)
            push!(expVariable_list, parsemonomialcon(nlexpr))
        end
    end

    # Build MultiVariable structure
    Pcopy = copyModel(P)
    multiVariable_list = MultiVariableCon[]
    multiVariable_convex = MultiVariable[]
    multiVariable_aBB = MultiVariable[]

    for i in 1:length(Pcopy.quadconstr)
        con = Pcopy.quadconstr[i]
        terms = con.terms
        mvs = MultiVariable[]

        remain = collect(1:length(terms.qvars1))
        while !isempty(remain)
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

            while !isempty(remain)
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
                isempty(added) && break
            end
        end

        # Associate affine terms
        affcopy = copy(terms.aff)
        nonlinearIndexs = Int[]
        for j in 1:length(terms.aff.vars)
            var_col = terms.aff.vars[j]
            coeff = terms.aff.coeffs[j]
            idx = findfirst(mv -> var_col in mv.qVarsId, mvs)
            if idx !== nothing
                push!(mvs[idx].terms.aff.vars, var_col)
                push!(mvs[idx].terms.aff.coeffs, coeff)
            else
                push!(nonlinearIndexs, j)
            end
        end
        keep = setdiff(1:length(affcopy.vars), nonlinearIndexs)
        affcopy.vars = affcopy.vars[keep]
        affcopy.coeffs = affcopy.coeffs[keep]

        # Build Q and check convexity
        for mv in mvs
            C = length(mv.qVarsId)
            Q = zeros(Float64, C, C)
            for k in 1:length(mv.terms.qvars1)
                v1 = mv.terms.qvars1[k]
                v2 = mv.terms.qvars2[k]
                I_idx = findfirst(x -> x == v1, mv.qVarsId)
                J_idx = findfirst(x -> x == v2, mv.qVarsId)
                Q[I_idx, J_idx] = mv.terms.qcoeffs[k]
                if v1 <= length(qbVarsId) && haskey(qbVarsId[v1], v2)
                    bid = qbVarsId[v1][v2]
                    push!(mv.bilinearVarsId, bid)
                end
            end
            mv.Q = copy(Q)
            Q_sym = (Q + Q') / 2
            vals = eigvals(Q_sym)

            all(v -> v >= 0, vals) && (mv.pd = 1; push!(multiVariable_convex, mv))
            all(v -> v <= 0, vals) && (mv.pd = -1; push!(multiVariable_convex, mv))
            all(v -> v == 0, vals) && (mv.pd = 0)

            if mv.pd == 0
                alpha = zeros(Float64, C)
                lambda_min = minimum(vals)
                added_flag = true
                all(diag(Q_sym) .== 0) && length(mv.qVarsId) < 3 && (added_flag = false)
                for k in 1:C
                    alpha[k] = max(0, min(-lambda_min, sum(abs.(Q_sym[k, :])) - 2*Q_sym[k,k]))
                    varId = mv.qVarsId[k]
                    if varId > length(qbVarsId) || !haskey(qbVarsId[varId], varId)
                        added_flag = false
                        break
                    end
                end
                if added_flag
                    mv.alpha = alpha
                    push!(multiVariable_aBB, mv)
                end
            end
        end
        push!(multiVariable_list, MultiVariableCon(mvs, copy(affcopy)))
    end

    # Create RLT constraints
    EqVconstr = LinearConstraintData[]
    if haskey(P.ext, :nlinconstrs)
        nlinconstrs = P.ext[:nlinconstrs]
        nnode = length(nlinconstrs)
        ncols = P.ext[:ncols]
        branchVarsIdNodes = Vector{Vector{Int}}()

        start_index = 1
        for nodeid in 1:nnode
            branchVarsId_local = Int[]
            Idend = ncols[nodeid]
            for i in start_index:length(branchVarsId)
                if branchVarsId[i] <= Idend
                    push!(branchVarsId_local, branchVarsId[i])
                else
                    start_index = i
                    break
                end
            end
            push!(branchVarsIdNodes, branchVarsId_local)
        end

        for nodeid in 1:nnode
            startcon = nodeid == 1 ? 1 : nlinconstrs[nodeid-1] + 1
            endcon = nlinconstrs[nodeid]
            (endcon - startcon) < 0 && continue
            varsId = branchVarsIdNodes[nodeid]

            for i in startcon:min(endcon, length(Pcopy.linconstr))
                con = Pcopy.linconstr[i]
                if con.lb == con.ub
                    constant = con.lb
                    for varId in varsId
                        newcon = copy(con)
                        aff = newcon.terms
                        accept = true
                        for k in 1:length(aff.vars)
                            varIdinEq = aff.vars[k]
                            if varId <= length(qbVarsId) && haskey(qbVarsId[varId], varIdinEq)
                                aff.vars[k] = qbVarsId[varId][varIdinEq]
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
        end
    end

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

    return pr
end
