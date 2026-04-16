"""
    Stochastic Model Management

Replaces the old PlasmoOld module with a lightweight implementation
that provides scenario tree management for two-stage stochastic programs.

Key types:
  - `StochasticModel`: master + scenario tree structure
  - `RandomStochasticModel()`: create random stochastic perturbations

All scenario models are `ModelWrapper` instances from compat.jl.
"""

# ─────────────────────────────────────────────────────────────────────
# StochasticModel
# ─────────────────────────────────────────────────────────────────────

mutable struct NetData
    children::Vector{ModelWrapper}
    parent::Union{ModelWrapper, Nothing}
    childrenDict::Dict{String, ModelWrapper}
end
NetData() = NetData(ModelWrapper[], nothing, Dict{String, ModelWrapper}())

"""
    getchildren(P::ModelWrapper) -> Vector{ModelWrapper}

Get the scenario (child) models from a stochastic master model.
"""
function getchildren(P::ModelWrapper)
    if haskey(P.ext, :Net)
        return P.ext[:Net].children
    else
        return ModelWrapper[]
    end
end

"""
    getparent(P::ModelWrapper) -> ModelWrapper or Nothing
"""
function getparent(P::ModelWrapper)
    if haskey(P.ext, :Net)
        return P.ext[:Net].parent
    else
        return nothing
    end
end

"""
    NetModel() -> ModelWrapper

Create a new master model for a stochastic program.
"""
function NetModel()
    m = ModelWrapper()
    m.ext[:Net] = NetData()
    m.ext[:BuildType] = "serial"
    m.ext[:linkingId] = Int[]
    return m
end

"""
    addNode!(master, node, name)

Add a scenario node to the master model.
"""
function addNode!(master::ModelWrapper, node::ModelWrapper, name::String)
    if !haskey(node.ext, :Net)
        node.ext[:Net] = NetData(ModelWrapper[], master, Dict{String, ModelWrapper}())
    else
        node.ext[:Net].parent = master
    end
    push!(master.ext[:Net].children, node)
    master.ext[:Net].childrenDict[name] = node
end

"""
    getsumobjectivevalue(m::ModelWrapper) -> Float64

Sum up objective values across master and all scenarios.
"""
function getsumobjectivevalue(m::ModelWrapper)
    children = getchildren(m)
    objVal = m.objVal
    for node in children
        objVal += node.objVal
    end
    return objVal
end

# ─────────────────────────────────────────────────────────────────────
# Random Stochastic Model
# ─────────────────────────────────────────────────────────────────────

"""
    addnoise(a, adl, adu, rdl, rdu) -> Float64

Add random noise to a value `a`. If a==0, uses additive uniform noise [adl,adu].
Otherwise uses multiplicative noise: a + |a| * Uniform(rdl, rdu).
"""
function addnoise(a::Float64, adl::Float64, adu::Float64,
                  rdl::Float64, rdu::Float64)
    if a == 0.0
        return a + rand(Uniform(adl, adu))
    else
        return a + abs(a) * rand(Uniform(rdl, rdu))
    end
end

"""
    RandomStochasticModel(createModel, nscen; nfirst, nparam, rdl, rdu, adl, adu)

Create a stochastic model with `nscen` scenarios by calling `createModel()`
and adding random perturbations to second-stage constraints/bounds.

Arguments:
  - `createModel`: function() -> JuMP.Model  (returns a deterministic scenario model)
  - `nscen`: number of scenarios
  - `nfirst`: number of first-stage variables (auto-detected from model)
  - `nparam`: number of parameters to perturb
  - `rdl, rdu`: relative perturbation bounds
  - `adl, adu`: additive perturbation bounds

Returns the master ModelWrapper with scenarios as children.
"""
function RandomStochasticModel(createModel::Function, nscen::Int=100;
                               nfirst::Int=0, nparam=5,
                               rdl::Float64=0.0, rdu::Float64=2.0,
                               adl::Float64=-10.0, adu::Float64=10.0)
    Random.seed!(1234)

    master = NetModel()

    # Create first scenario to determine nfirst
    node1_jump = createModel()
    node1 = import_jump_model(node1_jump)
    if nfirst == 0
        nfirst = node1.numCols
    end

    # Add first-stage variables to master
    for i in 1:nfirst
        add_variable!(master, node1.colLower[i], node1.colUpper[i],
                      node1.colCat[i], node1.colNames[i],
                      isnan(node1.colVal[i]) ? 0.0 : node1.colVal[i])
    end

    firstVarsId = collect(1:nfirst)

    for i in 1:nscen
        # Create scenario model
        node_jump = createModel()
        node = import_jump_model(node_jump)

        # Store first-stage variable mapping
        node.ext[:firstVarsId] = collect(1:nfirst)

        # Add to master
        addNode!(master, node, "s$i")

        # Add linking constraints: master.first[j] == node.vars[j]
        # These are stored as references in the master's linking ID list
        for j in 1:nfirst
            if firstVarsId[j] > 0
                # Store linking constraint info
                # (In the old code these were actual JuMP constraints; here we track them)
                lc_aff = AffExprData(Int[], Float64[], 0.0)
                # master var j has coeff +1, scenario var j has coeff -1
                # We track this as: master_col=j, scenario_idx=i, scenario_col=j
                push!(lc_aff.vars, j)  # master variable
                push!(lc_aff.coeffs, 1.0)
                push!(master.linconstr, LinearConstraintData(lc_aff, 0.0, 0.0))
                # Mark as linking
                push!(master.ext[:linkingId], length(master.linconstr))
            end
        end

        # Skip perturbation for first scenario
        if i == 1
            continue
        end

        # Apply perturbations to second-stage constraints
        nmodified = 0

        # Perturb quadratic constraints
        if isa(nparam, Int) && nmodified < div(nparam, 2)
            for c in 1:length(node.quadconstr)
                con = node.quadconstr[c]
                terms = con.terms
                # Collect variable IDs in this constraint
                varsId = sort(unique([terms.qvars1; terms.qvars2;
                                     terms.aff.vars]))
                # Skip first-stage-only constraints
                if all(v -> v in firstVarsId, varsId)
                    continue
                end
                if isa(nparam, Int) && nmodified >= nparam
                    break
                end

                if con.sense == :(==)
                    # Split equality into two inequalities with noise
                    connew = copy(con)
                    connew.terms.aff.constant = addnoise(connew.terms.aff.constant,
                                                         0.0, adu, rdl, rdu)
                    connew.sense = :(>=)
                    con.terms.aff.constant = addnoise(con.terms.aff.constant,
                                                      adl, 0.0, -rdu, -rdl)
                    con.sense = :(<=)
                    push!(node.quadconstr, connew)
                elseif con.sense == :(>=)
                    con.terms.aff.constant = addnoise(con.terms.aff.constant,
                                                      0.0, adu, rdl, rdu)
                elseif con.sense == :(<=)
                    con.terms.aff.constant = addnoise(con.terms.aff.constant,
                                                      adl, 0.0, -rdu, -rdl)
                end
                nmodified += 1
            end
        end

        # Perturb linear constraints
        for c in 1:length(node.linconstr)
            con = node.linconstr[c]
            terms = con.terms
            varsId = sort(unique(terms.vars))
            if all(v -> v in firstVarsId, varsId)
                continue
            end
            if isa(nparam, Int) && nmodified >= nparam
                break
            end

            if con.lb == con.ub
                # Split equality
                connew = copy(con)
                connew.lb = addnoise(connew.lb, adl, 0.0, -rdu, -rdl)
                connew.ub = Inf
                con.ub = addnoise(con.ub, 0.0, adu, rdl, rdu)
                con.lb = -Inf
                push!(node.linconstr, connew)
            elseif con.lb == -Inf
                con.ub = addnoise(con.ub, 0.0, adu, rdl, rdu)
            elseif con.ub == Inf
                con.lb = addnoise(con.lb, adl, 0.0, -rdu, -rdl)
            end
            nmodified += 1
        end

        # Perturb variable bounds
        for v in 1:node.numCols
            if isa(nparam, Int) && nmodified >= nparam
                break
            end
            if v in firstVarsId
                continue
            end
            node.colLower[v] = addnoise(node.colLower[v], adl, 0.0, -rdu, -rdl)
            node.colUpper[v] = addnoise(node.colUpper[v], 0.0, adu, rdl, rdu)
            nmodified += 1
        end

        if i == 1 && isa(nparam, Int) && nmodified < nparam
            println("warning: the number of linear/quadratic second stage constraint  ",
                    nmodified, " is less than nparam ", nparam)
        end
    end

    return master
end

# ─────────────────────────────────────────────────────────────────────
# Copy Stochastic Model
# ─────────────────────────────────────────────────────────────────────

"""
    copyStoModel(P::ModelWrapper) -> ModelWrapper

Deep copy a stochastic model (master + all scenarios).
"""
function copyStoModel(P::ModelWrapper)
    if !haskey(P.ext, :Net)
        return copyModel(P)
    end

    m = NetModel()
    children = getchildren(P)
    nscen = length(children)

    for scen in 1:nscen
        modelname = "s$scen"
        nodecopy = copyModel(children[scen])
        addNode!(m, nodecopy, modelname)
    end

    # Copy master data
    m.numCols = P.numCols
    m.colLower = copy(P.colLower)
    m.colUpper = copy(P.colUpper)
    m.colVal = copy(P.colVal)
    m.colCat = copy(P.colCat)
    m.colNames = copy(P.colNames)
    m.colNamesIJulia = copy(P.colNamesIJulia)
    m.linconstr = [copy(c) for c in P.linconstr]
    m.obj = copy(P.obj)
    m.objSense = P.objSense

    # Copy linking ID list
    if haskey(P.ext, :linkingId)
        m.ext[:linkingId] = copy(P.ext[:linkingId])
    end

    return m
end

# ─────────────────────────────────────────────────────────────────────
# Extensive Form Construction
# ─────────────────────────────────────────────────────────────────────

"""
    extensiveSimplifiedModel(P::ModelWrapper) -> ModelWrapper

Build the extensive form (EF) from the stochastic model:
  - Combine all scenario variables into one model
  - First-stage variables shared across scenarios
  - Second-stage variables are scenario-specific copies
  - Linear, quadratic, and NL constraints from each scenario included
  - Objective sums over all scenarios
"""
function extensiveSimplifiedModel(P::ModelWrapper)
    ncols_first = P.numCols
    scenarios = getchildren(P)
    nscen = length(scenarios)

    # Set up firstVarsId for each scenario
    for (idx, scenario) in enumerate(scenarios)
        if !haskey(scenario.ext, :firstVarsId)
            scenario.ext[:firstVarsId] = collect(1:ncols_first)
        end
    end

    # Sync first-stage bounds from scenarios to master
    for (idx, scenario) in enumerate(scenarios)
        firstVarsId = scenario.ext[:firstVarsId]
        for i in 1:ncols_first
            fvi = firstVarsId[i]
            if fvi > 0
                if scenario.colCat[fvi] == :Bin || scenario.colCat[fvi] == :Int
                    P.colCat[i] = scenario.colCat[fvi]
                end
                if scenario.colLower[fvi] > P.colLower[i]
                    P.colLower[i] = scenario.colLower[fvi]
                end
                if scenario.colUpper[fvi] < P.colUpper[i]
                    P.colUpper[i] = scenario.colUpper[fvi]
                end
            end
        end
    end

    # Build extensive form model
    m = ModelWrapper()
    m.ext[:v_map] = Vector{Vector{Int}}()

    # Add first-stage variables
    for i in 1:ncols_first
        add_variable!(m, P.colLower[i], P.colUpper[i],
                      P.colCat[i], P.colNames[i],
                      isnan(P.colVal[i]) ? 0.0 : P.colVal[i])
    end

    # Copy master objective
    m.obj = copy(P.obj)
    m.objSense = P.objSense

    ncols = Vector{Int}(undef, nscen + 1)
    nlinconstrs = Vector{Int}(undef, nscen + 1)
    ncols[1] = m.numCols
    nlinconstrs[1] = length(m.linconstr)

    # Add each scenario
    for scen in 1:nscen
        node = scenarios[scen]
        firstVarsId = node.ext[:firstVarsId]

        num_vars = node.numCols
        v_map = Vector{Int}(undef, num_vars)

        # Add scenario-specific variables (non-first-stage)
        for i in 1:num_vars
            first_idx = findfirst(x -> x == i, firstVarsId)
            if first_idx === nothing
                # Second-stage variable: add as new column
                col = add_variable!(m, node.colLower[i], node.colUpper[i],
                                    node.colCat[i],
                                    "s$(scen)_" * node.colNames[i],
                                    isnan(node.colVal[i]) ? 0.0 : node.colVal[i])
                v_map[i] = col
            else
                # First-stage variable: map to existing master column
                v_map[i] = first_idx
            end
        end
        push!(m.ext[:v_map], v_map)

        # Add scenario linear constraints (skip first-stage-only for scen>1)
        for i in 1:length(node.linconstr)
            con = node.linconstr[i]
            varsId = sort(unique(con.terms.vars))
            if all(v -> findfirst(x -> x == v, firstVarsId) !== nothing, varsId) && scen != 1
                continue
            end
            push!(m.linconstr, copy_lincon_with_map(con, v_map))
        end

        # Add scenario quadratic constraints
        for i in 1:length(node.quadconstr)
            con = node.quadconstr[i]
            varsId = sort(unique([con.terms.qvars1; con.terms.qvars2; con.terms.aff.vars]))
            if all(v -> findfirst(x -> x == v, firstVarsId) !== nothing, varsId) && scen != 1
                continue
            end
            push!(m.quadconstr, copy_quadcon_with_map(con, v_map))
        end

        # Add NL constraints (with remapped variable indices)
        for nlexpr in node.nlconstr
            new_expr = _remap_nl_vars(deepcopy(nlexpr), v_map)
            push!(m.nlconstr, new_expr)
        end

        # Accumulate objective terms from scenario
        node_obj = copy(node.obj)
        # Remap quadratic terms
        for k in 1:length(node_obj.qvars1)
            node_obj.qvars1[k] = v_map[node_obj.qvars1[k]]
            node_obj.qvars2[k] = v_map[node_obj.qvars2[k]]
        end
        # Remap affine terms
        for k in 1:length(node_obj.aff.vars)
            node_obj.aff.vars[k] = v_map[node_obj.aff.vars[k]]
        end
        # Add to master objective
        append!(m.obj.qvars1, node_obj.qvars1)
        append!(m.obj.qvars2, node_obj.qvars2)
        append!(m.obj.qcoeffs, node_obj.qcoeffs)
        append!(m.obj.aff.vars, node_obj.aff.vars)
        append!(m.obj.aff.coeffs, node_obj.aff.coeffs)
        m.obj.aff.constant += node_obj.aff.constant

        ncols[scen + 1] = m.numCols
        nlinconstrs[scen + 1] = length(m.linconstr)
    end

    m.ext[:ncols] = ncols
    m.ext[:nlinconstrs] = nlinconstrs

    # Initialize solution vectors
    m.linconstrDuals = zeros(length(m.linconstr))
    m.redCosts = zeros(m.numCols)

    return m
end

"""
    _remap_nl_vars(expr, v_map)

Remap variable indices in a nonlinear expression tree.
"""
function _remap_nl_vars(expr, v_map::Vector{Int})
    if isa(expr, Expr)
        if expr.head == :ref
            old_idx = expr.args[2]
            if isa(old_idx, Int) && old_idx >= 1 && old_idx <= length(v_map)
                expr.args[2] = v_map[old_idx]
            end
            return expr
        else
            for i in 1:length(expr.args)
                expr.args[i] = _remap_nl_vars(expr.args[i], v_map)
            end
            return expr
        end
    else
        return expr
    end
end

# ─────────────────────────────────────────────────────────────────────
# Bound Sync Utilities (used in BnB)
# ─────────────────────────────────────────────────────────────────────

"""
    updateStoFirstBounds!(P::ModelWrapper)

Synchronize first-stage variable bounds across all scenarios:
take the tightest (max lower, min upper) bounds from any scenario.
"""
function updateStoFirstBounds!(P::ModelWrapper)
    scenarios = getchildren(P)
    nfirst = P.numCols
    for (idx, scenario) in enumerate(scenarios)
        firstVarsId = scenario.ext[:firstVarsId]
        for i in 1:nfirst
            fvi = firstVarsId[i]
            if fvi > 0
                if scenario.colLower[fvi] > P.colLower[i]
                    P.colLower[i] = scenario.colLower[fvi]
                end
                if scenario.colUpper[fvi] < P.colUpper[i]
                    P.colUpper[i] = scenario.colUpper[fvi]
                end
            end
        end
    end
    # Propagate back to scenarios
    for (idx, scenario) in enumerate(scenarios)
        firstVarsId = scenario.ext[:firstVarsId]
        for i in 1:nfirst
            fvi = firstVarsId[i]
            if fvi > 0
                scenario.colLower[fvi] = P.colLower[i]
                scenario.colUpper[fvi] = P.colUpper[i]
            end
        end
    end
end

"""
    updateExtensiveBoundsFromSto!(P, Pex)

Copy first-stage bounds from the stochastic model to the extensive form.
"""
function updateExtensiveBoundsFromSto!(P::ModelWrapper, Pex::ModelWrapper)
    for i in 1:P.numCols
        if i <= Pex.numCols
            Pex.colLower[i] = P.colLower[i]
            Pex.colUpper[i] = P.colUpper[i]
        end
    end
end

"""
    updateStoBoundsFromExtensive!(Pex, P)

Copy first-stage bounds from the extensive form back to the stochastic model.
"""
function updateStoBoundsFromExtensive!(Pex::ModelWrapper, P::ModelWrapper)
    for i in 1:P.numCols
        if i <= Pex.numCols
            P.colLower[i] = Pex.colLower[i]
            P.colUpper[i] = Pex.colUpper[i]
        end
    end
    # Also update scenarios
    scenarios = getchildren(P)
    for scenario in scenarios
        firstVarsId = scenario.ext[:firstVarsId]
        for i in 1:P.numCols
            fvi = firstVarsId[i]
            if fvi > 0
                scenario.colLower[fvi] = P.colLower[i]
                scenario.colUpper[fvi] = P.colUpper[i]
            end
        end
    end
end

"""
    updateFirstBounds!(P, xl, xu)

Set first-stage bounds to the given arrays.
"""
function updateFirstBounds!(P::ModelWrapper, xl::Vector{Float64}, xu::Vector{Float64})
    nfirst = P.numCols
    for i in 1:nfirst
        P.colLower[i] = xl[i]
        P.colUpper[i] = xu[i]
    end
end

"""
    copyNLModel(P::ModelWrapper) -> ModelWrapper

Copy a model, converting quadratic constraints to nonlinear form.
(In modern JuMP, this distinction is less important.)
"""
function copyNLModel(P::ModelWrapper)
    # For modern JuMP, we can just deep-copy since we handle
    # quadratic and NL constraints uniformly
    return copyModel(P)
end

# ─────────────────────────────────────────────────────────────────────
# Additional stochastic helpers (ported from original)
# ─────────────────────────────────────────────────────────────────────

"""
    Ipopt_solve(P) -> Symbol

Solve each scenario of the stochastic model P independently with Ipopt.
"""
function Ipopt_solve(P::ModelWrapper)
    scenarios = getchildren(P)
    all_optimal = true
    for scenario in scenarios
        scen_copy = copyModel(scenario)
        status = solve_model!(scen_copy, JuMP.optimizer_with_attributes(
            Ipopt.Optimizer, "print_level" => 0, "max_cpu_time" => 100.0))
        if status == :Optimal
            scenario.colVal = copy(scen_copy.colVal)
            scenario.objVal = scen_copy.objVal
        else
            all_optimal = false
        end
    end
    return all_optimal ? :Optimal : :NotOptimal
end

"""
    getsumobjectivevalue(P) -> Float64

Sum objective values across all scenarios.
"""
function getsumobjectivevalue(P::ModelWrapper)
    scenarios = getchildren(P)
    return sum(s.objVal for s in scenarios)
end

"""
    updateStoBoundsFromSto!(P, Q)

Copy bounds from stochastic model P to stochastic model Q for all scenarios.
"""
function updateStoBoundsFromSto!(P::ModelWrapper, Q::ModelWrapper)
    # Copy first-stage bounds
    nfirst = P.numCols
    Q.colLower[1:nfirst] = copy(P.colLower[1:nfirst])
    Q.colUpper[1:nfirst] = copy(P.colUpper[1:nfirst])

    # Copy scenario bounds
    scenarios_P = getchildren(P)
    scenarios_Q = getchildren(Q)
    for (idx, (sp, sq)) in enumerate(zip(scenarios_P, scenarios_Q))
        sq.colLower = copy(sp.colLower)
        sq.colUpper = copy(sp.colUpper)
    end
end

"""
    updateStoSolFromExtensive!(Pex, P)

Copy solution from extensive form back to stochastic model scenarios.
"""
function updateStoSolFromExtensive!(Pex::ModelWrapper, P::ModelWrapper)
    scenarios = getchildren(P)
    nfirst = P.numCols
    offset = nfirst
    for scenario in scenarios
        n_scen = scenario.numCols
        for i in 1:n_scen
            if offset + i <= Pex.numCols
                scenario.colVal[i] = Pex.colVal[offset + i]
            end
        end
        offset += n_scen
    end
end

"""
    updateStoSolFromSto!(P, Q)

Copy solutions from stochastic model P to stochastic model Q.
"""
function updateStoSolFromSto!(P::ModelWrapper, Q::ModelWrapper)
    Q.colVal[1:P.numCols] = copy(P.colVal[1:P.numCols])
    scenarios_P = getchildren(P)
    scenarios_Q = getchildren(Q)
    for (sp, sq) in zip(scenarios_P, scenarios_Q)
        sq.colVal = copy(sp.colVal)
    end
end

"""
    updateFirstBounds!(P, lb, ub, varId)

Update first-stage bounds in all scenarios for a single variable.
"""
function updateFirstBounds!(P::ModelWrapper, lb::Float64, ub::Float64, varId::Int)
    P.colLower[varId] = lb
    P.colUpper[varId] = ub
    scenarios = getchildren(P)
    for scenario in scenarios
        firstVarsId = scenario.ext[:firstVarsId]
        fvi = firstVarsId[varId]
        if fvi > 0
            scenario.colLower[fvi] = lb
            scenario.colUpper[fvi] = ub
        end
    end
end

"""
    updateExtensiveFirstBounds!(Pex, P, lb, ub, varId)

Update extensive form bounds for a single first-stage variable.
"""
function updateExtensiveFirstBounds!(Pex::ModelWrapper, P::ModelWrapper,
                                      lb::Float64, ub::Float64, varId::Int)
    Pex.colLower[varId] = lb
    Pex.colUpper[varId] = ub

    # Update linked variables in extensive form
    scenarios = getchildren(P)
    if haskey(Pex.ext, :scenarioOffsets)
        offsets = Pex.ext[:scenarioOffsets]
        for (idx, scenario) in enumerate(scenarios)
            firstVarsId = scenario.ext[:firstVarsId]
            fvi = firstVarsId[varId]
            if fvi > 0 && idx <= length(offsets)
                ex_idx = offsets[idx] + fvi
                if ex_idx <= Pex.numCols
                    Pex.colLower[ex_idx] = lb
                    Pex.colUpper[ex_idx] = ub
                end
            end
        end
    end
end

