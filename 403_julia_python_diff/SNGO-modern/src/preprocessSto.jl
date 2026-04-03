"""
    Stochastic Preprocessing

Ported from preprocessSto.jl — preprocesses the stochastic model:
  1. Links first-stage variables between master and scenarios
  2. Sets default bounds on unbounded variables
  3. Calls preprocess! on each scenario
"""

function preprocessSto!(P::ModelWrapper)
    scenarios = getchildren(P)

    # Provide initial values if not defined
    for i in 1:length(P.colVal)
        if isnan(P.colVal[i])
            P.colVal[i] = 0.0
        end
    end

    # Find index of first-stage variables in each scenario
    ncols_first = P.numCols
    for (idx, scenario) in enumerate(scenarios)
        if !haskey(scenario.ext, :firstVarsId)
            scenario.ext[:firstVarsId] = collect(1:ncols_first)
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
    for (idx, scenario) in enumerate(scenarios)
        for i in 1:scenario.numCols
            if scenario.colLower[i] == -Inf
                scenario.colLower[i] = default_lower_bound_value
            end
            if scenario.colUpper[i] == Inf
                scenario.colUpper[i] = default_upper_bound_value
            end
        end
        if debug
            println("first:  ", scenario.ext[:firstVarsId])
        end
    end

    updateStoFirstBounds!(P)

    # Preprocess each scenario
    nscen = length(scenarios)
    pr_children = Any[]
    for (idx, scen) in enumerate(scenarios)
        pr, scenarios[idx] = preprocess!(scen)
        push!(pr_children, pr)
    end

    return pr_children
end
