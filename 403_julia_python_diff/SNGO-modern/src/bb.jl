"""
    Branch-and-Bound Main Loop

Ported from bb.jl (~1365 lines). Contains:
  - `branch_bound(P)`: main spatial BnB loop
  - Lower bounding via convex relaxation + WS bound
  - Upper bounding via Ipopt local solve + WSfix global solve
  - Node selection (best-bound)
  - Branching (pseudocost branching + solution-weighted split point)
"""

using Printf

# ─────────────────────────────────────────────────────────────────────
# Solver factories — configure once here
# ─────────────────────────────────────────────────────────────────────

function gurobi_lp_optimizer()
    optimizer = Gurobi.Optimizer
    return JuMP.optimizer_with_attributes(optimizer,
        "Method" => 2,          # Barrier
        "Threads" => 1,
        "Crossover" => 0,
        "OutputFlag" => 0,
        "DualReductions" => 0
    )
end

function gurobi_simplex_optimizer()
    optimizer = Gurobi.Optimizer
    return JuMP.optimizer_with_attributes(optimizer,
        "Method" => 1,          # Simplex
        "Threads" => 1,
        "OutputFlag" => 0,
        "DualReductions" => 0
    )
end

function scip_optimizer(gap::Float64=mingap/2, timelimit::Float64=100.0)
    optimizer = SCIP.Optimizer
    return JuMP.optimizer_with_attributes(optimizer,
        "display/verblevel" => 0,
        "limits/gap" => gap,
        "limits/time" => timelimit
    )
end

function ipopt_optimizer(timelimit::Float64=100.0)
    optimizer = Ipopt.Optimizer
    return JuMP.optimizer_with_attributes(optimizer,
        "max_cpu_time" => timelimit,
        "print_level" => 0
    )
end

# ─────────────────────────────────────────────────────────────────────
# Branch value computation
# ─────────────────────────────────────────────────────────────────────

"""
    computeBvalue(xl, xu, bValue, cat) -> Float64

Compute the branching split point as a weighted combination of
midpoint and best-known solution point.
"""
function computeBvalue(xl::Float64, xu::Float64, bValue::Float64,
                       cat::Symbol=:Cont)
    if cat == :Bin || cat == :Int
        return cat == :Bin ? 0.5 : floor((xl + xu) / 2)
    end

    lambda = 0.25
    alpha_bound = 0.1
    mid = (xl + xu) / 2
    bval = lambda * mid + (1 - lambda) * bValue
    # Clamp to stay at least alpha_bound*range from bounds
    range = xu - xl
    bval = max(bval, xl + alpha_bound * range)
    bval = min(bval, xu - alpha_bound * range)
    return bval
end

# ─────────────────────────────────────────────────────────────────────
# Lower Bound: Relaxation (iterative OA — full version)
# ─────────────────────────────────────────────────────────────────────

"""
    getRelaxLowerBound(Rold, Pex, prex, UB, defaultLB; nsolve=20) -> (status, LB, R)

Solve the LP relaxation iteratively:
  1. Solve LP
  2. Add OA + aBB cuts at current solution
  3. Re-solve (up to `nsolve` iterations or until no improvement)
"""
function getRelaxLowerBound(Rold::ModelWrapper, Pex::ModelWrapper,
                            prex::PreprocessResult, UB::Float64,
                            defaultLB::Float64;
                            relaxBin::Bool=false, nsolve::Int=20)
    R = updaterelax(Rold, Pex, prex, UB)
    Rx = copy(R.colVal)
    no_iter_wo_change = 0

    if relaxBin
        hasBin = any(c -> c == :Bin, R.colCat[1:R.numCols])
        if hasBin
            for i in 1:R.numCols
                if R.colCat[i] == :Bin
                    R.colCat[i] = :Cont
                end
            end
        end
    end

    relaxed_status = solve_model!(R, gurobi_simplex_optimizer())
    relaxed_LB = getRobjective(R, relaxed_status, defaultLB)

    if relaxed_status == :Optimal &&
       ((UB - relaxed_LB) <= mingap ||
        (UB - relaxed_LB) <= mingap * min(abs(relaxed_LB), abs(UB)))
        return (relaxed_status, relaxed_LB, R)
    end

    if relaxed_status == :Optimal
        Rx = copy(R.colVal)
        for i in 1:nsolve
            newncon_convex = addOuterApproximation!(R, prex)
            newncon_aBB = addaBB!(R, prex)
            addMonomialOA!(R, prex)
            addPowerOA!(R, prex)

            if (newncon_aBB + newncon_convex) != 0
                relaxed_status_trial = solve_model!(R, gurobi_simplex_optimizer())
                old_relaxed_LB = relaxed_LB
                relaxed_LB_trial = getRobjective(R, relaxed_status_trial, old_relaxed_LB)

                if relaxed_status_trial == :Optimal
                    Rx = copy(R.colVal)
                    relaxed_LB = relaxed_LB_trial
                    if (UB - relaxed_LB) <= mingap ||
                       (UB - relaxed_LB) <= mingap * min(abs(relaxed_LB), abs(UB))
                        return (:Optimal, relaxed_LB, R)
                    end
                elseif relaxed_status_trial == :Infeasible
                    relaxed_status = :Infeasible
                    relaxed_LB = UB
                    break
                else
                    println("warning: relaxation status = ", relaxed_status_trial)
                    break
                end

                Limprove = relaxed_LB_trial - old_relaxed_LB
                if Limprove >= LP_improve_tol
                    no_iter_wo_change = 0
                else
                    no_iter_wo_change += 1
                end
                if no_iter_wo_change >= 10
                    break
                end
            else
                break
            end
        end
    end

    R.colVal = Rx
    return (relaxed_status, relaxed_LB, R)
end

# ─────────────────────────────────────────────────────────────────────
# Lower Bound: Relaxation + Reduced-Cost BT (ported from old bb.jl:646-703)
# ─────────────────────────────────────────────────────────────────────

"""
    getRelaxLowerBoundBT!(P, pr_children, Rold, Pex, prex, UB, defaultLB; nsolve=20)
        -> (status, LB, R, reduction_first)

Solve the LP relaxation, then tighten bounds via:
  1. Copy R's bounds back to Pex
  2. Reduced-cost bound tightening
  3. FBBT on relaxation model
  4. Sync back to stochastic form
  5. Compute volume-reduction fraction
  6. If significant reduction, run Sto FBBT pass
"""
function getRelaxLowerBoundBT!(P, pr_children, Rold, Pex, prex, UB, defaultLB; nsolve::Int=20)
    nfirst = P.numCols
    xlold = copy(Pex.colLower)
    xuold = copy(Pex.colUpper)

    # Step 1: solve relaxation (reuses existing getRelaxLowerBound)
    relaxed_status, relaxed_LB, R = getRelaxLowerBound(Rold, Pex, prex, UB, defaultLB;
                                                        relaxBin=true, nsolve=nsolve)

    if relaxed_status == :Optimal &&
       ((UB - relaxed_LB) <= mingap || (UB - relaxed_LB) / abs(relaxed_LB) <= mingap)
        return (relaxed_status, relaxed_LB, R, 0.0)
    end
    if relaxed_status == :Infeasible
        return (:Infeasible, UB, R, 0.0)
    end

    if relaxed_status == :Optimal
        # Step 2: copy tightened bounds from R back to Pex
        n = min(length(Pex.colLower), length(R.colLower))
        Pex.colLower[1:n] .= R.colLower[1:n]
        Pex.colUpper[1:n] .= R.colUpper[1:n]

        # Step 3: reduced-cost bound tightening
        reduced_cost_BT!(Pex, prex, R, UB, relaxed_LB)

        # Step 4: FBBT on relaxation model
        fb = fast_feasibility_reduction!(R, nothing, UB)
        if !fb
            return (:Infeasible, UB, R, 0.0)
        end

        # Step 5: copy bounds back again after FBBT
        n2 = min(length(Pex.colLower), length(R.colLower))
        Pex.colLower[1:n2] .= R.colLower[1:n2]
        Pex.colUpper[1:n2] .= R.colUpper[1:n2]

        # Step 6: sync back to stochastic form
        updateStoBoundsFromExtensive!(Pex, P)
    end

    # Step 7: compute volume reduction (over all variables)
    left_all = 1.0
    for i in 1:length(Pex.colLower)
        if (xuold[i] + Pex.colLower[i] - xlold[i] - Pex.colUpper[i]) > small_bound_improve
            denom = xuold[i] - xlold[i]
            if denom > 0
                left_all *= (Pex.colUpper[i] - Pex.colLower[i]) / denom
            end
        end
    end
    reduction_all = 1.0 - left_all

    # Step 8: if significant reduction, run additional Sto FBBT + compute first-stage reduction
    reduction_first = 0.0
    if reduction_all > 0.1
        feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, Rold, UB, relaxed_LB)
        if !feasible
            return (:Infeasible, UB, R, 0.0)
        end
        updateExtensiveBoundsFromSto!(P, Pex)

        left_first = 1.0
        for i in 1:nfirst
            if (xuold[i] + Pex.colLower[i] - xlold[i] - Pex.colUpper[i]) > small_bound_improve
                denom = xuold[i] - xlold[i]
                if denom > 0
                    left_first *= (Pex.colUpper[i] - Pex.colLower[i]) / denom
                end
            end
        end
        reduction_first = 1.0 - left_first
    end

    return (relaxed_status, relaxed_LB, R, reduction_first)
end

# ─────────────────────────────────────────────────────────────────────
# Lower Bound: WS (Wait-and-See) — full version with SCIP
# ─────────────────────────────────────────────────────────────────────

"""
    getWSLowerBound(P, parWSSol, UB) -> (WS_status, WS_LB, WSfirst, WSSol)

Compute the wait-and-see lower bound by solving each scenario
  1. Ipopt local solve (warmstart from parent WS solution)
  2. SCIP global solve per scenario
"""
function getWSLowerBound(P::ModelWrapper, parWSSol, UB::Float64)
    nfirst = P.numCols
    scenarios = getchildren(P)
    nscen = length(scenarios)
    WS_LB = 0.0
    WS_status = :Optimal
    WSSol = StoSol(nscen)
    WSfirstSol = [Float64[] for _ in 1:nfirst]
    WSfirst = zeros(nfirst)

    for (idx, scenario) in enumerate(scenarios)
        # Check if parent solution is still feasible
        if parWSSol !== nothing && idx <= length(parWSSol.secondSols) &&
           !isempty(parWSSol.secondSols[idx])
            secondSol = parWSSol.secondSols[idx]
            scenario.colVal = copy(secondSol)
            if inBounds(scenario, secondSol)
                scenario_LB = parWSSol.secondobjVals[idx]
                for k in 1:nfirst
                    varid = scenario.ext[:firstVarsId][k]
                    if varid != -1
                        push!(WSfirstSol[k], scenario.colVal[varid])
                    end
                end
                WS_LB += scenario_LB
                WSSol.secondSols[idx] = secondSol
                WSSol.secondobjVals[idx] = scenario_LB
                continue
            end
        end

        # Local solve with Ipopt
        scenariocopy = copyModel(scenario)
        hasBin_local = any(c -> c == :Bin, scenariocopy.colCat[1:scenariocopy.numCols])
        if hasBin_local
            fixBinaryVar(scenariocopy)
        end
        scenariocopy_status = solve_model!(scenariocopy, ipopt_optimizer())

        if scenariocopy_status == :Unbounded ||
           (scenariocopy_status == :Optimal && scenariocopy.objVal <= -1e10)
            WS_status = :NotOptimal
            WS_LB = -1e20
            break
        elseif scenariocopy_status == :Optimal && parWSSol === nothing
            scenario.colVal = copy(scenariocopy.colVal)
        end

        # Global solve with SCIP
        if scenariocopy_status == :Optimal
            scen_solver = scip_optimizer(mingap/2, 100.0)
        else
            scen_solver = scip_optimizer(mingap/2, 100.0)
        end
        scenario_status = solve_model!(scenario, scen_solver)
        scenario_LB = scenario.objBound !== nothing ? scenario.objBound : scenario.objVal

        if scenariocopy_status == :Optimal && scenario_status == :Infeasible
            scenario_status = :Optimal
            scenario.colVal = copy(scenariocopy.colVal)
            scenario.objVal = scenariocopy.objVal
            scenario_LB = scenariocopy.objVal
        end

        projection!(scenario.colVal, scenario.colLower, scenario.colUpper)

        if scenario_status == :Optimal || scenario_status == :UserLimit
            for k in 1:nfirst
                varid = scenario.ext[:firstVarsId][k]
                if varid != -1
                    push!(WSfirstSol[k], scenario.colVal[varid])
                end
            end
            WS_LB += scenario_LB
            WSSol.secondSols[idx] = copy(scenario.colVal)
            WSSol.secondobjVals[idx] = scenario_LB

            # Early termination: check if already converged
            currentLB = WS_LB
            for j in (idx+1):nscen
                currentLB += haskey(scenarios[j].ext, :objLB) ? scenarios[j].ext[:objLB] : -1e10
            end
            if (UB - currentLB) <= mingap || (UB - currentLB) <= mingap * abs(currentLB)
                WS_status = :Infeasible
                break
            end
        elseif scenario_status == :Infeasible
            WS_status = :Infeasible
            break
        else
            WS_status = :NotOptimal
            println("WS: SCIP not optimal for scenario ", idx)
            error("WS_status NotOptimal")
        end
    end

    if WS_status == :Optimal
        for k in 1:nfirst
            if !isempty(WSfirstSol[k])
                WSfirst[k] = length(WSfirstSol[k]) > 0 ? median(WSfirstSol[k]) : 0.0
            end
        end
    end

    return (WS_status, WS_LB, WSfirst, WSSol)
end

function median(v::Vector{Float64})
    isempty(v) && return 0.0
    sv = sort(v)
    n = length(sv)
    if isodd(n)
        return sv[div(n+1, 2)]
    else
        return (sv[div(n, 2)] + sv[div(n, 2) + 1]) / 2.0
    end
end

# ─────────────────────────────────────────────────────────────────────
# Combined Lower Bound
# ─────────────────────────────────────────────────────────────────────

function getLowerBound(P::ModelWrapper, parWSSol, Rold, Pex, prex, UB, defaultLB)
    nfirst = P.numCols
    relaxed_status, relaxed_LB, R = getRelaxLowerBound(Rold, Pex, prex, UB, defaultLB)

    if relaxed_status == :Optimal &&
       ((UB - relaxed_LB) <= mingap || (UB - relaxed_LB) / abs(relaxed_LB) <= mingap)
        return (relaxed_status, relaxed_LB, :NotOptimal, relaxed_status, relaxed_LB,
                zeros(nfirst), R, nothing)
    end

    LB = relaxed_LB
    WS_LB = -1e10
    WS_status = :NotOptimal
    WSfirst = zeros(nfirst)
    WSSol = nothing

    updateStoBoundsFromExtensive!(Pex, P)
    WS_status, WS_LB, WSfirst, WSSol = getWSLowerBound(P, parWSSol, UB)

    if WS_status == :Infeasible
        return (:Infeasible, UB, :Infeasible, relaxed_status, relaxed_LB,
                WSfirst, R, WSSol)
    end
    println("WS_status  ", WS_status, "  WS_LB ", WS_LB)

    LB_status = :NotOptimal
    if WS_status == :Optimal || relaxed_status == :Optimal
        LB_status = :Optimal
    end
    LB = max(relaxed_LB, WS_LB)
    return (LB_status, LB, WS_status, relaxed_status, relaxed_LB, WSfirst, R, WSSol)
end

# ─────────────────────────────────────────────────────────────────────
# Upper Bound — full version (NLP + WSfix)
# ─────────────────────────────────────────────────────────────────────

function getUpperBound!(Pex::ModelWrapper, prex, PWSfix::ModelWrapper,
                        P::ModelWrapper, node_LB::Float64,
                        WS_status::Symbol, WSfirst::Vector{Float64},
                        level::Int)
    updateStoBoundsFromSto!(P, PWSfix)
    nfirst = P.numCols
    UB = 1e10
    UB_status = :NotOptimal
    WSfix_status = :NotOptimal

    # Local NLP solve via Ipopt on each scenario
    local_status = Ipopt_solve(P)
    local_UB = 1e10
    if local_status == :Optimal
        local_UB = getsumobjectivevalue(P)
        if local_UB < UB
            UB_status = local_status
            UB = local_UB
        end
        println("local_UB  ", local_status, "  ", local_UB)
    end

    # WSfix: fix first-stage and solve each scenario with SCIP
    if level <= 2 || level % 3 == 0
        if (local_status == :Optimal || WS_status == :Optimal) &&
           (UB - node_LB) >= mingap && (UB - node_LB) >= mingap * min(abs(node_LB), abs(UB))

            WSfixfirst = local_status == :Optimal ? copy(P.colVal) : copy(WSfirst)
            WSfix_UB = 0.0
            WSfix_status = :Optimal
            scenariosWSfix = getchildren(PWSfix)
            scenarios = getchildren(P)
            nscen = length(scenariosWSfix)

            for (idx, scenarioWSfix) in enumerate(scenariosWSfix)
                if local_status == :Optimal && idx <= length(scenarios)
                    scenarioWSfix.colVal = copy(scenarios[idx].colVal)
                end
                firstVarsId = scenarioWSfix.ext[:firstVarsId]
                for k in 1:nfirst
                    varid = firstVarsId[k]
                    if varid != -1
                        val = WSfixfirst[k]
                        if P.colCat[k] == :Bin
                            val = val >= 0.5 ? 1.0 : 0.0
                        end
                        scenarioWSfix.colCat[varid] = :Fixed
                        scenarioWSfix.colLower[varid] = val
                        scenarioWSfix.colUpper[varid] = val
                        scenarioWSfix.colVal[varid] = val
                    end
                end

                scen_solver = scip_optimizer(mingap/2, 100.0)
                scenarioWSfix_status = solve_model!(scenarioWSfix, scen_solver)

                if local_status == :Optimal && scenarioWSfix_status == :Infeasible &&
                   idx <= length(scenarios)
                    scenarioWSfix_status = :Optimal
                    scenarioWSfix.colVal = copy(scenarios[idx].colVal)
                    scenarioWSfix.objVal = scenarios[idx].objVal
                end

                projection!(scenarioWSfix.colVal, scenarioWSfix.colLower, scenarioWSfix.colUpper)

                if scenarioWSfix_status == :Optimal || scenarioWSfix_status == :UserLimit
                    WSfix_UB += scenarioWSfix.objVal
                elseif scenarioWSfix_status == :Infeasible
                    WSfix_status = :Infeasible
                    break
                else
                    WSfix_status = :NotOptimal
                    println("WSfix: SCIP not optimal for scenario ", idx)
                    break
                end
            end

            if WSfix_status == :Optimal
                UB_status = WSfix_status
                if WSfix_UB < UB
                    UB = WSfix_UB
                end
                println("WSfix_status  ", WSfix_status, "  WSfix_UB ", WSfix_UB)
            end
        end
    end

    return UB_status, WSfix_status, UB, local_UB
end

# ─────────────────────────────────────────────────────────────────────
# Variable selection — Pseudocost branching
# ─────────────────────────────────────────────────────────────────────

"""
    SelectVar!(vs, P, Rsol, WSfirst, WS_status, relaxed_status,
               PWS_child, Pex_child, Rold, prex, UB, node_LB, Pex,
               FLB, pr_children) -> (bVarId, max_score, exitBranch,
               child_left_LB, child_right_LB, FLB, node_LB)

Pseudocost-based variable selection with probing.
"""
function SelectVar!(vs::BranchVarScore, P::ModelWrapper, Rsol,
                    WSfirst::Vector{Float64}, WS_status::Symbol,
                    relaxed_status::Symbol,
                    PWS_child::ModelWrapper, Pex_child::ModelWrapper,
                    Rold, prex, UB::Float64, node_LB::Float64,
                    Pex::ModelWrapper, FLB::Float64,
                    pr_children, P_child=nothing)
    node_LB_old = node_LB
    nfirst = P.numCols
    exitBranch = false
    child_left_LB = -1e10
    child_right_LB = -1e10
    max_score = 0.0
    bVarId = vs.varId[1]
    no_consecutive_updates_wo_change = 0
    n_rel = 100
    lambda_consecutive = 8

    i = 1
    ninfeasible = 0
    while i <= length(vs.varId)
        varId = vs.varId[i]
        bVarl = Pex.colLower[varId]
        bVaru = Pex.colUpper[varId]

        if (bVaru - bVarl) <= small_bound_improve
            i += 1
            continue
        end

        bValue = computeBvalue(Pex, P, varId, Rsol, WSfirst, WS_status, relaxed_status)

        # Probe if reliability is low
        if min(vs.n_right[i], vs.n_left[i]) < n_rel &&
           no_consecutive_updates_wo_change <= lambda_consecutive

            vs.tried[i] = true
            updateStoBoundsFromExtensive!(Pex, PWS_child)
            Pex_child.colLower = copy(Pex.colLower)
            Pex_child.colUpper = copy(Pex.colUpper)
            updateFirstBounds!(PWS_child, bVarl, bValue, varId)
            updateExtensiveFirstBounds!(Pex_child, PWS_child, bVarl, bValue, varId)

            # Probe left child
            LB_status_left, obj_left, _ = getRelaxLowerBound(Rold, Pex_child, prex, UB, node_LB)
            improve_left = 0.0

            if LB_status_left == :Optimal
                if ((UB - obj_left) <= mingap) || ((UB - obj_left) <= mingap * abs(obj_left))
                    if (UB - obj_left) >= 0 && obj_left <= FLB
                        FLB = obj_left
                    end
                    P.colLower[varId] = bValue
                    Pex.colLower[varId] = bValue
                    LB_status_left = :Infeasible
                end
                improve_left = max(0.0, obj_left - node_LB_old)
                nlinfeasibility = bValue - bVarl
                if nlinfeasibility > 0
                    vs.pcost_left[i] = (vs.pcost_left[i]*vs.n_left[i] + improve_left/nlinfeasibility) / (vs.n_left[i] + 1)
                end
                vs.n_left[i] += 1
            elseif LB_status_left == :Infeasible
                P.colLower[varId] = bValue
                Pex.colLower[varId] = bValue
                improve_left = max(0.0, UB - node_LB_old)
            else
                i += 1
                continue
            end

            if LB_status_left == :Infeasible
                updateStoBoundsFromExtensive!(Pex, P)
                Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, Rold, UB, node_LB)
                updateExtensiveBoundsFromSto!(P, Pex)
                updateExtensiveBoundsFromSto!(P, Pex_child)
                updateStoBoundsFromSto!(P, PWS_child)
            end

            # Probe right child
            updateStoBoundsFromExtensive!(Pex, PWS_child)
            Pex_child.colLower = copy(Pex.colLower)
            Pex_child.colUpper = copy(Pex.colUpper)
            updateFirstBounds!(PWS_child, bValue, bVaru, varId)
            updateExtensiveFirstBounds!(Pex_child, PWS_child, bValue, bVaru, varId)

            LB_status_right, obj_right, _ = getRelaxLowerBound(Rold, Pex_child, prex, UB, node_LB)
            improve_right = 0.0

            if LB_status_right == :Optimal
                if ((UB - obj_right) <= mingap) || ((UB - obj_right) <= mingap * abs(obj_right))
                    if (UB - obj_right) >= 0 && obj_right <= FLB
                        FLB = obj_right
                    end
                    Pex.colUpper[varId] = bValue
                    P.colUpper[varId] = bValue
                    LB_status_right = :Infeasible
                    if obj_right >= UB
                        obj_right = UB
                    end
                end
                improve_right = max(0.0, obj_right - node_LB_old)
                nlinfeasibility = bVaru - bValue
                if nlinfeasibility > 0
                    vs.pcost_right[i] = (vs.pcost_right[i]*vs.n_right[i] + improve_right/nlinfeasibility) / (vs.n_right[i] + 1)
                end
                vs.n_right[i] += 1
            elseif LB_status_right == :Infeasible
                P.colUpper[varId] = bValue
                Pex.colUpper[varId] = bValue
                obj_right = UB
                improve_right = max(0.0, obj_right - node_LB_old)
            else
                i += 1
                continue
            end

            if LB_status_right == :Infeasible
                updateStoBoundsFromExtensive!(Pex, P)
                Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, Rold, UB, node_LB)
                updateExtensiveBoundsFromSto!(P, Pex)
            end

            # Both infeasible → fathom
            if LB_status_right == :Infeasible && LB_status_left == :Infeasible
                exitBranch = true
                break
            end

            if (LB_status_right == :Infeasible || LB_status_left == :Infeasible)
                ninfeasible += 1
                continue
            end

            # Update node LB from probing
            if LB_status_left == :Infeasible && LB_status_right == :Optimal && (obj_right - node_LB) >= 0
                node_LB = obj_right
            elseif LB_status_right == :Infeasible && LB_status_left == :Optimal && (obj_left - node_LB) >= 0
                node_LB = obj_left
            elseif LB_status_left == :Optimal && LB_status_right == :Optimal &&
                   (obj_left - node_LB) >= 0 && (obj_right - node_LB) >= 0
                node_LB = min(obj_left, obj_right)
            end

            vs.score[i] = compute_score(improve_left, improve_right)

            if vs.score[i] >= max_score
                max_score = vs.score[i]
                bVarId = varId
                if LB_status_left == :Optimal
                    child_left_LB = obj_left
                    LB_status_right == :Infeasible && (child_right_LB = child_left_LB)
                end
                if LB_status_right == :Optimal
                    child_right_LB = obj_right
                    LB_status_left == :Infeasible && (child_left_LB = obj_right)
                end
                no_consecutive_updates_wo_change = 0
            else
                no_consecutive_updates_wo_change += 1
            end
        end  # probing block

        # Check pseudocost score
        if vs.score[i] >= max_score
            max_score = vs.score[i]
            bVarId = varId
        end

        if no_consecutive_updates_wo_change > lambda_consecutive && max_score > 0
            break
        end

        i += 1
        ninfeasible = 0
    end

    updateStoBoundsFromExtensive!(Pex, P)
    return bVarId, max_score, exitBranch, child_left_LB, child_right_LB, FLB, node_LB
end

# ─────────────────────────────────────────────────────────────────────
# Branch
# ─────────────────────────────────────────────────────────────────────

function branch!(nodeList, P, Pex, bVarId, bValue, node, node_LB,
                 WSSol, child_left_LB, child_right_LB,
                 WS_status, relaxed_status, Rsol, max_score)
    nfirst = P.numCols
    hasBin_local = any(c -> c == :Bin, P.colCat[1:nfirst])

    # Create left child
    xl_left = copy(Pex.colLower)
    xu_left = copy(Pex.colUpper)
    bval_left = bValue
    if P.colCat[bVarId] == :Bin
        bval_left = 0.0
    end
    xu_left[bVarId] = bval_left
    # Sync first-stage bounds
    scenarios = getchildren(P)
    for scen in scenarios
        firstVarsId = scen.ext[:firstVarsId]
        if firstVarsId[bVarId] > 0
            scen.colUpper[firstVarsId[bVarId]] = bval_left
        end
    end

    left = Node(nfirst, length(scenarios))
    left.xlower = xl_left[1:nfirst]
    left.xupper = xu_left[1:nfirst]
    left.LB = node_LB
    left.parent_LB = node_LB
    if WS_status == :Optimal && WSSol !== nothing
        left.x_ws = [copy(WSSol.secondSols[i]) for i in 1:length(scenarios)]
    end
    if child_left_LB != -1e10 && child_left_LB >= node_LB
        left.LB = child_left_LB
    end

    # Create right child
    xl_right = copy(Pex.colLower)
    xu_right = copy(Pex.colUpper)
    bval_right = bValue
    if P.colCat[bVarId] == :Bin
        bval_right = 1.0
    end
    xl_right[bVarId] = bval_right
    for scen in scenarios
        firstVarsId = scen.ext[:firstVarsId]
        if firstVarsId[bVarId] > 0
            scen.colLower[firstVarsId[bVarId]] = bval_right
        end
    end

    right = Node(nfirst, length(scenarios))
    right.xlower = xl_right[1:nfirst]
    right.xupper = xu_right[1:nfirst]
    right.LB = node_LB
    right.parent_LB = node_LB
    if WS_status == :Optimal && WSSol !== nothing
        right.x_ws = [copy(WSSol.secondSols[i]) for i in 1:length(scenarios)]
    end
    if child_right_LB != -1e10 && child_right_LB >= node_LB
        right.LB = child_right_LB
    end

    # Copy relaxation solution
    if relaxed_status == :Optimal && Rsol !== nothing
        left.x_relax = copy(Rsol)
        right.x_relax = copy(Rsol)
    end

    push!(nodeList, left)
    push!(nodeList, right)
end

# ─────────────────────────────────────────────────────────────────────
# Multi-start initial solve
# ─────────────────────────────────────────────────────────────────────

function multi_start!(P::ModelWrapper, nstarts::Int, optimizer_factory)
    best_UB = Inf
    best_x = zeros(P.numCols)

    for s in 1:nstarts
        Pex = extensiveSimplifiedModel(P)
        Pnl = copyNLModel(Pex)

        # Random starting point
        if s > 1
            for i in 1:Pnl.numCols
                lb = Pnl.colLower[i]
                ub = Pnl.colUpper[i]
                if isfinite(lb) && isfinite(ub)
                    Pnl.colVal[i] = lb + rand() * (ub - lb)
                end
            end
        end

        status = solve_model!(Pnl, optimizer_factory)

        if status == :Optimal && Pnl.objVal < best_UB
            best_UB = Pnl.objVal
            nfirst = P.numCols
            best_x = Pnl.colVal[1:nfirst]
        end
    end

    return (best_UB, best_x)
end

# ─────────────────────────────────────────────────────────────────────
# Main Branch-and-Bound Function
# ─────────────────────────────────────────────────────────────────────

"""
    branch_bound(P::ModelWrapper; max_iter=5000) -> (LB, UB, gap, niter)

Main spatial branch-and-bound algorithm for the stochastic
nonlinear programming problem.
"""
function branch_bound(P::ModelWrapper; max_iter::Int=5000)
    t_start = time()

    scenarios = getchildren(P)
    nscen = length(scenarios)
    nfirst = P.numCols

    println("=" ^ 60)
    println("SNGO — Structured Nonlinear Global Optimizer (Modern)")
    println("Scenarios: ", nscen, "  First-stage vars: ", nfirst)
    println("=" ^ 60)

    # ── Step 1: Preprocessing ──
    println("\n[1] Preprocessing scenarios...")
    pr_children = preprocessSto!(P)
    println("  Preprocessing complete. Scenarios: ", length(pr_children))

    # ── Step 2: Build extensive form & preprocess ──
    println("\n[2] Building extensive form...")
    Pex = extensiveSimplifiedModel(P)
    prex = preprocessex!(Pex)
    println("  Extensive form: ", Pex.numCols, " vars, ", length(Pex.linconstr), " lin constrs")

    # ── Step 3: Initial relaxation ──
    println("\n[3] Building initial relaxation...")
    R = relax(Pex, prex)
    Roriginal = copyModel(R)    # (2.7) keep clean copy for per-node reset
    println("  Relaxation: ", R.numCols, " vars, ", length(R.linconstr), " constrs")

    # ── Step 4: Initial bounds ──
    println("\n[4] Computing initial bounds...")

    # Initial UB via multi-start NLP
    UB, best_x = multi_start!(P, 3, ipopt_optimizer())
    println("  Initial UB (multi-start): ", UB)

    # (2.4a) No pre-loop getLowerBound — old code computes first LB inside the main loop
    hasBin_global = any(c -> c == :Bin, P.colCat[1:nfirst])

    # ── Step 5: Pre-loop FBBT (old bb.jl:119) ──
    println("\n[5] Root FBBT...")
    feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, nothing, UB)
    if !feasible
        println("  Root node infeasible!")
        return (-1e10, UB, Inf, 0)
    end
    updateStoFirstBounds!(P)

    # ── Step 6: Initialize BnB tree ──
    bVarsId = prex.branchVarsId
    bVarsIdFirst = filter(v -> v <= nfirst, bVarsId)

    # Initialize BranchVarScore for pseudocost branching
    vs = BranchVarScore(bVarsIdFirst)

    # Create child copies for probing
    PWS_child = copyModel(P)
    Pex_child = copyModel(Pex)

    # (2.4a) root.LB = -1e10, matching old code (old bb.jl:170)
    root = Node(nfirst, nscen)
    root.xlower = copy(P.colLower[1:nfirst])
    root.xupper = copy(P.colUpper[1:nfirst])
    root.LB = -1e10

    queue = [root]
    best_UB = UB
    best_x_first = copy(best_x)
    niter = 0
    FLB = UB    # (2.4a) old code: FLB = UB (old bb.jl:187)
    solved_nodes = 0

    println("\n[6] Starting Branch-and-Bound...")
    @printf("\n%6s  %14s  %14s  %10s  %6s\n",
            "Iter", "Lower Bound", "Upper Bound", "Gap(%)", "Nodes")
    println("-" ^ 60)

    # ── Main BnB Loop ──
    while !isempty(queue) && niter < max_iter
        niter += 1
        solved_nodes += 1

        # Select node (best-bound)
        best_idx = argmin([n.LB for n in queue])
        node = queue[best_idx]
        deleteat!(queue, best_idx)

        # Update global LB
        if !isempty(queue)
            LB = max(minimum([n.LB for n in queue]), FLB)
        else
            LB = max(node.LB, FLB)
        end

        # Check convergence
        gap = abs(best_UB - LB) / max(1.0, min(abs(LB), abs(best_UB)))
        if gap <= mingap
            @printf("%6d  %14.4f  %14.4f  %10.4f  %6d  CONVERGED\n",
                    niter, LB, best_UB, gap * 100, length(queue))
            break
        end

        # Set bounds from node
        P.colLower[1:nfirst] = copy(node.xlower)
        P.colUpper[1:nfirst] = copy(node.xupper)
        for (idx, scen) in enumerate(scenarios)
            firstVarsId = scen.ext[:firstVarsId]
            for i in 1:nfirst
                fvi = firstVarsId[i]
                if fvi > 0
                    scen.colLower[fvi] = node.xlower[i]
                    scen.colUpper[fvi] = node.xupper[i]
                end
            end
        end
        updateExtensiveBoundsFromSto!(P, Pex)

        # (2.7) Reset relaxation model from clean copy (old bb.jl:219)
        Rold = copyModel(Roriginal)

        # ── Iterative BT + Relax (old bb.jl:249-432) ──
        reduction_relax = 1.0
        node_LB = node.LB
        relaxed_status = :Optimal
        relaxed_LB = node.LB
        R_node = Rold
        feasible = true

        while reduction_relax >= 0.1
            reduction_relax = 0.0

            # FBBT+OBBT (old bb.jl:256: Sto_medium_feasibility_reduction)
            feasible = Sto_medium_feasibility_reduction(
                P, pr_children, Pex, prex, Rold, best_UB, LB,
                prex.branchVarsId, gurobi_simplex_optimizer())
            updateExtensiveBoundsFromSto!(P, Pex)
            if !feasible
                node_LB = best_UB
                break
            end

            # Relaxation LB with reduced-cost BT (old bb.jl:408)
            relaxed_status, relaxed_LB, R_node, reduction_relax =
                getRelaxLowerBoundBT!(P, pr_children, Rold, Pex, prex, best_UB, node.LB)
            node_LB = max(node_LB, relaxed_LB)

            # Convergence check within inner loop (old bb.jl:414-425)
            if relaxed_status == :Optimal
                if (best_UB - relaxed_LB) <= mingap ||
                   (best_UB - relaxed_LB) <= mingap * min(abs(best_UB), abs(relaxed_LB))
                    if (best_UB - relaxed_LB) >= 0 && relaxed_LB <= FLB
                        FLB = relaxed_LB
                    end
                    relaxed_status = :Infeasible
                end
            end
            relaxed_status == :Infeasible && break
        end  # while reduction_relax

        # Fathom if infeasible (old bb.jl:436-439)
        if !feasible || relaxed_status == :Infeasible
            FLB = max(FLB, node_LB)
            continue
        end

        # WS lower bound — computed AFTER the inner loop (old bb.jl:441+)
        WS_status = :NotOptimal
        WSfirst_node = zeros(nfirst)
        WSSol_node = nothing
        updateStoBoundsFromExtensive!(Pex, P)
        WS_status, WS_LB, WSfirst_node, WSSol_node = getWSLowerBound(P, nothing, best_UB)
        if WS_status == :Optimal
            node_LB = max(node_LB, WS_LB)
        end

        # Prune by bound
        node_LB = max(node_LB, node.LB)
        if node_LB >= best_UB - mingap ||
           (node_LB >= best_UB && (best_UB - node_LB) <= mingap * min(abs(node_LB), abs(best_UB)))
            FLB = max(FLB, node_LB)
            continue
        end

        # Upper bound
        updateStoBoundsFromSto!(P, PWS_child)
        UB_status, WSfix_status, node_UB, local_UB =
            getUpperBound!(Pex, prex, PWS_child, P, node_LB, WS_status, WSfirst_node, niter)

        if node_UB < best_UB
            best_UB = node_UB
            best_x_first = copy(P.colVal[1:nfirst])
            println("  *** New best UB: ", best_UB)
        end

        # Re-check gap after UB update
        if (best_UB - node_LB) <= mingap ||
           (best_UB - node_LB) <= mingap * min(abs(node_LB), abs(best_UB))
            FLB = max(FLB, node_LB)
            continue
        end

        # Variable selection — pseudocost branching with probing
        Rsol = relaxed_status == :Optimal ? R_node.colVal : nothing
        bVarId, max_score, exitBranch, child_left_LB, child_right_LB, FLB, node_LB =
            SelectVar!(vs, P, Rsol, WSfirst_node, WS_status, relaxed_status,
                       PWS_child, Pex_child, Rold, prex, best_UB, node_LB, Pex,
                       FLB, pr_children)

        if exitBranch
            FLB = max(FLB, node_LB)
            continue
        end

        # (2.5) Compute split point using 6-arg computeBvalue (old bb.jl:541)
        Rsol_branch = relaxed_status == :Optimal ? R_node.colVal : nothing
        bValue = computeBvalue(Pex, P, bVarId, Rsol_branch, WSfirst_node,
                               WS_status, relaxed_status)

        # Branch
        branch!(queue, P, Pex, bVarId, bValue, node, node_LB,
                WSSol_node, child_left_LB, child_right_LB,
                WS_status, relaxed_status, Rsol, max_score)

        # Print progress
        gap = abs(best_UB - LB) / max(1.0, min(abs(LB), abs(best_UB)))
        if niter % 10 == 1 || niter <= 5
            @printf("%6d  %14.4f  %14.4f  %10.4f  %6d\n",
                    niter, LB, best_UB, gap * 100, length(queue))
        end
    end

    t_elapsed = time() - t_start

    # Final output
    println("\n" * "=" ^ 60)
    println("SNGO Results")
    println("=" ^ 60)
    @printf("  Lower Bound:    %14.6f\n", LB)
    @printf("  Upper Bound:    %14.6f\n", best_UB)
    gap = abs(best_UB - LB) / max(1.0, min(abs(LB), abs(best_UB)))
    @printf("  Gap:            %14.6f %%\n", gap * 100)
    @printf("  Iterations:     %14d\n", niter)
    println("  solved nodes:  ", solved_nodes)
    @printf("  Solution time:  %14.2f seconds\n", t_elapsed)
    println("  Best first-stage solution: ", best_x_first)
    println("=" ^ 60)

    return (LB, best_UB, gap, niter)
end
