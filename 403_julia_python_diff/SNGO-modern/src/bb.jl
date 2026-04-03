"""
    Branch-and-Bound Main Loop

Ported from bb.jl (~1365 lines). Contains:
  - `branch_bound(P)`: main spatial BnB loop
  - Lower bounding via convex relaxation + WS bound
  - Upper bounding via Ipopt local solve + WSfix global solve
  - Node selection (best-bound)
  - Branching (max-range variable selection + solution-weighted split point)
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
# Lower Bound: WS (Wait-and-See)
# ─────────────────────────────────────────────────────────────────────

"""
    getWSLowerBound(P, pr_children, optimizer_factory) -> (LB, solutions)

Compute the wait-and-see lower bound by solving each scenario
independently to global optimality.
"""
function getWSLowerBound(P::ModelWrapper, pr_children, optimizer_factory)
    scenarios = getchildren(P)
    nscen = length(scenarios)
    nfirst = P.numCols
    WS_LB = 0.0
    solutions = [Float64[] for _ in 1:nscen]

    for (idx, scen) in enumerate(scenarios)
        status = solve_model!(scen, optimizer_factory)

        if status == :Optimal
            WS_LB += scen.objVal
            solutions[idx] = copy(scen.colVal)
        elseif status == :Infeasible
            return (Inf, solutions)
        else
            # Use a large penalty
            WS_LB += 1e10
        end
    end

    return (WS_LB / nscen, solutions)  # Average over scenarios
end

# ─────────────────────────────────────────────────────────────────────
# Lower Bound: Relaxation
# ─────────────────────────────────────────────────────────────────────

"""
    getRelaxLowerBound(R, optimizer_factory) -> (LB, solution, status)

Solve the LP relaxation for a lower bound.
"""
function getRelaxLowerBound(R::ModelWrapper, optimizer_factory)
    status = solve_model!(R, optimizer_factory)
    if status == :Optimal
        return (R.objVal, copy(R.colVal), status)
    elseif status == :Infeasible
        return (Inf, Float64[], status)
    else
        return (-1e20, Float64[], status)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Upper Bound
# ─────────────────────────────────────────────────────────────────────

"""
    getUpperBound!(P, x_first, optimizer_factory) -> (UB, status)

Fix first-stage variables to `x_first`, solve each scenario to
global optimality, sum objectives.
"""
function getUpperBound!(P::ModelWrapper, x_first::Vector{Float64},
                        optimizer_factory)
    scenarios = getchildren(P)
    nscen = length(scenarios)
    UB = 0.0

    for (idx, scen) in enumerate(scenarios)
        firstVarsId = scen.ext[:firstVarsId]
        # Create a copy with fixed first-stage variables
        scen_fix = copyModel(scen)
        for i in 1:length(firstVarsId)
            fvi = firstVarsId[i]
            if fvi > 0 && i <= length(x_first)
                scen_fix.colLower[fvi] = x_first[i]
                scen_fix.colUpper[fvi] = x_first[i]
            end
        end

        status = solve_model!(scen_fix, optimizer_factory)

        if status == :Optimal
            UB += scen_fix.objVal
        else
            return (Inf, :Infeasible)
        end
    end

    return (UB / nscen, :Optimal)
end

# ─────────────────────────────────────────────────────────────────────
# Local Upper Bound (NLP)
# ─────────────────────────────────────────────────────────────────────

"""
    getLocalUpperBound(P, optimizer_factory) -> (UB, x_first)

Solve the stochastic problem locally with an NLP solver.
"""
function getLocalUpperBound(P::ModelWrapper, optimizer_factory)
    # Build extensive form and solve locally
    Pex = extensiveSimplifiedModel(P)
    Pnl = copyNLModel(Pex)
    status = solve_model!(Pnl, optimizer_factory)

    if status == :Optimal
        nfirst = P.numCols
        x_first = Pnl.colVal[1:nfirst]
        return (Pnl.objVal, x_first)
    else
        return (Inf, zeros(P.numCols))
    end
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
    println("  Relaxation: ", R.numCols, " vars, ", length(R.linconstr), " constrs")

    # ── Step 4: Initial bounds ──
    println("\n[4] Computing initial bounds...")

    # Initial UB via multi-start NLP
    UB, best_x = multi_start!(P, 3, ipopt_optimizer())
    println("  Initial UB (multi-start): ", UB)

    # Initial LB via relaxation
    relax_LB, relax_sol, relax_status = getRelaxLowerBound(R, gurobi_lp_optimizer())
    println("  Initial Relax LB: ", relax_LB)

    # WS lower bound
    ws_LB, ws_solutions = getWSLowerBound(P, pr_children, scip_optimizer())
    println("  Initial WS LB: ", ws_LB)

    LB = max(relax_LB, ws_LB)

    # ── Step 5: FBBT at root ──
    println("\n[5] Root FBBT...")
    feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, R, UB, LB)
    if !feasible
        println("  Root node infeasible!")
        return (LB, UB, Inf, 0)
    end
    updateStoFirstBounds!(P)

    # ── Step 6: Initialize BnB tree ──
    bVarsId = prex.branchVarsId
    bVarsIdFirst = filter(v -> v <= nfirst, bVarsId)

    root = Node(nfirst, nscen)
    root.xlower = copy(P.colLower[1:nfirst])
    root.xupper = copy(P.colUpper[1:nfirst])
    root.LB = LB
    root.x_ws = ws_solutions
    root.x_relax = relax_sol[1:min(nfirst, length(relax_sol))]

    queue = [root]
    best_UB = UB
    best_x_first = copy(best_x)
    niter = 0
    FLB = -1e20  # fathomed lower bound

    println("\n[6] Starting Branch-and-Bound...")
    @printf("\n%6s  %14s  %14s  %10s  %6s\n",
            "Iter", "Lower Bound", "Upper Bound", "Gap(%)", "Nodes")
    println("-" ^ 60)

    # ── Main BnB Loop ──
    while !isempty(queue) && niter < max_iter
        niter += 1

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

        # FBBT on this node
        feasible = Sto_fast_feasibility_reduction!(P, pr_children, Pex, prex, R, best_UB, LB)
        if !feasible
            FLB = max(FLB, node.LB)
            continue
        end

        # Lower bound: relaxation + WS
        R_updated = updaterelax(R, Pex, prex, best_UB)
        node_relax_LB, node_relax_sol, relax_status = getRelaxLowerBound(R_updated, gurobi_lp_optimizer())

        # Add OA cuts and re-solve
        if relax_status == :Optimal
            nOA = addOuterApproximation!(R_updated, prex)
            naBB = addaBB!(R_updated, prex)
            if nOA > 0 || naBB > 0
                node_relax_LB2, _, _ = getRelaxLowerBound(R_updated, gurobi_lp_optimizer())
                node_relax_LB = max(node_relax_LB, node_relax_LB2)
            end
        end

        # WS bound
        ws_LB_node, ws_sol_node = getWSLowerBound(P, pr_children, scip_optimizer())

        node_LB = max(node_relax_LB, ws_LB_node, node.LB)

        # Prune by bound
        if node_LB >= best_UB - mingap
            FLB = max(FLB, node_LB)
            continue
        end

        # Upper bound: find best first-stage point from WS solutions
        # Median first-stage solution across scenarios
        x_first_candidate = zeros(nfirst)
        n_valid = 0
        for idx in 1:nscen
            if !isempty(ws_sol_node[idx])
                firstVarsId = scenarios[idx].ext[:firstVarsId]
                for i in 1:nfirst
                    fvi = firstVarsId[i]
                    if fvi > 0 && fvi <= length(ws_sol_node[idx])
                        x_first_candidate[i] += ws_sol_node[idx][fvi]
                    end
                end
                n_valid += 1
            end
        end
        if n_valid > 0
            x_first_candidate ./= n_valid
            # Clamp to node bounds
            for i in 1:nfirst
                x_first_candidate[i] = clamp(x_first_candidate[i],
                                              node.xlower[i], node.xupper[i])
            end

            node_UB, ub_status = getUpperBound!(P, x_first_candidate, scip_optimizer())
            if ub_status == :Optimal && node_UB < best_UB
                best_UB = node_UB
                best_x_first = copy(x_first_candidate)
            end
        end

        # Branch: select variable (max range)
        branch_var = 0
        max_range = 0.0
        for i in 1:nfirst
            range = node.xupper[i] - node.xlower[i]
            if range > max_range && range > 1e-6
                max_range = range
                branch_var = i
            end
        end

        if branch_var == 0
            # No variable to branch — fathom
            FLB = max(FLB, node_LB)
            continue
        end

        # Compute split point
        bval = computeBvalue(node.xlower[branch_var], node.xupper[branch_var],
                             best_x_first[branch_var],
                             node.cat[branch_var])

        # Create child nodes
        left = Node(nfirst, nscen)
        left.xlower = copy(node.xlower)
        left.xupper = copy(node.xupper)
        left.xupper[branch_var] = bval
        left.LB = node_LB
        left.parent_LB = node_LB
        left.cat = copy(node.cat)
        left.x_ws = ws_sol_node

        right = Node(nfirst, nscen)
        right.xlower = copy(node.xlower)
        right.xupper = copy(node.xupper)
        right.xlower[branch_var] = bval
        right.LB = node_LB
        right.parent_LB = node_LB
        right.cat = copy(node.cat)
        right.x_ws = ws_sol_node

        push!(queue, left)
        push!(queue, right)

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
    @printf("  Solution time:  %14.2f seconds\n", t_elapsed)
    println("  Best first-stage solution: ", best_x_first)
    println("=" ^ 60)

    return (LB, best_UB, gap, niter)
end
