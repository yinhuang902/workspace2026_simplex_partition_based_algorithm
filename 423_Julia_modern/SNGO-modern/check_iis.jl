using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface

println("Reading model...")
model = read_from_file("failed_relax.lp")
set_optimizer(model, Gurobi.Optimizer)
set_optimizer_attribute(model, "OutputFlag", 0)

println("Optimizing...")
optimize!(model)
status = termination_status(model)
println("Status = ", status)

if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
    println("Computing conflict (IIS)...")
    compute_conflict!(model)
    
    println("Model is Infeasible. IIS constraints:")
    cons = all_constraints(model, include_variable_in_set_constraints=false)
    for c in cons
        if MOI.get(model, MOI.ConstraintConflictStatus(), c.index) == MOI.IN_CONFLICT
            println("Constraint in conflict: ", c)
        end
    end
    
    println("\nIIS variables (Bounds in conflict):")
    for v in all_variables(model)
        if MOI.get(model, MOI.ConstraintConflictStatus(), MOI.VariableIndex(v.value)) == MOI.IN_CONFLICT
            println("Variable bounds in conflict: ", v, " (lower: ", has_lower_bound(v) ? lower_bound(v) : -Inf, " upper: ", has_upper_bound(v) ? upper_bound(v) : Inf, ")")
        end
    end
end
