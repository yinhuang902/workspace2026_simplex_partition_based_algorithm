using Pkg
Pkg.activate(".")
include("src/SNGO.jl")
using JuMP

function createModel()
    m = JuMP.Model()
    for i in 1:20
        @variable(m)
    end
    colLower = fill(0.0, 20)
    colUpper = fill(40.0, 20)
    for i in 1:20
        JuMP.set_lower_bound(JuMP.all_variables(m)[i], colLower[i])
        JuMP.set_upper_bound(JuMP.all_variables(m)[i], colUpper[i])
    end
    @constraint(m, sum(JuMP.all_variables(m)) <= 40)
    
    # Just a simple objective to replicate the test
    x = JuMP.all_variables(m)
    @objective(m, Min, -0.5 * (10*x[10]^2 - 40*x[10] + 40))
    return m
end

P = SNGO.RandomStochasticModel(createModel, 100)
Pex = SNGO.extensiveSimplifiedModel(P)
Pnl = SNGO.copyNLModel(Pex)

status = SNGO.solve_model!(Pnl, JuMP.optimizer_with_attributes(SNGO.Ipopt.Optimizer, "print_level" => 5))
println("Status: ", status)
println("ObjVal: ", Pnl.objVal)
