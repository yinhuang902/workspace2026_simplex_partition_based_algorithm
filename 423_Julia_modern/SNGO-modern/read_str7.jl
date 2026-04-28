using Pkg
Pkg.activate(".")
include("src/SNGO.jl")
include("test/test_2_1_7.jl")
P_jump = createModel()
wrapper = SNGO.import_jump_model(P_jump)
P = SNGO.RandomStochasticModel(() -> createModel(), 100)
Pex = SNGO.extensiveSimplifiedModel(P)
Pnl = SNGO.copyNLModel(Pex)

status = SNGO.solve_model!(Pnl, JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))
println("Status: ", status)
println("ObjVal: ", Pnl.objVal)
