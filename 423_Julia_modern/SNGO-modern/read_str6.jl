using Pkg
Pkg.activate(".")
include("src/SNGO.jl")
include("test/test_2_1_7.jl")
P_jump = createModel()
wrapper = SNGO.import_jump_model(P_jump)
nan_count = 0
for v in wrapper.colVal
    if isnan(v)
        nan_count += 1
    end
end
println("NaN count in wrapper: ", nan_count)

Pex = SNGO.extensiveSimplifiedModel(wrapper)
Pnl = SNGO.copyNLModel(Pex)
nan_count2 = 0
for v in Pnl.colVal
    if isnan(v)
        nan_count2 += 1
    end
end
println("NaN count in Pnl limit: ", nan_count2)
