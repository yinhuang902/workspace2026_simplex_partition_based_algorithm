using Pkg
Pkg.activate(".")
include("src/SNGO.jl")
include("test/test_2_1_7.jl")
P_jump = createModel()
wrapper = SNGO.import_jump_model(P_jump)
println("Quad coeffs: ", wrapper.obj.qcoeffs[1:5])
println("First Aff Vars: ", wrapper.obj.aff.vars[1:10])
println("First Aff coeffs: ", wrapper.obj.aff.coeffs[1:10])
