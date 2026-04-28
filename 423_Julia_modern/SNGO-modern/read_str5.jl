using Pkg
Pkg.activate(".")
include("src/SNGO.jl")
using JuMP

m = JuMP.Model()
@variable(m, x1 >= 0); @variable(m, x2 >= 0); @variable(m, x3 >= 0); @variable(m, x4 >= 0)
@variable(m, x5 >= 0); @variable(m, x6 >= 0); @variable(m, x7 >= 0); @variable(m, x8 >= 0)
@variable(m, x9 >= 0); @variable(m, x10 >= 0); @variable(m, x11 >= 0); @variable(m, x12 >= 0)
@variable(m, x13 >= 0); @variable(m, x14 >= 0); @variable(m, x15 >= 0); @variable(m, x16 >= 0)
@variable(m, x17 >= 0); @variable(m, x18 >= 0); @variable(m, x19 >= 0); @variable(m, x20 >= 0)

@objective(m, Min,
    -0.5 * (1 * x1 * x1 - 4 * x1 + 4
            + 2 * x2 * x2 - 8 * x2 + 8
            + 10 * x10 * x10 - 40 * x10 + 40
            + 20 * x20 * x20 - 80 * x20 + 80))

wrapper = SNGO.import_jump_model(m)
println("Quad coeffs: ", wrapper.obj.qcoeffs)
println("First Aff Vars: ", wrapper.obj.aff.vars)
println("First Aff coeffs: ", wrapper.obj.aff.coeffs)
