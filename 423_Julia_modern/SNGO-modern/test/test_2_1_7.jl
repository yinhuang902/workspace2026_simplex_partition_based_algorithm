"""
Test case: Global/2_1_7

This is a quadratic programming problem with 20 variables,
10 inequality constraints (<=) plus 1 sum constraint (<=),
and a concave quadratic objective:  Min -0.5 * Σ i*(x_i - 2)^2.

Matches the old SNGO-master/Global/2_1_7 configuration exactly:
  NS = 100, nfirst = 5, nparam = 5
  rdl = 0.0, rdu = 2.0, adl = -10.0, adu = 10.0
"""

# Activate the project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "SNGO.jl"))
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra
using Random
using Printf
using Distributions
using Gurobi
using SCIP
using Ipopt

NS = 100  # Number of scenarios (matches old setup.jl)

function createModel()
    m = JuMP.Model()

    # Variables: all >= 0, no upper bound (default_upper_bound_value will be applied)
    # start values match the old 2_1_7 exactly
    @variable(m, x1 >= 0)
    @variable(m, x2 >= 0)
    @variable(m, x3 >= 0, start = 1.04289)
    @variable(m, x4 >= 0)
    @variable(m, x5 >= 0)
    @variable(m, x6 >= 0)
    @variable(m, x7 >= 0)
    @variable(m, x8 >= 0)
    @variable(m, x9 >= 0)
    @variable(m, x10 >= 0)
    @variable(m, x11 >= 0, start = 1.74674)
    @variable(m, x12 >= 0)
    @variable(m, x13 >= 0, start = 0.43147)
    @variable(m, x14 >= 0)
    @variable(m, x15 >= 0)
    @variable(m, x16 >= 0, start = 4.43305)
    @variable(m, x17 >= 0)
    @variable(m, x18 >= 0, start = 15.85893)
    @variable(m, x19 >= 0)
    @variable(m, x20 >= 0, start = 16.4889)

    # Constraints — 10 inequalities (<=) exactly matching old 2_1_7
    @constraint(m, -3*x1 + 7*x2 - 5*x4 + x5 + x6 + 2*x8 - x9 - x10 - 9*x11 + 3*x12 + 5*x13 + x16 + 7*x17 - 7*x18 - 4*x19 - 6*x20 <= -5)
    @constraint(m, 7*x1 - 5*x3 + x4 + x5 + 2*x7 - x8 - x9 - 9*x10 + 3*x11 + 5*x12 + x15 + 7*x16 - 7*x17 - 4*x18 - 6*x19 - 3*x20 <= 2)
    @constraint(m, -5*x2 + x3 + x4 + 2*x6 - x7 - x8 - 9*x9 + 3*x10 + 5*x11 + x14 + 7*x15 - 7*x16 - 4*x17 - 6*x18 - 3*x19 + 7*x20 <= -1)
    @constraint(m, -5*x1 + x2 + x3 + 2*x5 - x6 - x7 - 9*x8 + 3*x9 + 5*x10 + x13 + 7*x14 - 7*x15 - 4*x16 - 6*x17 - 3*x18 + 7*x19 <= -3)
    @constraint(m, x1 + x2 + 2*x4 - x5 - x6 - 9*x7 + 3*x8 + 5*x9 + x12 + 7*x13 - 7*x14 - 4*x15 - 6*x16 - 3*x17 + 7*x18 - 5*x20 <= 5)

    @constraint(m, x1 + 2*x3 - x4 - x5 - 9*x6 + 3*x7 + 5*x8 + x11 + 7*x12 - 7*x13 - 4*x14 - 6*x15 - 3*x16 + 7*x17 - 5*x19 + x20 <= 4)
    @constraint(m, 2*x2 - x3 - x4 - 9*x5 + 3*x6 + 5*x7 + x10 + 7*x11 - 7*x12 - 4*x13 - 6*x14 - 3*x15 + 7*x16 - 5*x18 + x19 + x20 <= -1)
    @constraint(m, 2*x1 - x2 - x3 - 9*x4 + 3*x5 + 5*x6 + x9 + 7*x10 - 7*x11 - 4*x12 - 6*x13 - 3*x14 + 7*x15 - 5*x17 + x18 + x19 <= 0)
    @constraint(m, -x1 - x2 - 9*x3 + 3*x4 + 5*x5 + x8 + 7*x9 - 7*x10 - 4*x11 - 6*x12 - 3*x13 + 7*x14 - 5*x16 + x17 + x18 + 2*x20 <= 9)
    @constraint(m, x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 <= 40)

    # Objective: Min -0.5 * Σ_i i*(x_i - 2)^2
    # Expanded: -0.5 * i * (x_i^2 - 4*x_i + 4)  =  -0.5*i * x_i^2 + 2*i * x_i - 2*i
    # Written in explicit x*x form for compatibility with legacy quadratic parsing
    @objective(m, Min,
        -0.5 * (1 * x1 * x1 - 4 * x1 + 4
              + 2 * x2 * x2 - 8 * x2 + 8
              + 3 * x3 * x3 - 12 * x3 + 12
              + 4 * x4 * x4 - 16 * x4 + 16
              + 5 * x5 * x5 - 20 * x5 + 20
              + 6 * x6 * x6 - 24 * x6 + 24
              + 7 * x7 * x7 - 28 * x7 + 28
              + 8 * x8 * x8 - 32 * x8 + 32
              + 9 * x9 * x9 - 36 * x9 + 36
              + 10 * x10 * x10 - 40 * x10 + 40
              + 11 * x11 * x11 - 44 * x11 + 44
              + 12 * x12 * x12 - 48 * x12 + 48
              + 13 * x13 * x13 - 52 * x13 + 52
              + 14 * x14 * x14 - 56 * x14 + 56
              + 15 * x15 * x15 - 60 * x15 + 60
              + 16 * x16 * x16 - 64 * x16 + 64
              + 17 * x17 * x17 - 68 * x17 + 68
              + 18 * x18 * x18 - 72 * x18 + 72
              + 19 * x19 * x19 - 76 * x19 + 76
              + 20 * x20 * x20 - 80 * x20 + 80))

    return m
end

# Create stochastic model and solve
# Old 2_1_7: RandomStochasticModel(createModel, NS)
#   → nfirst=5 (old default), nparam=5 (default), rdl=0, rdu=2, adl=-10, adu=10
println("Creating stochastic model with $NS scenarios...")
P = SNGO.RandomStochasticModel(createModel, NS; nfirst=5)
println("Stochastic model created.")

println("\nCopying stochastic model...")
m = SNGO.copyStoModel(P)
println("Copy complete.")

println("\nStarting branch and bound...")
LB, UB, gap, niter = SNGO.branch_bound(m)

println("\n\nFinal: LB=$LB, UB=$UB, gap=$(gap*100)%, iterations=$niter")
# Known global optimum for deterministic 2_1_7: approximately -332.04
