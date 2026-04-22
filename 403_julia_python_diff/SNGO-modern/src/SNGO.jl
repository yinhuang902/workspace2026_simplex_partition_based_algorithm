"""
    SNGO — Structured Nonlinear Global Optimizer (Modern Julia Port)

Port of the original Julia 0.6 SNGO codebase to Julia 1.x / JuMP 1.x.
Original authors: Yankai Cao, Victor M. Zavala (UW-Madison)
"""
module SNGO

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra
using Random
using Printf
using Distributions
using Ipopt
using Gurobi
using SCIP

# Compatibility layer for column-indexed JuMP model access
include("compat.jl")

# PlasmoOld replacement — stochastic model management
include("stochastic.jl")

# Core data structures
include("core.jl")

# Preprocessing
include("preprocess.jl")
include("preprocessSto.jl")
include("preprocessex.jl")

# Relaxation construction & update
include("relax.jl")
include("updaterelax.jl")

# Bound tightening (FBBT / OBBT)
include("boundT.jl")

# Branch-and-bound main loop
include("bb.jl")

export ModelWrapper, StochasticModel
export RandomStochasticModel, copyStoModel, extensiveSimplifiedModel
export branch_bound

end # module
