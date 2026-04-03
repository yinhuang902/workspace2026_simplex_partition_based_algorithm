"""
    Core data structures for SNGO

Ported from Julia 0.6 core.jl:
  - type → mutable struct
  - find → findall
  - findin → findall(in(...), ...)
  - Array{T}(n) → Array{T}(undef, n)
  - eig → eigen
  - etc.

All JuMP-internal access replaced with ModelWrapper column indices.
"""

# ─────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────

const machine_error = 1e-6
const small_bound_improve = 1e-4
const large_bound_improve = 1e-2
const probing_improve = 0.1
const sigma_violation = 1e-3
const local_obj_improve = 1e-2
const mingap = 1e-2
const default_lower_bound_value = -1e4
const default_upper_bound_value = 1e4
const debug = false

# Global flags
hasBin = false
OBBT = true

# ─────────────────────────────────────────────────────────────────────
# Nonlinear variable types (for constraint classification)
# ─────────────────────────────────────────────────────────────────────

"""
ExpVariable: represents  lvar op exp(b * nlvar)
"""
mutable struct ExpVariable
    lvarId::Int
    nlvarId::Int
    op::Symbol      # :<=, :>=, :==
    b::Float64
    cid::Vector{Int}
end

"""
LogVariable: represents  lvar op log(b * nlvar)
"""
mutable struct LogVariable
    lvarId::Int
    nlvarId::Int
    op::Symbol
    b::Float64
    cid::Vector{Int}
end

"""
PowerVariable: represents  lvar op (b * nlvar)^d
"""
mutable struct PowerVariable
    lvarId::Int
    nlvarId::Int
    op::Symbol
    b::Float64
    d::Float64
    cid::Vector{Int}
end

"""
MonomialVariable: represents  lvar op d^(b * nlvar)
"""
mutable struct MonomialVariable
    lvarId::Int
    nlvarId::Int
    op::Symbol
    b::Float64
    d::Float64
    cid::Vector{Int}
end

# ─────────────────────────────────────────────────────────────────────
# MultiVariable — grouping of quadratic terms in a constraint
# ─────────────────────────────────────────────────────────────────────

mutable struct MultiVariable
    terms::QuadExprData
    qVarsId::Vector{Int}
    bilinearVarsId::Vector{Int}
    bilinearVars::Vector{Int}       # column indices of bilinear variables
    Q::Matrix{Float64}
    pd::Int                         # 1=convex, -1=concave, 0=indefinite
    alpha::Vector{Float64}          # aBB parameters
end

function MultiVariable()
    MultiVariable(
        QuadExprData(),     # terms
        Int[],              # qVarsId
        Int[],              # bilinearVarsId
        Int[],              # bilinearVars
        zeros(0,0),         # Q
        0,                  # pd
        Float64[]           # alpha
    )
end

function Base.copy(mv::MultiVariable)
    MultiVariable(
        copy(mv.terms),
        copy(mv.qVarsId),
        copy(mv.bilinearVarsId),
        copy(mv.bilinearVars),
        copy(mv.Q),
        mv.pd,
        copy(mv.alpha)
    )
end

"""
MultiVariableCon: a constraint decomposed into component MultiVariables + affine remainder
"""
mutable struct MultiVariableCon
    mvs::Vector{MultiVariable}
    aff::AffExprData
end

function Base.copy(mvc::MultiVariableCon)
    MultiVariableCon(
        [copy(mv) for mv in mvc.mvs],
        copy(mvc.aff)
    )
end

function copy(mvc::MultiVariableCon, P::ModelWrapper)
    # Same as copy but doesn't need model remapping since we use column indices
    MultiVariableCon(
        [copy(mv) for mv in mvc.mvs],
        copy(mvc.aff)
    )
end

# ─────────────────────────────────────────────────────────────────────
# PreprocessResult
# ─────────────────────────────────────────────────────────────────────

mutable struct PreprocessResult
    branchVarsId::Vector{Int}
    qbVarsId::Any                   # Dict or Vector{Dict}
    EqVconstr::Vector{LinearConstraintData}
    multiVariable_list::Vector{MultiVariableCon}
    multiVariable_convex::Vector{MultiVariable}
    multiVariable_aBB::Vector{MultiVariable}
    expVariable_list::Vector{Any}
    logVariable_list::Vector{Any}
    powerVariable_list::Vector{Any}
    monomialVariable_list::Vector{Any}
end

function PreprocessResult()
    PreprocessResult(
        Int[],                              # branchVarsId
        Dict{Tuple{Int,Int}, Any}(),        # qbVarsId
        LinearConstraintData[],             # EqVconstr
        MultiVariableCon[],                 # multiVariable_list
        MultiVariable[],                    # multiVariable_convex
        MultiVariable[],                    # multiVariable_aBB
        Any[],                              # expVariable_list
        Any[],                              # logVariable_list
        Any[],                              # powerVariable_list
        Any[]                               # monomialVariable_list
    )
end

# ─────────────────────────────────────────────────────────────────────
# BnB Node
# ─────────────────────────────────────────────────────────────────────

mutable struct Node
    xlower::Vector{Float64}
    xupper::Vector{Float64}
    LB::Float64
    parent_LB::Float64
    x_ws::Vector{Vector{Float64}}       # WS solutions per scenario
    x_relax::Vector{Float64}            # relaxation solution
    cat::Vector{Symbol}                 # variable categories
    inherit_ws::Vector{Bool}            # can inherit WS solution?
    pseudocost_max_score::Float64
    pseudocost_var_col::Int
end

function Node(nfirst::Int, nscen::Int)
    Node(
        zeros(nfirst),                  # xlower
        zeros(nfirst),                  # xupper
        -1e20,                          # LB
        -1e20,                          # parent_LB
        [Float64[] for _ in 1:nscen],   # x_ws
        Float64[],                      # x_relax
        fill(:Cont, nfirst),            # cat
        falses(nscen),                  # inherit_ws
        0.0,                            # pseudocost_max_score
        0                               # pseudocost_var_col
    )
end

# ─────────────────────────────────────────────────────────────────────
# Variable selection (pseudocost branching)
# ─────────────────────────────────────────────────────────────────────

mutable struct VarSelector
    down_improvement::Vector{Float64}
    up_improvement::Vector{Float64}
    down_count::Vector{Int}
    up_count::Vector{Int}
    mu::Float64
end

function VarSelector(nfirst::Int; mu::Float64=0.15)
    VarSelector(
        zeros(nfirst),     # down_improvement
        zeros(nfirst),     # up_improvement
        zeros(Int, nfirst),# down_count
        zeros(Int, nfirst),# up_count
        mu
    )
end

function updateScore!(vs::VarSelector, varId::Int, parent_LB::Float64,
                      left_LB::Float64, right_LB::Float64)
    down_delta = max(left_LB - parent_LB, 0.0)
    up_delta = max(right_LB - parent_LB, 0.0)
    vs.down_improvement[varId] += down_delta
    vs.up_improvement[varId] += up_delta
    vs.down_count[varId] += 1
    vs.up_count[varId] += 1
end

function getScore(vs::VarSelector, varId::Int)
    avg_down = vs.down_count[varId] > 0 ?
               vs.down_improvement[varId] / vs.down_count[varId] : 0.0
    avg_up = vs.up_count[varId] > 0 ?
             vs.up_improvement[varId] / vs.up_count[varId] : 0.0
    return (1 - vs.mu) * min(avg_down, avg_up) + vs.mu * max(avg_down, avg_up)
end

# ─────────────────────────────────────────────────────────────────────
# Solution storage
# ─────────────────────────────────────────────────────────────────────

mutable struct StoSol
    x::Vector{Vector{Float64}}      # solutions per scenario
    LB::Float64
    UB::Float64
    x_first::Vector{Float64}        # first-stage solution
end

function StoSol(nscen::Int, nfirst::Int)
    StoSol(
        [Float64[] for _ in 1:nscen],
        -1e20,
        1e20,
        zeros(nfirst)
    )
end

# ─────────────────────────────────────────────────────────────────────
# Helper functions for number classification
# ─────────────────────────────────────────────────────────────────────

positiveEven(d) = d > 0 && isinteger(d) && iseven(Int(d))
negativeEven(d) = d < 0 && isinteger(d) && iseven(Int(d))
positiveOdd(d) = d > 0 && isinteger(d) && isodd(Int(d))
negativeOdd(d) = d < 0 && isinteger(d) && isodd(Int(d))
Odd(d) = isinteger(d) && isodd(Int(d))
positiveFrac(d) = d > 0 && !isinteger(d)
negativeFrac(d) = d < 0 && !isinteger(d)

# ─────────────────────────────────────────────────────────────────────
# Expression parsing — classify NL constraints
# ─────────────────────────────────────────────────────────────────────

"""
    isexponentialcon(expr) -> Bool
    parseexponentialcon(expr) -> ExpVariable

Check/parse constraints of the form: lvar op exp(b * nlvar)
"""
function isexponentialcon(expr::Expr)
    # Pattern: :(comparison(-(x[lvarId], call(:exp, *(b, x[nlvarId]))), 0.0))
    # or similar forms
    try
        if expr.head == :call && length(expr.args) >= 3
            op = expr.args[1]
            lhs = expr.args[2]
            if isa(lhs, Expr) && lhs.head == :call
                if any(a -> isa(a, Expr) && _contains_call(a, :exp), lhs.args)
                    return true
                end
            end
        end
    catch
    end
    return false
end

function parseexponentialcon(expr::Expr)
    # Simplified parser — extracts lvarId, nlvarId, op, b
    op = expr.args[1]
    lhs = expr.args[2]
    lvarId, nlvarId, b = _extract_exp_params(lhs)
    return ExpVariable(lvarId, nlvarId, op, b, [-1, -1, -1])
end

"""
    islogcon(expr) -> Bool
    parselogcon(expr) -> LogVariable
"""
function islogcon(expr::Expr)
    try
        if expr.head == :call && length(expr.args) >= 3
            lhs = expr.args[2]
            if isa(lhs, Expr) && _contains_call(lhs, :log)
                return true
            end
        end
    catch
    end
    return false
end

function parselogcon(expr::Expr)
    op = expr.args[1]
    lhs = expr.args[2]
    lvarId, nlvarId, b = _extract_log_params(lhs)
    return LogVariable(lvarId, nlvarId, op, b, [-1, -1, -1])
end

"""
    ispowercon(expr) -> Bool
    parsepowercon(expr) -> PowerVariable
"""
function ispowercon(expr::Expr)
    try
        if expr.head == :call && length(expr.args) >= 3
            lhs = expr.args[2]
            if isa(lhs, Expr) && _contains_call(lhs, :^)
                return true
            end
        end
    catch
    end
    return false
end

function parsepowercon(expr::Expr)
    op = expr.args[1]
    lhs = expr.args[2]
    lvarId, nlvarId, b, d = _extract_power_params(lhs)
    return PowerVariable(lvarId, nlvarId, op, b, d, [-1, -1, -1])
end

"""
    ismonomialcon(expr) -> Bool
    parsemonomialcon(expr) -> MonomialVariable
"""
function ismonomialcon(expr::Expr)
    # Similar to exponential but with base^(b*x) form
    return false  # Placeholder — monomials are rare in practice
end

function parsemonomialcon(expr::Expr)
    op = expr.args[1]
    lhs = expr.args[2]
    return MonomialVariable(0, 0, op, 1.0, 1.0, [-1, -1, -1])
end

# ─────────────────────────────────────────────────────────────────────
# Expression tree helpers
# ─────────────────────────────────────────────────────────────────────

function _contains_call(expr::Expr, func_name::Symbol)
    if expr.head == :call && length(expr.args) >= 1 && expr.args[1] == func_name
        return true
    end
    for a in expr.args
        if isa(a, Expr) && _contains_call(a, func_name)
            return true
        end
    end
    return false
end
_contains_call(x, func_name::Symbol) = false

function _extract_var_index(expr)
    if isa(expr, Expr) && expr.head == :ref
        return expr.args[2]
    end
    return nothing
end

function _extract_exp_params(expr::Expr)
    # Try to parse: -(x[lvarId], exp(*(b, x[nlvarId])))
    # Returns: (lvarId, nlvarId, b)
    lvarId = 0
    nlvarId = 0
    b = 1.0

    # Walk the expression tree
    _walk_exp_tree(expr, Ref(lvarId), Ref(nlvarId), Ref(b))
    return (lvarId, nlvarId, b)
end

function _walk_exp_tree(expr, lvarId::Ref{Int}, nlvarId::Ref{Int}, b::Ref{Float64})
    if !isa(expr, Expr)
        return
    end
    if expr.head == :ref
        if lvarId[] == 0
            lvarId[] = expr.args[2]
        end
        return
    end
    if expr.head == :call
        if expr.args[1] == :exp
            # Found exp(...) — extract inner variable
            inner = expr.args[2]
            if isa(inner, Expr) && inner.head == :call && inner.args[1] == :*
                # exp(b * x[i])
                for a in inner.args[2:end]
                    if isa(a, Number)
                        b[] = Float64(a)
                    elseif isa(a, Expr) && a.head == :ref
                        nlvarId[] = a.args[2]
                    end
                end
            elseif isa(inner, Expr) && inner.head == :ref
                nlvarId[] = inner.args[2]
                b[] = 1.0
            end
            return
        end
    end
    for a in expr.args
        if isa(a, Expr)
            _walk_exp_tree(a, lvarId, nlvarId, b)
        end
    end
end

function _extract_log_params(expr::Expr)
    lvarId = 0
    nlvarId = 0
    b = 1.0
    _walk_log_tree(expr, Ref(lvarId), Ref(nlvarId), Ref(b))
    return (lvarId, nlvarId, b)
end

function _walk_log_tree(expr, lvarId::Ref{Int}, nlvarId::Ref{Int}, b::Ref{Float64})
    if !isa(expr, Expr)
        return
    end
    if expr.head == :ref
        if lvarId[] == 0
            lvarId[] = expr.args[2]
        end
        return
    end
    if expr.head == :call && expr.args[1] == :log
        inner = expr.args[2]
        if isa(inner, Expr) && inner.head == :call && inner.args[1] == :*
            for a in inner.args[2:end]
                if isa(a, Number)
                    b[] = Float64(a)
                elseif isa(a, Expr) && a.head == :ref
                    nlvarId[] = a.args[2]
                end
            end
        elseif isa(inner, Expr) && inner.head == :ref
            nlvarId[] = inner.args[2]
        end
        return
    end
    for a in expr.args
        if isa(a, Expr)
            _walk_log_tree(a, lvarId, nlvarId, b)
        end
    end
end

function _extract_power_params(expr::Expr)
    lvarId = 0
    nlvarId = 0
    b = 1.0
    d = 2.0
    _walk_power_tree(expr, Ref(lvarId), Ref(nlvarId), Ref(b), Ref(d))
    return (lvarId, nlvarId, b, d)
end

function _walk_power_tree(expr, lvarId::Ref{Int}, nlvarId::Ref{Int},
                          b::Ref{Float64}, d::Ref{Float64})
    if !isa(expr, Expr)
        return
    end
    if expr.head == :ref
        if lvarId[] == 0
            lvarId[] = expr.args[2]
        end
        return
    end
    if expr.head == :call && expr.args[1] == :^
        base = expr.args[2]
        exponent = expr.args[3]
        if isa(exponent, Number)
            d[] = Float64(exponent)
        end
        if isa(base, Expr)
            if base.head == :call && base.args[1] == :*
                for a in base.args[2:end]
                    if isa(a, Number)
                        b[] = Float64(a)
                    elseif isa(a, Expr) && a.head == :ref
                        nlvarId[] = a.args[2]
                    end
                end
            elseif base.head == :ref
                nlvarId[] = base.args[2]
            end
        end
        return
    end
    for a in expr.args
        if isa(a, Expr)
            _walk_power_tree(a, lvarId, nlvarId, b, d)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# Factorable programming — detect & extract NL structure from JuMP model
# ─────────────────────────────────────────────────────────────────────

"""
    factorable!(mw::ModelWrapper) -> ModelWrapper

Process a ModelWrapper to extract nonlinear structure:
  - Move quadratic objective to constraints (add objective_value variable)
  - Set default bounds on unbounded variables
"""
function factorable!(mw::ModelWrapper)
    # Provide initial values where missing
    for i in 1:mw.numCols
        if isnan(mw.colVal[i])
            mw.colVal[i] = 0.0
        end
    end

    # Move quadratic objective to constraint
    quadObj = copy(mw.obj)
    obj_val = eval_quad(mw.obj, mw.colVal)
    obj_col = add_variable!(mw, -Inf, Inf, :Cont, "objective_value", obj_val)

    # Add constraint: objective_value == original_obj
    if length(mw.obj.qvars1) > 0
        # Quadratic objective: objective_value - quad_expr == 0
        new_q = copy(mw.obj)
        # Negate all terms (moving to LHS)
        new_q.qcoeffs .*= -1
        new_q.aff.coeffs .*= -1
        new_q.aff.constant *= -1
        # Add objective_value with coeff +1
        push!(new_q.aff.vars, obj_col)
        push!(new_q.aff.coeffs, 1.0)
        add_quad_constraint!(mw, new_q, :(==))
    else
        # Linear objective: objective_value == linear_expr
        new_aff = copy(mw.obj.aff)
        new_aff.coeffs .*= -1
        new_aff.constant *= -1
        push!(new_aff.vars, obj_col)
        push!(new_aff.coeffs, 1.0)
        add_linear_constraint!(mw, new_aff, 0.0, 0.0)
    end

    # New objective: just minimize/maximize objective_value
    mw.obj = QuadExprData(Int[], Int[], Float64[],
                          AffExprData([obj_col], [1.0], 0.0))

    # Set default bounds for unbounded variables
    for i in 1:mw.numCols
        if mw.colLower[i] == -Inf
            mw.colLower[i] = default_lower_bound_value
        end
        if mw.colUpper[i] == Inf
            mw.colUpper[i] = default_upper_bound_value
        end
    end

    # Squeeze linear constraints
    for lc in mw.linconstr
        squeeze!(lc.terms)
    end

    return mw
end

# ─────────────────────────────────────────────────────────────────────
# Extracting variable IDs from expressions
# ─────────────────────────────────────────────────────────────────────

function extractVarsId(expr::Expr)
    varsId = Int[]
    if expr.head == :ref
        return [expr.args[2]]
    end
    for a in expr.args
        if isa(a, Expr)
            append!(varsId, extractVarsId(a))
        end
    end
    return sort(unique(varsId))
end

function extractVarsId(vars::Vector{Int})
    return sort(unique(vars))
end
