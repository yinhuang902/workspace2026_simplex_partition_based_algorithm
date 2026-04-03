"""
    JuMP Compatibility Layer

Provides a `ModelWrapper` type that wraps a JuMP.Model and exposes
the old Julia 0.6 / JuMP 0.18 column-indexed interface that the SNGO
algorithm code relies on heavily.

This lets us keep the algorithm logic nearly unchanged while using modern JuMP.
"""

# ─────────────────────────────────────────────────────────────────────
# LinearConstraintData — mirrors the old JuMP 0.18 LinearConstraint
# ─────────────────────────────────────────────────────────────────────

"""
Stores a linear constraint as:  lb ≤ aff ≤ ub
where aff = Σ coeffs[k] * vars[k] + constant
"""
mutable struct AffExprData
    vars::Vector{Int}           # column indices
    coeffs::Vector{Float64}
    constant::Float64
end
AffExprData() = AffExprData(Int[], Float64[], 0.0)

function Base.copy(a::AffExprData)
    AffExprData(copy(a.vars), copy(a.coeffs), a.constant)
end

mutable struct LinearConstraintData
    terms::AffExprData
    lb::Float64
    ub::Float64
end

function Base.copy(c::LinearConstraintData)
    LinearConstraintData(copy(c.terms), c.lb, c.ub)
end

"""
Stores a quadratic constraint as:  Σ qcoeffs[k]*qvars1[k]*qvars2[k] + aff  sense 0
where sense ∈ {:<=, :>=, :==}
"""
mutable struct QuadExprData
    qvars1::Vector{Int}         # column indices
    qvars2::Vector{Int}         # column indices
    qcoeffs::Vector{Float64}
    aff::AffExprData
end
QuadExprData() = QuadExprData(Int[], Int[], Float64[], AffExprData())

function Base.copy(q::QuadExprData)
    QuadExprData(copy(q.qvars1), copy(q.qvars2), copy(q.qcoeffs), copy(q.aff))
end

mutable struct QuadConstraintData
    terms::QuadExprData
    sense::Symbol               # :<=, :>=, :==
end

function Base.copy(c::QuadConstraintData)
    QuadConstraintData(copy(c.terms), c.sense)
end

# ─────────────────────────────────────────────────────────────────────
# ModelWrapper — column-indexed model abstraction
# ─────────────────────────────────────────────────────────────────────

"""
    ModelWrapper

Wraps a JuMP Model and provides a column-indexed interface compatible
with the old JuMP 0.18 API that SNGO's algorithm code depends on.

Fields mirror the old `Model` fields:
  - `numCols`, `colLower`, `colUpper`, `colVal`, `colCat`, `colNames`
  - `linconstr`, `quadconstr`
  - `obj`, `objSense`
  - `ext` (extension dictionary)
"""
mutable struct ModelWrapper
    # JuMP model (used for actual solves)
    model::Union{JuMP.Model, Nothing}

    # Variable data (column-indexed)
    numCols::Int
    colLower::Vector{Float64}
    colUpper::Vector{Float64}
    colVal::Vector{Float64}
    colCat::Vector{Symbol}              # :Cont, :Bin, :Int
    colNames::Vector{String}
    colNamesIJulia::Vector{String}

    # Constraints (stored in our data format for manipulation)
    linconstr::Vector{LinearConstraintData}
    quadconstr::Vector{QuadConstraintData}

    # Objective
    obj::QuadExprData           # quadratic objective expression
    objSense::Symbol            # :Min or :Max

    # Extension dictionary
    ext::Dict{Symbol, Any}

    # Solution data
    linconstrDuals::Vector{Float64}
    redCosts::Vector{Float64}
    objVal::Float64

    # NLP data reference (for nonlinear constraints)
    nlpdata::Any
    internalModelLoaded::Bool
    nlconstr::Vector{Any}       # nonlinear constraint expressions
end

function ModelWrapper()
    ModelWrapper(
        nothing,        # model
        0,              # numCols
        Float64[],      # colLower
        Float64[],      # colUpper
        Float64[],      # colVal
        Symbol[],       # colCat
        String[],       # colNames
        String[],       # colNamesIJulia
        LinearConstraintData[],  # linconstr
        QuadConstraintData[],    # quadconstr
        QuadExprData(),          # obj
        :Min,                    # objSense
        Dict{Symbol, Any}(),     # ext
        Float64[],               # linconstrDuals
        Float64[],               # redCosts
        0.0,                     # objVal
        nothing,                 # nlpdata
        false,                   # internalModelLoaded
        Any[]                    # nlconstr
    )
end

# ─────────────────────────────────────────────────────────────────────
# Variable management
# ─────────────────────────────────────────────────────────────────────

"""
    add_variable!(mw, lb, ub, cat, name, start_val) -> col_index

Add a variable to the ModelWrapper and return its column index.
"""
function add_variable!(mw::ModelWrapper, lb::Float64, ub::Float64,
                       cat::Symbol=:Cont, name::String="",
                       start_val::Float64=NaN)
    mw.numCols += 1
    push!(mw.colLower, lb)
    push!(mw.colUpper, ub)
    push!(mw.colCat, cat)
    push!(mw.colNames, name)
    push!(mw.colNamesIJulia, name)
    push!(mw.colVal, isnan(start_val) ? 0.0 : start_val)
    return mw.numCols
end

"""
    fixVar!(mw, col, val)

Fix variable at column `col` to value `val`.
"""
function fixVar!(mw::ModelWrapper, col::Int, val::Float64)
    mw.colLower[col] = val
    mw.colUpper[col] = val
    mw.colCat[col] = :Fixed
end

"""
    getlowerbound(mw, col)
    getupperbound(mw, col)
    setlowerbound!(mw, col, val)
    setupperbound!(mw, col, val)
"""
getlowerbound(mw::ModelWrapper, col::Int) = mw.colLower[col]
getupperbound(mw::ModelWrapper, col::Int) = mw.colUpper[col]

function setlowerbound!(mw::ModelWrapper, col::Int, val::Float64)
    mw.colLower[col] = val
end
function setupperbound!(mw::ModelWrapper, col::Int, val::Float64)
    mw.colUpper[col] = val
end

# Convenience: work directly with column index (like old Variable(m, i))
getlowerbound_col(mw::ModelWrapper, col::Int) = mw.colLower[col]
getupperbound_col(mw::ModelWrapper, col::Int) = mw.colUpper[col]

# ─────────────────────────────────────────────────────────────────────
# Model copy — replaces old copyModel
# ─────────────────────────────────────────────────────────────────────

"""
    copyModel(P::ModelWrapper) -> ModelWrapper

Deep copy a ModelWrapper, preserving all variable data, constraints,
objective, and extension dictionary.
"""
function copyModel(P::ModelWrapper)
    mw = ModelWrapper()
    mw.numCols = P.numCols
    mw.colLower = copy(P.colLower)
    mw.colUpper = copy(P.colUpper)
    mw.colVal = copy(P.colVal)
    mw.colCat = copy(P.colCat)
    mw.colNames = copy(P.colNames)
    mw.colNamesIJulia = copy(P.colNamesIJulia)
    mw.linconstr = [copy(c) for c in P.linconstr]
    mw.quadconstr = [copy(c) for c in P.quadconstr]
    mw.obj = copy(P.obj)
    mw.objSense = P.objSense
    mw.linconstrDuals = copy(P.linconstrDuals)
    mw.redCosts = copy(P.redCosts)
    mw.objVal = P.objVal
    mw.nlconstr = [copy_expr(e) for e in P.nlconstr]

    # Deep copy extension dict
    mw.ext = Dict{Symbol, Any}()
    for (key, val) in P.ext
        try
            mw.ext[key] = deepcopy(val)
        catch
            # silently skip un-copyable extensions
        end
    end

    return mw
end

"""
    copy_expr(e)

Deep-copy a Julia expression or pass-through non-expression types.
"""
copy_expr(e::Expr) = deepcopy(e)
copy_expr(e) = e

# ─────────────────────────────────────────────────────────────────────
# Model building — build JuMP model from ModelWrapper for solving
# ─────────────────────────────────────────────────────────────────────

"""
    build_jump_model!(mw, optimizer_factory) -> JuMP.Model

Construct a JuMP Model from the ModelWrapper data structures.
This is called before each solve to sync the wrapper state with JuMP.
"""
function build_jump_model!(mw::ModelWrapper, optimizer_factory)
    model = JuMP.Model(optimizer_factory)
    JuMP.set_silent(model)

    n = mw.numCols
    # Create variables matching the column-indexed data
    vars = Vector{JuMP.VariableRef}(undef, n)
    for i in 1:n
        if mw.colCat[i] == :Bin
            vars[i] = @variable(model, binary=true)
        elseif mw.colCat[i] == :Int
            vars[i] = @variable(model, integer=true)
        elseif mw.colCat[i] == :Fixed
            vars[i] = @variable(model)
            JuMP.fix(vars[i], mw.colLower[i])
        else
            vars[i] = @variable(model)
        end

        # Set bounds (unless fixed)
        if mw.colCat[i] != :Fixed
            if isfinite(mw.colLower[i])
                JuMP.set_lower_bound(vars[i], mw.colLower[i])
            end
            if isfinite(mw.colUpper[i])
                JuMP.set_upper_bound(vars[i], mw.colUpper[i])
            end
        end

        # Set name
        if !isempty(mw.colNames[i])
            JuMP.set_name(vars[i], mw.colNames[i])
        end

        # Set start value
        if !isnan(mw.colVal[i])
            JuMP.set_start_value(vars[i], mw.colVal[i])
        end
    end

    # Add linear constraints
    for lc in mw.linconstr
        expr = JuMP.AffExpr(lc.terms.constant)
        for k in 1:length(lc.terms.vars)
            JuMP.add_to_expression!(expr, lc.terms.coeffs[k], vars[lc.terms.vars[k]])
        end

        if lc.lb == lc.ub
            @constraint(model, expr == lc.lb)
        elseif isfinite(lc.lb) && isfinite(lc.ub)
            @constraint(model, lc.lb <= expr <= lc.ub)
        elseif isfinite(lc.lb)
            @constraint(model, expr >= lc.lb)
        elseif isfinite(lc.ub)
            @constraint(model, expr <= lc.ub)
        end
    end

    # Add quadratic constraints
    for qc in mw.quadconstr
        expr = JuMP.QuadExpr()
        # Quadratic terms
        for k in 1:length(qc.terms.qvars1)
            JuMP.add_to_expression!(expr, qc.terms.qcoeffs[k],
                                    vars[qc.terms.qvars1[k]],
                                    vars[qc.terms.qvars2[k]])
        end
        # Affine terms
        for k in 1:length(qc.terms.aff.vars)
            JuMP.add_to_expression!(expr, qc.terms.aff.coeffs[k],
                                    vars[qc.terms.aff.vars[k]])
        end
        JuMP.add_to_expression!(expr, qc.terms.aff.constant)

        if qc.sense == :(<=)
            @constraint(model, expr <= 0)
        elseif qc.sense == :(>=)
            @constraint(model, expr >= 0)
        else
            @constraint(model, expr == 0)
        end
    end

    # Add nonlinear constraints
    for nlexpr in mw.nlconstr
        _add_nl_constraint!(model, nlexpr, vars)
    end

    # Set objective
    obj_expr = JuMP.QuadExpr()
    for k in 1:length(mw.obj.qvars1)
        JuMP.add_to_expression!(obj_expr, mw.obj.qcoeffs[k],
                                vars[mw.obj.qvars1[k]],
                                vars[mw.obj.qvars2[k]])
    end
    for k in 1:length(mw.obj.aff.vars)
        JuMP.add_to_expression!(obj_expr, mw.obj.aff.coeffs[k],
                                vars[mw.obj.aff.vars[k]])
    end
    JuMP.add_to_expression!(obj_expr, mw.obj.aff.constant)

    if mw.objSense == :Min
        @objective(model, Min, obj_expr)
    else
        @objective(model, Max, obj_expr)
    end

    mw.model = model
    return model
end

"""
    _add_nl_constraint!(model, expr, vars)

Add a nonlinear constraint expression to the JuMP model.
The expression is in the old MathProgBase format:
    comparison(lhs_expr, 0.0) where comparison ∈ {<=, >=, ==}
"""
function _add_nl_constraint!(model::JuMP.Model, expr::Expr, vars::Vector{JuMP.VariableRef})
    # Expression format: :(call, op, lhs, 0.0)
    if !isa(expr, Expr) || expr.head != :call
        return
    end
    op = expr.args[1]
    lhs = _substitute_vars(expr.args[2], vars)
    # Build NL constraint via macro
    if op == :(<=) || op == :(≤)
        JuMP.add_NL_constraint(model, :($lhs <= 0.0))
    elseif op == :(>=) || op == :(≥)
        JuMP.add_NL_constraint(model, :($lhs >= 0.0))
    elseif op == :(==)
        JuMP.add_NL_constraint(model, :($lhs == 0.0))
    end
end

"""
    _substitute_vars(expr, vars)

Replace :(x[i]) references in an expression tree with JuMP.VariableRef objects.
"""
function _substitute_vars(expr, vars::Vector{JuMP.VariableRef})
    if isa(expr, Expr)
        if expr.head == :ref
            idx = expr.args[2]
            return vars[idx]
        else
            new_args = [_substitute_vars(a, vars) for a in expr.args]
            return Expr(expr.head, new_args...)
        end
    else
        return expr
    end
end

# ─────────────────────────────────────────────────────────────────────
# Solve interface
# ─────────────────────────────────────────────────────────────────────

"""
    solve_model!(mw, optimizer_factory) -> Symbol

Build a JuMP model from the wrapper, solve it, and cache results back.
Returns a status symbol: :Optimal, :Infeasible, :Unbounded, etc.
"""
function solve_model!(mw::ModelWrapper, optimizer_factory)
    model = build_jump_model!(mw, optimizer_factory)
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    status_sym = _status_to_symbol(status)

    if status_sym == :Optimal
        # Cache solution values
        vars = JuMP.all_variables(model)
        for (i, v) in enumerate(vars)
            if i <= mw.numCols
                mw.colVal[i] = JuMP.value(v)
            end
        end
        mw.objVal = JuMP.objective_value(model)

        # Cache reduced costs if available (LP)
        try
            mw.redCosts = [JuMP.reduced_cost(v) for v in vars[1:min(mw.numCols, length(vars))]]
        catch
            mw.redCosts = Float64[]
        end

        # Cache duals if available
        try
            cons = JuMP.all_constraints(model, include_variable_in_set_constraints=false)
            # simplified — just resize
            mw.linconstrDuals = zeros(length(mw.linconstr))
        catch
            mw.linconstrDuals = Float64[]
        end
    end

    return status_sym
end

function _status_to_symbol(status::MOI.TerminationStatusCode)
    if status == MOI.OPTIMAL
        return :Optimal
    elseif status == MOI.INFEASIBLE
        return :Infeasible
    elseif status == MOI.DUAL_INFEASIBLE
        return :Unbounded
    elseif status == MOI.TIME_LIMIT
        return :TimeLimit
    elseif status == MOI.LOCALLY_SOLVED
        return :Optimal  # treat local optimal as optimal for NLP
    elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
        return :Infeasible
    else
        return :Error
    end
end

"""
    getobjectivevalue(mw) -> Float64

Return the cached objective value from the last solve.
"""
getobjectivevalue(mw::ModelWrapper) = mw.objVal

"""
    getRobjective(mw, status, default_LB)

Get the relaxation objective value, handling infeasible/error cases.
"""
function getRobjective(mw::ModelWrapper, status::Symbol, default_LB::Float64=-1e20)
    if status == :Optimal
        return mw.objVal
    else
        return default_LB
    end
end

# ─────────────────────────────────────────────────────────────────────
# Affine expression evaluation
# ─────────────────────────────────────────────────────────────────────

"""
    eval_aff(aff::AffExprData, colVal) -> Float64

Evaluate an affine expression at the given variable values.
"""
function eval_aff(aff::AffExprData, colVal::Vector{Float64})
    val = aff.constant
    for k in 1:length(aff.vars)
        val += aff.coeffs[k] * colVal[aff.vars[k]]
    end
    return val
end

"""
    eval_quad(q::QuadExprData, colVal) -> Float64

Evaluate a quadratic expression at the given variable values.
"""
function eval_quad(q::QuadExprData, colVal::Vector{Float64})
    val = eval_aff(q.aff, colVal)
    for k in 1:length(q.qvars1)
        val += q.qcoeffs[k] * colVal[q.qvars1[k]] * colVal[q.qvars2[k]]
    end
    return val
end

# Alias for old eval_g
eval_g(q::QuadExprData, colVal::Vector{Float64}) = eval_quad(q, colVal)

# ─────────────────────────────────────────────────────────────────────
# Constraint manipulation helpers
# ─────────────────────────────────────────────────────────────────────

"""
    add_linear_constraint!(mw, aff, lb, ub)

Add a linear constraint lb ≤ aff ≤ ub to the model wrapper.
"""
function add_linear_constraint!(mw::ModelWrapper, aff::AffExprData,
                                 lb::Float64, ub::Float64)
    push!(mw.linconstr, LinearConstraintData(copy(aff), lb, ub))
    return length(mw.linconstr)
end

"""
    add_quad_constraint!(mw, qexpr, sense)

Add a quadratic constraint to the model wrapper.
"""
function add_quad_constraint!(mw::ModelWrapper, qexpr::QuadExprData, sense::Symbol)
    push!(mw.quadconstr, QuadConstraintData(copy(qexpr), sense))
    return length(mw.quadconstr)
end

# ─────────────────────────────────────────────────────────────────────
# AffExprData manipulation — mirroring old JuMP AffExpr operations
# ─────────────────────────────────────────────────────────────────────

"""
    copy_aff_to_model(aff::AffExprData, col_offset::Int) -> AffExprData

Copy an affine expression, offsetting all variable column indices.
"""
function copy_aff_to_model(aff::AffExprData, col_offset::Int)
    new_aff = copy(aff)
    for k in 1:length(new_aff.vars)
        new_aff.vars[k] += col_offset
    end
    return new_aff
end

"""
    copy_aff_with_map(aff::AffExprData, v_map::Vector{Int}) -> AffExprData

Copy an affine expression, remapping variable indices via v_map.
"""
function copy_aff_with_map(aff::AffExprData, v_map::Vector{Int})
    new_aff = copy(aff)
    for k in 1:length(new_aff.vars)
        new_aff.vars[k] = v_map[new_aff.vars[k]]
    end
    return new_aff
end

"""
    copy_lincon_with_map(con::LinearConstraintData, v_map) -> LinearConstraintData
"""
function copy_lincon_with_map(con::LinearConstraintData, v_map::Vector{Int})
    LinearConstraintData(copy_aff_with_map(con.terms, v_map), con.lb, con.ub)
end

"""
    copy_quadcon_with_map(con::QuadConstraintData, v_map) -> QuadConstraintData
"""
function copy_quadcon_with_map(con::QuadConstraintData, v_map::Vector{Int})
    new_q = copy(con.terms)
    for k in 1:length(new_q.qvars1)
        new_q.qvars1[k] = v_map[new_q.qvars1[k]]
        new_q.qvars2[k] = v_map[new_q.qvars2[k]]
    end
    new_q.aff = copy_aff_with_map(con.terms.aff, v_map)
    QuadConstraintData(new_q, con.sense)
end

# ─────────────────────────────────────────────────────────────────────
# Squeeze — combine duplicate terms (from old preprocess.jl)
# ─────────────────────────────────────────────────────────────────────

"""
    squeeze!(aff::AffExprData)

Combine duplicate variable entries in an affine expression.
"""
function squeeze!(aff::AffExprData)
    redund = true
    while redund
        redund = false
        for k in 1:length(aff.vars)
            x = aff.vars[k]
            equal_index = findall(v -> v == x, aff.vars)
            if length(equal_index) > 1
                aff.coeffs[k] = sum(aff.coeffs[equal_index])
                deleteat!(aff.vars, equal_index[2:end])
                deleteat!(aff.coeffs, equal_index[2:end])
                redund = true
                break
            end
        end
    end
end

"""
    squeeze!(q::QuadExprData)

Combine duplicate entries in a quadratic expression.
"""
function squeeze!(q::QuadExprData)
    redund = true
    while redund
        redund = false
        for k in 1:length(q.qvars1)
            x = q.qvars1[k]
            y = q.qvars2[k]
            equal_index = Int[]
            coeff = 0.0
            for j in 1:length(q.qvars1)
                if (q.qvars1[j] == x && q.qvars2[j] == y) ||
                   (q.qvars1[j] == y && q.qvars2[j] == x)
                    push!(equal_index, j)
                    coeff += q.qcoeffs[j]
                end
            end
            if length(equal_index) > 1
                q.qcoeffs[k] = coeff
                deleteat!(q.qvars1, equal_index[2:end])
                deleteat!(q.qvars2, equal_index[2:end])
                deleteat!(q.qcoeffs, equal_index[2:end])
                redund = true
                break
            end
        end
    end
    squeeze!(q.aff)
end

# ─────────────────────────────────────────────────────────────────────
# Model import from JuMP — parse an existing JuMP model into wrapper
# ─────────────────────────────────────────────────────────────────────

"""
    import_jump_model(m::JuMP.Model) -> ModelWrapper

Import a JuMP Model into a ModelWrapper, extracting variables,
linear constraints, quadratic constraints, and objective.
"""
function import_jump_model(m::JuMP.Model)
    mw = ModelWrapper()

    # Import variables
    all_vars = JuMP.all_variables(m)
    mw.numCols = length(all_vars)
    mw.colLower = Float64[]
    mw.colUpper = Float64[]
    mw.colVal = Float64[]
    mw.colCat = Symbol[]
    mw.colNames = String[]
    mw.colNamesIJulia = String[]

    for v in all_vars
        # Bounds
        lb = JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
        ub = JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
        push!(mw.colLower, lb)
        push!(mw.colUpper, ub)

        # Start value
        sv = JuMP.start_value(v)
        push!(mw.colVal, sv === nothing ? NaN : sv)

        # Category
        if JuMP.is_binary(v)
            push!(mw.colCat, :Bin)
        elseif JuMP.is_integer(v)
            push!(mw.colCat, :Int)
        elseif JuMP.is_fixed(v)
            push!(mw.colCat, :Fixed)
        else
            push!(mw.colCat, :Cont)
        end

        # Name
        nm = JuMP.name(v)
        push!(mw.colNames, nm)
        push!(mw.colNamesIJulia, nm)
    end

    # Build a mapping from JuMP VariableRef -> column index
    var_to_col = Dict{JuMP.VariableRef, Int}()
    for (i, v) in enumerate(all_vars)
        var_to_col[v] = i
    end

    # Import linear constraints
    mw.linconstr = LinearConstraintData[]
    # Equality constraints: AffExpr == constant
    for ci in JuMP.all_constraints(m, JuMP.AffExpr, MOI.EqualTo{Float64})
        con = JuMP.constraint_object(ci)
        aff = _jump_aff_to_data(con.func, var_to_col)
        rhs = con.set.value
        # Move constant: aff_data includes constant, constraint is aff == rhs
        # so effective: aff_no_const == rhs - constant
        actual_rhs = rhs - aff.constant
        aff.constant = 0.0
        push!(mw.linconstr, LinearConstraintData(aff, actual_rhs, actual_rhs))
    end
    # LessThan constraints
    for ci in JuMP.all_constraints(m, JuMP.AffExpr, MOI.LessThan{Float64})
        con = JuMP.constraint_object(ci)
        aff = _jump_aff_to_data(con.func, var_to_col)
        rhs = con.set.upper - aff.constant
        aff.constant = 0.0
        push!(mw.linconstr, LinearConstraintData(aff, -Inf, rhs))
    end
    # GreaterThan constraints
    for ci in JuMP.all_constraints(m, JuMP.AffExpr, MOI.GreaterThan{Float64})
        con = JuMP.constraint_object(ci)
        aff = _jump_aff_to_data(con.func, var_to_col)
        rhs = con.set.lower - aff.constant
        aff.constant = 0.0
        push!(mw.linconstr, LinearConstraintData(aff, rhs, Inf))
    end

    # Import quadratic constraints
    mw.quadconstr = QuadConstraintData[]
    for ci in JuMP.all_constraints(m, JuMP.QuadExpr, MOI.LessThan{Float64})
        con = JuMP.constraint_object(ci)
        qd = _jump_quad_to_data(con.func, var_to_col)
        push!(mw.quadconstr, QuadConstraintData(qd, :(<=)))
    end
    for ci in JuMP.all_constraints(m, JuMP.QuadExpr, MOI.GreaterThan{Float64})
        con = JuMP.constraint_object(ci)
        qd = _jump_quad_to_data(con.func, var_to_col)
        push!(mw.quadconstr, QuadConstraintData(qd, :(>=)))
    end
    for ci in JuMP.all_constraints(m, JuMP.QuadExpr, MOI.EqualTo{Float64})
        con = JuMP.constraint_object(ci)
        qd = _jump_quad_to_data(con.func, var_to_col)
        push!(mw.quadconstr, QuadConstraintData(qd, :(==)))
    end

    # Import objective
    obj_func = JuMP.objective_function(m)
    if obj_func isa JuMP.QuadExpr
        mw.obj = _jump_quad_to_data(obj_func, var_to_col)
    elseif obj_func isa JuMP.AffExpr
        aff_data = _jump_aff_to_data(obj_func, var_to_col)
        mw.obj = QuadExprData(Int[], Int[], Float64[], aff_data)
    elseif obj_func isa JuMP.VariableRef
        aff_data = AffExprData([var_to_col[obj_func]], [1.0], 0.0)
        mw.obj = QuadExprData(Int[], Int[], Float64[], aff_data)
    end
    mw.objSense = JuMP.objective_sense(m) == MOI.MIN_SENSE ? :Min : :Max

    # Init duals/redcosts
    mw.linconstrDuals = zeros(length(mw.linconstr))
    mw.redCosts = zeros(mw.numCols)
    mw.objVal = 0.0
    mw.model = m

    return mw
end

function _jump_aff_to_data(aff::JuMP.AffExpr, var_to_col::Dict{JuMP.VariableRef, Int})
    vars = Int[]
    coeffs = Float64[]
    for (v, c) in aff.terms
        push!(vars, var_to_col[v])
        push!(coeffs, c)
    end
    AffExprData(vars, coeffs, aff.constant)
end

function _jump_quad_to_data(quad::JuMP.QuadExpr, var_to_col::Dict{JuMP.VariableRef, Int})
    qvars1 = Int[]
    qvars2 = Int[]
    qcoeffs = Float64[]
    for (pair, c) in quad.terms
        push!(qvars1, var_to_col[pair.a])
        push!(qvars2, var_to_col[pair.b])
        push!(qcoeffs, c)
    end
    aff_data = _jump_aff_to_data(quad.aff, var_to_col)
    QuadExprData(qvars1, qvars2, qcoeffs, aff_data)
end

# ─────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────

"""
    bounded(x) -> Bool

Check if a value is finite (not ±Inf, not NaN).
"""
bounded(x::Float64) = isfinite(x)
bounded(x) = isfinite(Float64(x))
