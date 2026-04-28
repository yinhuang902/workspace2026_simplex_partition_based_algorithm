# ─────────────────────────────────────────────────────────────────────
# Run Logger for SNGO-Modern
# ─────────────────────────────────────────────────────────────────────

using Printf
using Dates

mutable struct RunLogger
    logdir::String
    runid::String
    test_name::String
    txt_path::String
    csv_path::String
    io_txt::Union{IO, Nothing}

    # Tracking variables
    root_data::Dict{Symbol, Any}
    meta_data::Dict{Symbol, Any}
    final_data::Dict{Symbol, Any}
    iter_count::Int
end

function init_logger(P::ModelWrapper, scenarios, pr_children)
    logdir = joinpath(dirname(@__DIR__), "run_logs")
    if !isdir(logdir)
        mkdir(logdir)
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    script_path = PROGRAM_FILE
    test_name = isempty(script_path) ? "run" : basename(script_path)
    runid = "$(test_name)_$(timestamp)"

    txt_path = joinpath(logdir, "$(runid)_debug.txt")
    csv_path = joinpath(logdir, "$(runid)_summary.csv")

    io_txt = open(txt_path, "w")

    logger = RunLogger(logdir, runid, test_name, txt_path, csv_path, io_txt, 
                       Dict{Symbol,Any}(), Dict{Symbol,Any}(), Dict{Symbol,Any}(), 0)

    # Scrape metadata
    # (A) Environment
    logger.meta_data[:timestamp] = timestamp
    logger.meta_data[:test_name] = test_name
    logger.meta_data[:working_dir] = pwd()
    logger.meta_data[:julia_version] = string(VERSION)
    
    # (B) Stochastic Setting (from P.ext[:stochastic_params])
    sto_params = get(P.ext, :stochastic_params, Dict{String,Any}())
    logger.meta_data[:NS] = get(sto_params, "NS", length(scenarios))
    logger.meta_data[:nfirst] = get(sto_params, "nfirst", P.numCols)
    logger.meta_data[:nparam] = get(sto_params, "nparam", -1)
    logger.meta_data[:rdl] = get(sto_params, "rdl", NaN)
    logger.meta_data[:rdu] = get(sto_params, "rdu", NaN)
    logger.meta_data[:adl] = get(sto_params, "adl", NaN)
    logger.meta_data[:adu] = get(sto_params, "adu", NaN)

    # Write textual metadata
    println(io_txt, "="^80)
    println(io_txt, "SNGO-Modern Run Log: ", runid)
    println(io_txt, "="^80)
    println(io_txt, "Timestamp:        ", timestamp)
    println(io_txt, "Test Name:        ", test_name)
    println(io_txt, "Working Dir:      ", pwd())
    println(io_txt, "Julia Version:    ", VERSION)
    println(io_txt, "-"^80)
    println(io_txt, "Solver Constants:")
    println(io_txt, "  machine_error:             ", machine_error)
    println(io_txt, "  small_bound_improve:       ", small_bound_improve)
    println(io_txt, "  large_bound_improve:       ", large_bound_improve)
    println(io_txt, "  probing_improve:           ", probing_improve)
    println(io_txt, "  sigma_violation:           ", sigma_violation)
    println(io_txt, "  local_obj_improve:         ", local_obj_improve)
    println(io_txt, "  mingap:                    ", mingap)
    println(io_txt, "  default_lower_bound_value: ", default_lower_bound_value)
    println(io_txt, "  default_upper_bound_value: ", default_upper_bound_value)
    println(io_txt, "-"^80)
    println(io_txt, "Problem Setup:")
    println(io_txt, "  Scenarios:                 ", logger.meta_data[:NS])
    println(io_txt, "  First-stage Vars (nfirst): ", logger.meta_data[:nfirst])
    println(io_txt, "  Noise param (nparam):      ", logger.meta_data[:nparam])
    println(io_txt, "  Noise limits (rdl,rdu):    ", logger.meta_data[:rdl], ", ", logger.meta_data[:rdu])
    println(io_txt, "  Additive limits (adl,adu): ", logger.meta_data[:adl], ", ", logger.meta_data[:adu])
    
    # Init blank root data
    logger.root_data[:initial_UB] = NaN
    logger.root_data[:relaxed_LB] = NaN
    logger.root_data[:WS_LB] = NaN
    logger.root_data[:node_LB] = NaN
    logger.root_data[:FBBT_feasible] = false

    flush(logger.io_txt)
    return logger
end

function log_model_sizes!(logger::RunLogger, extensive_num_vars::Int, extensive_num_linconstr::Int,
                          relaxation_num_vars::Int, relaxation_num_constr::Int)
    io = logger.io_txt
    logger.meta_data[:extensive_num_vars] = extensive_num_vars
    logger.meta_data[:extensive_num_linconstr] = extensive_num_linconstr
    logger.meta_data[:relaxation_num_vars] = relaxation_num_vars
    logger.meta_data[:relaxation_num_constr] = relaxation_num_constr
    
    println(io, "-"^80)
    println(io, "Model Sizes:")
    println(io, "  Extensive Variables:       ", extensive_num_vars)
    println(io, "  Extensive Lin Constrs:     ", extensive_num_linconstr)
    println(io, "  Relaxation Variables:      ", relaxation_num_vars)
    println(io, "  Relaxation Constraints:    ", relaxation_num_constr)
    flush(io)
end

function log_root_info!(logger::RunLogger, initial_UB::Float64, root_node_LB::Float64,
                        FBBT_feasible::Bool, relaxed_status::Symbol, relaxed_LB::Float64,
                        WS_status::Symbol, WS_LB::Float64,
                        local_UB::Float64, WSfix_UB::Float64,
                        event_note::String)
    io = logger.io_txt
    logger.root_data[:initial_UB] = initial_UB
    logger.root_data[:relaxed_LB] = relaxed_LB
    logger.root_data[:WS_LB] = WS_LB
    logger.root_data[:node_LB] = root_node_LB
    logger.root_data[:FBBT_feasible] = FBBT_feasible

    println(io, "-"^80)
    println(io, "Root Node Diagnostics:")
    println(io, "  Initial UB (Multi-start):  ", initial_UB)
    println(io, "  FBBT Feasible:             ", FBBT_feasible)
    println(io, "  Relaxation Status:         ", relaxed_status)
    println(io, "  Relaxation LB:             ", relaxed_LB)
    println(io, "  Wait-and-See (WS) Status:  ", WS_status)
    println(io, "  Wait-and-See (WS) LB:      ", WS_LB)
    println(io, "  Local UB:                  ", local_UB)
    println(io, "  WSfix UB:                  ", WSfix_UB)
    println(io, "  Final Root Node LB:        ", root_node_LB)
    println(io, "  Root Event Note / Term:    ", event_note)
    println(io, "-"^80)
    println(io, "Iterations:")
    @printf(io, "%6s | %10s | %14s | %14s | %10s | %6s | %18s | %s\n",
            "Iter", "Time(s)", "Lower Bound", "Upper Bound", "Gap(%)", "Nodes", "Branching (Id,Val)", "Note")
    flush(io)
end

function log_iteration!(logger::RunLogger, iter::Int, elapsed::Float64, LB::Float64, UB::Float64, 
                        gap_pct::Float64, queue_size::Int, bVarId::Int, bValue::Float64, event_note::String)
    io = logger.io_txt
    logger.iter_count = iter
    branch_str = bVarId > 0 ? @sprintf("v%d=%.4f", bVarId, bValue) : "none"
    @printf(io, "%6d | %10.2f | %14.4f | %14.4f | %10.4f | %6d | %18s | %s\n",
            iter, elapsed, LB, UB, gap_pct, queue_size, branch_str, event_note)
    
    if iter % 10 == 0
        flush(io)
    end
end

function log_error!(logger::RunLogger, e::Exception, bt)
    io = logger.io_txt
    println(io, "\n", "!"^80)
    println(io, "EXCEPTION CAUGHT DURING EXECUTION:")
    showerror(io, e, bt)
    println(io, "\n", "!"^80)
    flush(io)
end

function finish_logger!(logger::RunLogger, LB::Float64, UB::Float64, gap_pct::Float64, 
                        total_time::Float64, solved_nodes::Int, best_x::Vector{Float64}, 
                        termination_reason::String, final_LB_postprocessed::Bool)
    io = logger.io_txt
    
    logger.final_data[:final_LB] = LB
    logger.final_data[:final_UB] = UB
    logger.final_data[:final_gap_pct] = gap_pct
    logger.final_data[:iterations] = logger.iter_count
    logger.final_data[:solved_nodes] = solved_nodes
    logger.final_data[:total_time] = total_time
    logger.final_data[:termination_reason] = termination_reason
    logger.final_data[:final_LB_postprocessed] = final_LB_postprocessed

    println(io, "-"^80)
    println(io, "Final Results Block:")
    println(io, "  Final Lower Bound:         ", LB)
    println(io, "  Final Upper Bound:         ", UB)
    println(io, "  Final Gap:                 ", gap_pct, " %")
    println(io, "  Iterations:                ", logger.iter_count)
    println(io, "  Solved Nodes:              ", solved_nodes)
    println(io, "  Total Time:                ", total_time, " seconds")
    println(io, "  Termination Reason:        ", termination_reason)
    println(io, "  LB Postprocessed:          ", final_LB_postprocessed)
    println(io, "  Best First-Stage Sol:      ", best_x)
    println(io, "="^80)
    
    # Safely close text logging
    try
        close(io)
    catch
    end

    # Write CSV summary
    open(logger.csv_path, "w") do f
        header = [
            "timestamp", "test_name", "NS", "nfirst", "nparam", 
            "rdl", "rdu", "adl", "adu",
            "machine_error", "default_lower_bound_value", "default_upper_bound_value",
            "extensive_num_vars", "extensive_num_linconstr", 
            "relaxation_num_vars", "relaxation_num_constr",
            "initial_UB", "root_node_LB", "root_FBBT_feasible", "root_relaxed_LB", "root_WS_LB",
            "final_LB", "final_UB", "final_gap_pct", "iterations",
            "solved_nodes", "total_time", "termination_reason", "final_LB_postprocessed",
            "best_first_stage_solution"
        ]
        
        row = Any[
            logger.meta_data[:timestamp], 
            logger.meta_data[:test_name],
            logger.meta_data[:NS], 
            logger.meta_data[:nfirst], 
            logger.meta_data[:nparam],
            logger.meta_data[:rdl], logger.meta_data[:rdu], 
            logger.meta_data[:adl], logger.meta_data[:adu],
            machine_error, default_lower_bound_value, default_upper_bound_value,
            get(logger.meta_data, :extensive_num_vars, -1),
            get(logger.meta_data, :extensive_num_linconstr, -1),
            get(logger.meta_data, :relaxation_num_vars, -1),
            get(logger.meta_data, :relaxation_num_constr, -1),
            logger.root_data[:initial_UB], 
            logger.root_data[:node_LB],
            logger.root_data[:FBBT_feasible],
            logger.root_data[:relaxed_LB], 
            logger.root_data[:WS_LB],
            LB, UB, gap_pct, logger.iter_count,
            solved_nodes, total_time, termination_reason, final_LB_postprocessed,
            "\"$(best_x)\"" # Wrap vector in quotes to not break CSV
        ]
        
        println(f, join(header, ","))
        println(f, join(map(x -> string(x), row), ","))
    end
end
