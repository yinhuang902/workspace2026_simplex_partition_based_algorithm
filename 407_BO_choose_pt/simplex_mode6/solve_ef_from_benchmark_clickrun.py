"""
solve_ef_from_benchmark_clickrun.py

Double-click / direct-run version of the EF solver.

Behavior
--------
- No command-line arguments required.
- Assumes this script is placed in the SAME folder as the benchmark file.
- By default it looks for:
      run_st_center2_gurobi_multicluster_case.py
  and if that file does not exist, it falls back to:
      run_st_center3_gurobi_multicluster_case.py
- Regenerates the same scenario data from (nscen, seed),
  builds an extensive-form model, and solves it with Gurobi.
- Prints the EF objective value and the corresponding first-stage point x*.

Edit the CONFIG block below if needed.
"""
from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path
from typing import Any, List

import pyomo.environ as pyo


# =============================================================================
# CONFIG: edit these if needed
# =============================================================================
NSCEN = 2
SEED = 1234
NFIRST = 3
NPARAM = 10

OBJECTIVE_SCALE = "avg"   # "avg" or "sum"
SOLVER_NAME = "gurobi"
TIME_LIMIT = 600.0
MIPGAP = 1e-6
MIPGAPABS = 1e-6

PRINT_SCENARIO_Y = False
TEE = True

# Benchmark file is assumed to be in the same folder as this script.
BENCHMARK_CANDIDATES = [
    "run_st_center2_gurobi_multicluster_case.py",
    #"run_st_center3_gurobi_multicluster_case.py",
]


# =============================================================================
# Helpers
# =============================================================================
def load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_scenario_builder(mod: Any):
    candidates = [
        "build_models_center3_gurobi",
        "build_models_center2_gurobi",
    ]
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name), name
    raise AttributeError(
        "Could not find a supported scenario builder in the benchmark file. "
        "Expected one of: " + ", ".join(candidates)
    )


def first_stage_var_names(model: pyo.ConcreteModel) -> List[str]:
    out = []
    for name in ("x1", "x2", "x3"):
        if hasattr(model, name):
            out.append(name)
    if not out:
        raise ValueError("Could not detect first-stage variable names x1/x2/x3.")
    return out


def scenario_objective_expr(model: pyo.ConcreteModel):
    if hasattr(model, "obj_expr"):
        return model.obj_expr
    if hasattr(model, "obj"):
        obj = model.obj
        if hasattr(obj, "expr"):
            return obj.expr
    raise ValueError("Could not find obj_expr or obj on scenario model.")


def build_extensive_form(scenario_models: List[pyo.ConcreteModel], objective_scale: str = "avg"):
    if not scenario_models:
        raise ValueError("scenario_models is empty.")

    ef = pyo.ConcreteModel()
    S = len(scenario_models)
    ef.S = pyo.RangeSet(0, S - 1)

    ef.scen = pyo.Block(ef.S)
    for s, m in enumerate(scenario_models):
        ef.scen[s].transfer_attributes_from(m)

    xnames = first_stage_var_names(ef.scen[0])

    ef.nac = pyo.ConstraintList()
    for s in range(1, S):
        for xn in xnames:
            ef.nac.add(getattr(ef.scen[s], xn) == getattr(ef.scen[0], xn))

    total_expr = sum(scenario_objective_expr(ef.scen[s]) for s in range(S))
    if objective_scale == "avg":
        total_expr = total_expr / S

    ef.obj = pyo.Objective(expr=total_expr, sense=pyo.minimize)
    return ef, xnames


def solve_ef(
    ef: pyo.ConcreteModel,
    solver_name: str,
    time_limit: float,
    mipgap: float,
    mipgapabs: float,
    tee: bool,
):
    solver = pyo.SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create solver '{solver_name}'.")

    if solver_name.lower() == "gurobi":
        solver.options["NonConvex"] = 2
        solver.options["TimeLimit"] = float(time_limit)
        solver.options["MIPGap"] = float(mipgap)
        solver.options["MIPGapAbs"] = float(mipgapabs)

    results = solver.solve(ef, tee=tee)
    return results


def value_or_none(v):
    try:
        return pyo.value(v)
    except Exception:
        return None


def resolve_benchmark_file(script_dir: Path) -> Path:
    for name in BENCHMARK_CANDIDATES:
        p = script_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find any benchmark file in the same folder as this script.\n"
        f"Tried: {BENCHMARK_CANDIDATES}\n"
        f"Script folder: {script_dir}"
    )


# =============================================================================
# Main
# =============================================================================
def main():
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    benchmark_file = resolve_benchmark_file(script_dir)

    print("=" * 72)
    print("Standalone EF solve from benchmark (click-run version)")
    print(f"script file     : {script_path}")
    print(f"benchmark file  : {benchmark_file}")
    print(f"nscen           : {NSCEN}")
    print(f"seed            : {SEED}")
    print(f"objective_scale : {OBJECTIVE_SCALE}")
    print(f"solver          : {SOLVER_NAME}")
    print("=" * 72)

    mod = load_module_from_file(benchmark_file)
    builder, builder_name = find_scenario_builder(mod)
    print(f"scenario builder: {builder_name}")

    scenario_models, _ = builder(
        nscen=NSCEN,
        nfirst=NFIRST,
        nparam=NPARAM,
        seed=SEED,
        print_first_k_rhs=0,
    )

    ef, xnames = build_extensive_form(scenario_models, objective_scale=OBJECTIVE_SCALE)
    results = solve_ef(
        ef=ef,
        solver_name=SOLVER_NAME,
        time_limit=TIME_LIMIT,
        mipgap=MIPGAP,
        mipgapabs=MIPGAPABS,
        tee=TEE,
    )

    term = getattr(results.solver, "termination_condition", None)
    status = getattr(results.solver, "status", None)

    x_star = {xn: value_or_none(getattr(ef.scen[0], xn)) for xn in xnames}
    ub_val = value_or_none(ef.obj)

    print("\n=== EF solve status ===")
    print(f"solver_status        : {status}")
    print(f"termination_condition: {term}")

    print("\n=== EF objective ===")
    print(f"objective ({OBJECTIVE_SCALE}) = {ub_val}")

    print("\n=== First-stage point x* ===")
    for xn in xnames:
        print(f"{xn} = {x_star[xn]}")

    print("\n=== Scenario objective values at x* ===")
    scen_vals = []
    for s in ef.S:
        val = value_or_none(scenario_objective_expr(ef.scen[s]))
        scen_vals.append(val)
        print(f"s={int(s)}  obj={val}")

    if OBJECTIVE_SCALE == "avg":
        print(f"\ncheck avg = {sum(scen_vals)/len(scen_vals)}")
    else:
        print(f"\ncheck sum = {sum(scen_vals)}")

    if PRINT_SCENARIO_Y:
        print("\n=== Scenario recourse y* ===")
        for s in ef.S:
            vals = []
            for yn in ("y1", "y2", "y3"):
                if hasattr(ef.scen[s], yn):
                    vals.append(f"{yn}={value_or_none(getattr(ef.scen[s], yn))}")
            print(f"s={int(s)}  " + "  ".join(vals))

    print("\nDone.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:")
        print(e)
        print("\nFull traceback:\n")
        traceback.print_exc()
        input("\nPress Enter to exit...")
