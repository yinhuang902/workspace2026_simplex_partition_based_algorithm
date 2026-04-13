"""
run_st_center3_gurobi_multicluster_case.py — Simplex runner for a synthetic
multi-cluster nonconvex QCQP-style two-stage stochastic program with 3
first-stage variables.

Design goals
------------
- First-stage dimension fixed at 3.
- First-stage box is large (about 150 per dimension).
- Scenario-wise optima stay in the interior, but are NOT all clustered at one point.
- Gurobi-compatible: objective is quadratic (possibly indefinite), constraints are linear.
- Scenario generation follows the same style as run_st_fp7a_case.py:
  Julia-compatible RNG, scenario 1 unperturbed, mutable Pyomo Params,
  and per-scenario parameter perturbations.

Model
-----
First-stage:
  x1, x2, x3 in [-75, 75]

Second-stage:
  y1, y2, y3 in [-35, 35]

For each scenario s, solve

    min  alpha * sum_i (x_i - c_i)^2
       + gamma * sum_i (x_i + y_i - (c_i + xi_i))^2
       + eta   * (y1 + y2 + y3)^2
       - beta  * (y1*y2 + y1*y3 + y2*y3)
       + rho   * sum_i shift_i * y_i

    s.t.  -cap_rhs <= y1 + y2 + y3 <= cap_rhs

The first-stage quadratic term pulls x toward a scenario-dependent interior
target c = (c1, c2, c3). Scenario targets are generated from several interior
clusters, so different scenario-wise optima are separated but still remain far
from the box corners. Nonconvexity comes from the negative pairwise quadratic
terms in y, while the model remains a Gurobi-compatible nonconvex QCQP.
"""
import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import pyomo.environ as pyo

from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d


# =============================================================================
# Julia-compatible RNG + noise helpers
# =============================================================================
OBJ_SHIFT = 100


class JuliaMT19937:
    N = 624; M = 397
    MATRIX_A = 0x9908B0DF; UPPER_MASK = 0x80000000; LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int = 1234):
        self.mt = [0] * self.N
        self.mti = self.N + 1
        self.seed(seed)

    def seed(self, seed: int):
        seed &= 0xFFFFFFFF
        self.mt[0] = seed
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = self.N

    def rand_uint32(self) -> int:
        mag01 = [0x0, self.MATRIX_A]
        if self.mti >= self.N:
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + (self.M - self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.N - 1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N - 1] = self.mt[self.M - 1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0
        y = self.mt[self.mti]; self.mti += 1
        y ^= (y >> 11); y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000; y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def rand_uint64(self) -> int:
        return ((self.rand_uint32() << 32) | self.rand_uint32()) & 0xFFFFFFFFFFFFFFFF

    def rand_float64(self) -> float:
        return (self.rand_uint64() >> 12) * (1.0 / (1 << 52))

    def rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rand_float64()


def addnoise_sym(a: float, width: float, rng: JuliaMT19937) -> float:
    return a + rng.rand_uniform(-width, width)


def addnoise_pos(a: float, rel: float, rng: JuliaMT19937, floor: float = 1e-6) -> float:
    val = a * (1.0 + rng.rand_uniform(-rel, rel))
    return max(floor, val)


# =============================================================================
# Deterministic base model
# =============================================================================

def create_model_center3_gurobi() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # First-stage variables (simplex variables)
    m.x1 = pyo.Var(bounds=(-75.0, 75.0), initialize=0.0)
    m.x2 = pyo.Var(bounds=(-75.0, 75.0), initialize=0.0)
    m.x3 = pyo.Var(bounds=(-75.0, 75.0), initialize=0.0)

    # Second-stage recourse variables
    m.y1 = pyo.Var(bounds=(-35.0, 35.0), initialize=0.0)
    m.y2 = pyo.Var(bounds=(-35.0, 35.0), initialize=0.0)
    m.y3 = pyo.Var(bounds=(-35.0, 35.0), initialize=0.0)

    # Mutable scenario parameters
    m.c1 = pyo.Param(mutable=True, initialize=-30.0)
    m.c2 = pyo.Param(mutable=True, initialize=20.0)
    m.c3 = pyo.Param(mutable=True, initialize=10.0)

    m.xi1 = pyo.Param(mutable=True, initialize=0.00)
    m.xi2 = pyo.Param(mutable=True, initialize=0.00)
    m.xi3 = pyo.Param(mutable=True, initialize=0.00)

    m.shift1 = pyo.Param(mutable=True, initialize=2.0)
    m.shift2 = pyo.Param(mutable=True, initialize=-1.0)
    m.shift3 = pyo.Param(mutable=True, initialize=1.5)

    m.cap_rhs = pyo.Param(mutable=True, initialize=22.0)

    # Objective weights (deterministic across scenarios)
    alpha = 1.4   # attraction of x toward scenario target c
    gamma = 2.1   # tracking term around c + xi
    eta   = 0.55  # recourse coupling / regularization
    beta  = 1.65  # negative pairwise y coupling => nonconvex quadratic
    rho   = 0.12  # mild scenario tilt on y

    # Linear constraints
    m.csum_ub = pyo.Constraint(expr=m.y1 + m.y2 + m.y3 <= m.cap_rhs)
    m.csum_lb = pyo.Constraint(expr=-m.cap_rhs <= m.y1 + m.y2 + m.y3)

    x_center = ((m.x1 - m.c1)**2 + (m.x2 - m.c2)**2 + (m.x3 - m.c3)**2)
    track = ((m.x1 + m.y1 - (m.c1 + m.xi1))**2 +
             (m.x2 + m.y2 - (m.c2 + m.xi2))**2 +
             (m.x3 + m.y3 - (m.c3 + m.xi3))**2)
    couple = (m.y1 + m.y2 + m.y3)**2
    nonconvex_pair = (m.y1 * m.y2 + m.y1 * m.y3 + m.y2 * m.y3)
    shift = (m.shift1 * m.y1 + m.shift2 * m.y2 + m.shift3 * m.y3)

    m.obj_expr = alpha * x_center + gamma * track + eta * couple - beta * nonconvex_pair + rho * shift + OBJ_SHIFT
    return m


def all_vars_center3_gurobi(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [m.x1, m.x2, m.x3, m.y1, m.y2, m.y3]


# =============================================================================
# Scenario generator
# =============================================================================

# Three interior target clusters for first-stage optima
_CLUSTER_BASES = [
    (-30.0, 20.0, 10.0),
    (28.0, -18.0, 34.0),
    (12.0, 33.0, -26.0),
]


def _cluster_center_for_scenario(s: int) -> Tuple[float, float, float]:
    # scenario 0 remains exactly unperturbed at cluster 0
    if s == 0:
        return _CLUSTER_BASES[0]
    return _CLUSTER_BASES[s % len(_CLUSTER_BASES)]


def _set_cluster_center(m: pyo.ConcreteModel, center: Tuple[float, float, float]) -> None:
    m.c1.set_value(center[0])
    m.c2.set_value(center[1])
    m.c3.set_value(center[2])


_PARAM_BASE = [
    ("c1",     0.00, "sym", 6.0),
    ("c2",     0.00, "sym", 6.0),
    ("c3",     0.00, "sym", 6.0),
    ("xi1",    0.00, "sym", 8.0),
    ("xi2",    0.00, "sym", 8.0),
    ("xi3",    0.00, "sym", 8.0),
    ("shift1", 2.00, "sym", 3.0),
    ("shift2", -1.00, "sym", 3.0),
    ("shift3", 1.50, "sym", 3.0),
    ("cap_rhs", 22.0, "pos", 0.20),
]


def build_models_center3_gurobi(
    nscen: int,
    nfirst: int = 3,
    nparam: int = 10,
    seed: int = 1234,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    if nfirst != 3:
        raise ValueError("This synthetic benchmark fixes the first-stage dimension at 3 (use --nfirst 3).")

    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    max_mods = nparam

    for s in range(nscen):
        m = create_model_center3_gurobi()
        allv = all_vars_center3_gurobi(m)
        first = allv[:nfirst]

        base_center = _cluster_center_for_scenario(s)
        _set_cluster_center(m, base_center)

        if s > 0:
            for idx in range(min(max_mods, len(_PARAM_BASE))):
                pname, base_val, ptype, width = _PARAM_BASE[idx]
                if pname in ("c1", "c2", "c3"):
                    # perturb around the assigned cluster center, not around zero
                    current_val = float(pyo.value(getattr(m, pname)))
                    new_val = addnoise_sym(current_val, width, rng)
                elif ptype == "sym":
                    new_val = addnoise_sym(base_val, width, rng)
                elif ptype == "pos":
                    new_val = addnoise_pos(base_val, width, rng)
                else:
                    raise ValueError(f"Unknown parameter perturbation type: {ptype}")
                getattr(m, pname).set_value(new_val)

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            vals = " ".join(
                f"{p}={float(pyo.value(getattr(m, p))):.4f}"
                for p, _, _, _ in _PARAM_BASE[:max_mods]
            )
            print(f"[SCEN {s:04d}] cluster={base_center} {vals}")

        model_list.append(m)
        first_vars_list.append(first)

    return model_list, first_vars_list


# =============================================================================
# Runner config
# =============================================================================

MODE_PARAMS = {
    "smoke": {
        "nscen": 5,
        "target_nodes": 100,
        "gap_stop_tol": 1e-5,
        "time_limit": 60*2,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/st_center3_gurobi_multicluster_smoke/plots",
        "output_csv_path": "results/st_center3_gurobi_multicluster_smoke/simplex_result.csv",
        "use_fbbt":        True,
        "use_obbt":        True,
        "obbt_tol":        1e-2,
        "max_obbt_rounds": 3,
        "obbt_solver_name": "gurobi",
        "obbt_solver_opts": {"TimeLimit": 5},
    },
    "full": {
        "nscen": 100,
        "target_nodes": 900,
        "gap_stop_tol": 1e-4,
        "time_limit": 60*60*1,
        "enable_ef_ub": True,
        "ef_time_ub": 60,
        "plot_every": None,
        "plot_output_dir": "results/st_center3_gurobi_multicluster_full/plots",
        "output_csv_path": "results/st_center3_gurobi_multicluster_full/simplex_result.csv",
        "use_fbbt":        True,
        "use_obbt":        False,
        "obbt_tol":        1e-2,
        "max_obbt_rounds": 3,
        "obbt_solver_name": "gurobi",
        "obbt_solver_opts": {"TimeLimit": 5},
    },
}

BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-3, "TimeLimit": 30}
Q_MAX = 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfirst", type=int, default=3)
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()

    nfirst = args.nfirst
    nparam = 10

    if nfirst != 3:
        raise ValueError("This benchmark is hard-coded to nfirst=3.")

    cfg = dict(MODE_PARAMS[args.mode])
    out_csv = Path(cfg["output_csv_path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"] is not None:
        Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("st_center3_gurobi_multicluster (Python) — synthetic interior-multicluster QCQP benchmark")
    print(f"Mode: {args.mode}, nscen={cfg['nscen']}, nfirst={nfirst}, nparam={nparam}, seed={args.seed}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print(f"use_fbbt={cfg['use_fbbt']}, use_obbt={cfg['use_obbt']}, "
          f"obbt_tol={cfg['obbt_tol']}, max_obbt_rounds={cfg['max_obbt_rounds']}, "
          f"obbt_solver={cfg['obbt_solver_name']}, obbt_solver_opts={cfg['obbt_solver_opts']}")
    print("=" * 60)

    t0 = perf_counter()
    model_list, first_vars_list = build_models_center3_gurobi(
        nscen=cfg["nscen"], nfirst=nfirst, nparam=nparam,
        seed=args.seed, print_first_k_rhs=args.print_first_k_rhs)
    S = len(model_list)

    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    ms_bundles   = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]

    axis_labels = tuple(f"x{i+1}" for i in range(nfirst))

    res = run_pid_simplex_3d(
        model_list=model_list,
        first_vars_list=first_vars_list,
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        target_nodes=cfg["target_nodes"],
        min_dist=1e-3,
        gap_stop_tol=cfg["gap_stop_tol"],
        time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"],
        ef_time_ub=cfg["ef_time_ub"],
        plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"],
        output_csv_path=str(out_csv),
        enable_3d_plot=False,
        axis_labels=axis_labels,
        use_fbbt=cfg["use_fbbt"],
        use_obbt=cfg["use_obbt"],
        obbt_tol=cfg["obbt_tol"],
        max_obbt_rounds=cfg["max_obbt_rounds"],
        obbt_solver_name=cfg["obbt_solver_name"],
        obbt_solver_opts=cfg["obbt_solver_opts"],
    )
    t1 = perf_counter()

    LB_hist = res.get("LB_hist", [])
    UB_hist = res.get("UB_hist", [])
    if LB_hist and UB_hist:
        print(f"\n=== Final (sum) ===\nLB_sum = {float(LB_hist[-1]):.12f}\nUB_sum = {float(UB_hist[-1]):.12f}")
        print(f"\n=== Final (per-scenario) ===\nLB = {float(LB_hist[-1])/S:.12f}\nUB = {float(UB_hist[-1])/S:.12f}")
    print(f"{'='*60}\nDone. Wall time: {t1-t0:.2f} sec\nCSV: {out_csv}\n{'='*60}")


if __name__ == "__main__":
    main()
