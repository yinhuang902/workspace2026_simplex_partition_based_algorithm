"""
Julia-style LP relaxation of the extensive form for lower bounding.

Mirrors the logic in Julia SNGO-master:
  - relax.jl        → _build_lp_for_node()  (McCormick envelopes)
  - updaterelax.jl   → _rebuild_lp_for_node() (update bounds per node)
  - bb.jl:707-806    → _solve_lp_with_cuts()  (iterative OA/αBB loop)
  - preprocessex.jl  → _preprocess_ef()        (identify bilinear terms)

Phase 1 scope: bilinear (x*y) and quadratic (x²) terms ONLY.
Unsupported nonlinear forms are detected, logged, and skipped.
"""
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from snoglode.utils.supported import SupportedVars

logger = logging.getLogger(__name__)

# Numerical constants (matching Julia core.jl)
_SMALL_BOUND = 1e-8
_COEFF_BOUND = 1e8
_SIGMA_VIOLATION = 1e-6
_LP_IMPROVE_TOL = 1e-6


# ──────────────────────────── Data Structures ────────────────────────────

@dataclass
class BilinearEntry:
    """One registered bilinear product  w = x * y  (or w = x² if is_squared)."""
    key: Tuple[int, int]      # (x_idx, y_idx), with x_idx <= y_idx
    x_idx: int
    y_idx: int
    aux_idx: int              # position in the auxiliary-variable array
    is_squared: bool


@dataclass
class ConvexGroup:
    """Bilinear terms forming a convex (pd=+1) or concave (pd=-1) quadratic form."""
    bilinear_keys: List[Tuple[int, int]]
    coefficients: List[float]
    pd: int                   # +1 = convex, -1 = concave


@dataclass
class ABBGroup:
    """Indefinite quadratic form needing αBB perturbation."""
    bilinear_keys: List[Tuple[int, int]]
    coefficients: List[float]
    diag_indices: List[int]   # var indices that need α-shift (squared terms)
    alpha_values: List[float] # per-diagonal-var α value


@dataclass
class _ConstraintInfo:
    """Cached decomposition of one constraint."""
    name: str
    linear_terms: Dict[int, float]   # var_idx → coefficient
    bilinear_terms: List[Tuple[Tuple[int, int], float]]   # (bilinear_key, coeff)
    constant: float
    lb: Optional[float]
    ub: Optional[float]
    is_quad: bool


@dataclass
class PreprocessResult:
    """All metadata extracted from the EF.  Computed ONCE, reused every node."""
    n_ef_vars: int
    var_names: List[str]
    var_bounds_original: List[Tuple[Optional[float], Optional[float]]]
    var_domain_str: List[str]

    # lifted_var_id → list of EF var indices
    lifted_var_to_indices: Dict[Any, List[int]]
    # lifted_var_id → SupportedVars enum value (for node.state lookup)
    lifted_var_domain: Dict[Any, Any]

    bilinear_registry: Dict[Tuple[int, int], BilinearEntry]
    constraints: List[_ConstraintInfo]

    # Objective (linearised)
    obj_linear: Dict[int, float]
    obj_bilinear: List[Tuple[Tuple[int, int], float]]
    obj_constant: float
    obj_sense: int              # 1 = minimise

    convex_groups: List[ConvexGroup]
    abb_groups: List[ABBGroup]

    n_unsupported_skipped: int


# ──────────────────────────── Helper Utilities ────────────────────────────

def _safe_lb(v):
    """Return finite lower bound or a large negative number."""
    lb = v if isinstance(v, (int, float)) else (v.lb if hasattr(v, 'lb') else None)
    if lb is None or lb == float('-inf'):
        return -1e10
    return float(lb)

def _safe_ub(v):
    ub = v if isinstance(v, (int, float)) else (v.ub if hasattr(v, 'ub') else None)
    if ub is None or ub == float('inf'):
        return 1e10
    return float(ub)

def _finite(x):
    return x is not None and abs(x) < 1e15

def _bilinear_bounds(xL, xU, yL, yU, is_squared):
    corners = [xL*yL, xL*yU, xU*yL, xU*yU]
    wL = min(corners)
    wU = max(corners)
    if is_squared:
        wL = max(wL, 0.0)
    return wL, wU


# ──────────────────────────── Main Class ────────────────────────────

class EFRelaxationContext:
    """
    Self-contained LP-relaxation engine.

    Lifecycle
    ---------
    1.  __init__(subproblems)  —  build private EF copy + preprocess
    2.  solve_node(...)        —  called at each B&B node
    """

    # ─── initialisation ───

    def __init__(self, subproblems):
        logger.info("EFRelaxationContext: building private EF and preprocessing …")

        # Build a temporary, private EF model
        ef_model, var_map, lifted_map, lifted_dom = \
            self._build_private_ef(subproblems)

        # Preprocess: walk the EF, extract metadata
        self._pp = self._preprocess_ef(ef_model, var_map, lifted_map, lifted_dom)

        # We no longer need the temporary EF model
        del ef_model

        logger.info(
            f"EFRelaxationContext ready: {self._pp.n_ef_vars} vars, "
            f"{len(self._pp.bilinear_registry)} bilinear terms, "
            f"{len(self._pp.constraints)} constraints, "
            f"{self._pp.n_unsupported_skipped} unsupported skipped"
        )

    # ─── private EF builder ───

    @staticmethod
    def _build_private_ef(subproblems):
        """Build an independent EF ConcreteModel from fresh scenario clones."""
        from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

        ef = pyo.ConcreteModel("_lp_relax_ef")
        ef.obj_expr_parts = []

        # ComponentMap: Pyomo Var → integer index
        var_map = pyo.ComponentMap()
        idx_counter = [0]

        # lifted_var_id → [(var_idx, ...)]
        lifted_map: Dict[Any, List[int]] = {}
        lifted_dom: Dict[Any, Any] = {}

        def _register_var(var, scenario_name, lifted_vars_dict):
            """Assign an integer index to a Pyomo Var."""
            if var in var_map:
                return
            idx = idx_counter[0]
            idx_counter[0] += 1
            var_map[var] = idx

        # Populate lifted_dom from the existing subproblems metadata
        for var_type in SupportedVars:
            for var_id in subproblems.root_node_state[var_type]:
                lifted_dom[var_id] = var_type

        for scenario_name in subproblems.all_names:
            model, lifted_vars_dict, prob = subproblems.subproblem_creator(scenario_name)
            model.name = scenario_name

            # Register ALL variables (including indexed)
            for var in model.component_data_objects(pyo.Var, active=True):
                _register_var(var, scenario_name, lifted_vars_dict)

            # Track which indices are lifted (first-stage)
            for var_id, lvar in lifted_vars_dict.items():
                if lvar not in var_map:
                    _register_var(lvar, scenario_name, lifted_vars_dict)
                vidx = var_map[lvar]
                if var_id not in lifted_map:
                    lifted_map[var_id] = []
                lifted_map[var_id].append(vidx)

            # Collect objective part
            scen_obj = get_active_objective(model)
            ef.obj_expr_parts.append((prob, scen_obj.expr))

            # Add model as a block on the EF
            ef.add_component(scenario_name, model)

        # Build nonant constraints  (linear: ref_var == other_var)
        ef.nonant_cons = pyo.ConstraintList()
        for var_id, idx_list in lifted_map.items():
            if len(idx_list) > 1:
                # Find actual Pyomo Var objects for each index
                vars_for_id = []
                for var_obj, vidx in var_map.items():
                    if vidx in idx_list:
                        vars_for_id.append(var_obj)
                ref_var = vars_for_id[0]
                for other_var in vars_for_id[1:]:
                    ef.nonant_cons.add(ref_var == other_var)

        return ef, var_map, lifted_map, lifted_dom

    # ─── preprocessing ───

    def _preprocess_ef(self, ef_model, var_map, lifted_map, lifted_dom):
        """Walk EF constraints + objective.  Extract PreprocessResult."""
        n_vars = len(var_map)

        # Build ordered var info arrays
        var_names = [""] * n_vars
        var_bounds = [(None, None)] * n_vars
        var_domains = ["Reals"] * n_vars
        for var_obj, idx in var_map.items():
            var_names[idx] = var_obj.name
            var_bounds[idx] = (var_obj.lb, var_obj.ub)
            var_domains[idx] = str(var_obj.domain) if var_obj.domain else "Reals"

        bilinear_registry: Dict[Tuple[int, int], BilinearEntry] = {}
        aux_counter = [0]
        constraints: List[_ConstraintInfo] = []
        convex_groups: List[ConvexGroup] = []
        abb_groups: List[ABBGroup] = []
        n_unsupported = 0

        def _get_or_create_bilinear(v1_idx, v2_idx):
            key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            if key not in bilinear_registry:
                bilinear_registry[key] = BilinearEntry(
                    key=key, x_idx=key[0], y_idx=key[1],
                    aux_idx=aux_counter[0],
                    is_squared=(key[0] == key[1])
                )
                aux_counter[0] += 1
            return key

        def _decompose_expr(expr):
            """Decompose expression → (linear, bilinear, constant, unsupported)."""
            repn = generate_standard_repn(expr, quadratic=True)
            linear = {}
            for var_obj, coef in zip(repn.linear_vars, repn.linear_coefs):
                if var_obj not in var_map:
                    return None, None, None, True
                idx = var_map[var_obj]
                linear[idx] = linear.get(idx, 0.0) + float(coef)

            bilinear = []
            if repn.quadratic_vars:
                for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
                    if v1 not in var_map or v2 not in var_map:
                        return None, None, None, True
                    key = _get_or_create_bilinear(var_map[v1], var_map[v2])
                    bilinear.append((key, float(coef)))

            unsupported = (repn.nonlinear_expr is not None)
            return linear, bilinear, float(repn.constant), unsupported

        # ── Walk constraints ──
        for con in ef_model.component_data_objects(pyo.Constraint, active=True):
            if con.body is None:
                continue
            try:
                lb_val = pyo.value(con.lower) if con.lower is not None else None
                ub_val = pyo.value(con.upper) if con.upper is not None else None
            except Exception:
                lb_val, ub_val = None, None

            linear, bilinear, constant, unsupported = _decompose_expr(con.body)

            if unsupported or linear is None:
                n_unsupported += 1
                logger.debug(f"Skipping unsupported constraint: {con.name}")
                continue

            is_quad = len(bilinear) > 0
            constraints.append(_ConstraintInfo(
                name=con.name, linear_terms=linear,
                bilinear_terms=bilinear, constant=constant,
                lb=lb_val, ub=ub_val, is_quad=is_quad
            ))

            # Convexity classification for quad constraints
            if is_quad:
                self._classify_quadratic(
                    bilinear, var_map, convex_groups, abb_groups, con.name
                )

        # ── Walk objective ──
        combined_obj_expr = sum(p * e for p, e in ef_model.obj_expr_parts)
        obj_linear, obj_bilinear, obj_constant, obj_unsup = \
            _decompose_expr(combined_obj_expr)
        if obj_unsup:
            logger.warning("Objective has unsupported NL terms — LP relax objective may be weak")
            if obj_linear is None:
                obj_linear, obj_bilinear, obj_constant = {}, [], 0.0

        return PreprocessResult(
            n_ef_vars=n_vars,
            var_names=var_names,
            var_bounds_original=var_bounds,
            var_domain_str=var_domains,
            lifted_var_to_indices=lifted_map,
            lifted_var_domain=lifted_dom,
            bilinear_registry=bilinear_registry,
            constraints=constraints,
            obj_linear=obj_linear or {},
            obj_bilinear=obj_bilinear or [],
            obj_constant=obj_constant or 0.0,
            obj_sense=1,
            convex_groups=convex_groups,
            abb_groups=abb_groups,
            n_unsupported_skipped=n_unsupported,
        )

    @staticmethod
    def _classify_quadratic(bilinear_terms, var_map, convex_groups, abb_groups, con_name):
        """Classify a set of bilinear terms as convex / concave / indefinite."""
        if not bilinear_terms:
            return

        # Collect unique variable indices involved
        all_indices = set()
        for (i, j), _ in bilinear_terms:
            all_indices.add(i)
            all_indices.add(j)
        idx_list = sorted(all_indices)
        local_map = {g: k for k, g in enumerate(idx_list)}
        n = len(idx_list)

        # Build Hessian H  (H[i,j] = ∂²f/∂x_i∂x_j)
        H = np.zeros((n, n))
        for (gi, gj), coef in bilinear_terms:
            li, lj = local_map[gi], local_map[gj]
            if li == lj:
                H[li, li] += 2.0 * coef   # d²(c·x²)/dx² = 2c
            else:
                H[li, lj] += coef          # d²(c·x·y)/(dxdy) = c
                H[lj, li] += coef

        try:
            eigenvalues = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            return

        keys = [k for k, _ in bilinear_terms]
        coefs = [c for _, c in bilinear_terms]

        if np.all(eigenvalues >= -1e-10):
            convex_groups.append(ConvexGroup(
                bilinear_keys=keys, coefficients=coefs, pd=1
            ))
        elif np.all(eigenvalues <= 1e-10):
            convex_groups.append(ConvexGroup(
                bilinear_keys=keys, coefficients=coefs, pd=-1
            ))
        else:
            # Indefinite → αBB
            lambda_min = float(np.min(eigenvalues))
            alpha = max(0.0, -lambda_min / 2.0)
            # α applied to all diagonal (squared) variables
            diag_indices = [g for g in idx_list]
            abb_groups.append(ABBGroup(
                bilinear_keys=keys, coefficients=coefs,
                diag_indices=diag_indices,
                alpha_values=[alpha] * len(diag_indices)
            ))

    # ─── node-level solve ───

    def solve_node(self, node_state, current_UB,
                   mingap_abs=1e-3, mingap_rel=1e-2,
                   max_oa_iters=20):
        """
        Build LP relaxation for this node and solve with OA/αBB loop.

        Returns dict with: feasible, lp_relax_lb, n_oa_rounds, n_oa_cuts,
                           n_abb_cuts, lp_status, lp_solution
        """
        R, var_r, aux_r = self._build_lp_for_node(node_state, current_UB)
        return self._solve_lp_with_cuts(
            R, var_r, aux_r, max_oa_iters, mingap_abs, mingap_rel, current_UB
        )

    # ─── LP model builder  (Julia: relax + updaterelax) ───

    def _build_lp_for_node(self, node_state, current_UB):
        pp = self._pp
        R = pyo.ConcreteModel("lp_relax")

        # ── Variables ──
        R.var_set = pyo.RangeSet(0, pp.n_ef_vars - 1)
        R.x = pyo.Var(R.var_set, within=pyo.Reals)

        # Set bounds — first-stage from node_state, second-stage from original
        first_stage_indices = set()
        for var_id, idx_list in pp.lifted_var_to_indices.items():
            for idx in idx_list:
                first_stage_indices.add(idx)

        for idx in range(pp.n_ef_vars):
            orig_lb, orig_ub = pp.var_bounds_original[idx]
            R.x[idx].setlb(orig_lb)
            R.x[idx].setub(orig_ub)

        # Override first-stage bounds from node.state
        for var_id, idx_list in pp.lifted_var_to_indices.items():
            if var_id in pp.lifted_var_domain:
                vtype = pp.lifted_var_domain[var_id]
                if vtype in node_state and var_id in node_state[vtype]:
                    lv = node_state[vtype][var_id]
                    for idx in idx_list:
                        R.x[idx].setlb(lv.lb)
                        R.x[idx].setub(lv.ub)

        # ── Auxiliary bilinear variables ──
        n_aux = len(pp.bilinear_registry)
        if n_aux > 0:
            R.aux_set = pyo.RangeSet(0, n_aux - 1)
            R.w = pyo.Var(R.aux_set, within=pyo.Reals)
        else:
            R.w = {}

        # Set aux var bounds  +  McCormick envelopes
        R.mccormick = pyo.ConstraintList()
        for key, entry in pp.bilinear_registry.items():
            xL = _safe_lb(R.x[entry.x_idx])
            xU = _safe_ub(R.x[entry.x_idx])
            yL = _safe_lb(R.x[entry.y_idx])
            yU = _safe_ub(R.x[entry.y_idx])
            wL, wU = _bilinear_bounds(xL, xU, yL, yU, entry.is_squared)
            w = R.w[entry.aux_idx]
            w.setlb(wL)
            w.setub(wU)
            x_var = R.x[entry.x_idx]
            y_var = R.x[entry.y_idx]

            if entry.is_squared:
                # Julia relax.jl:94-106
                if _finite(xU) and _finite(yU):
                    R.mccormick.add(w >= (yU + xU) * x_var - xU * yU)
                if _finite(xL) and _finite(yL):
                    R.mccormick.add(w >= (yL + xL) * x_var - xL * yL)
                if _finite(xL) and _finite(yU):
                    R.mccormick.add(w <= (yU + xL) * x_var - xL * yU)
            else:
                # Julia relax.jl:108-119  (4 McCormick)
                if _finite(xU) and _finite(yU):
                    R.mccormick.add(w >= yU * x_var + xU * y_var - xU * yU)
                if _finite(xL) and _finite(yL):
                    R.mccormick.add(w >= yL * x_var + xL * y_var - xL * yL)
                if _finite(xL) and _finite(yU):
                    R.mccormick.add(w <= yU * x_var + xL * y_var - xL * yU)
                if _finite(xU) and _finite(yL):
                    R.mccormick.add(w <= yL * x_var + xU * y_var - xU * yL)

        # ── Constraints  (linear + linearised quadratic) ──
        R.cons = pyo.ConstraintList()
        for ci in pp.constraints:
            expr = sum(c * R.x[idx] for idx, c in ci.linear_terms.items())
            for bkey, coef in ci.bilinear_terms:
                aux_idx = pp.bilinear_registry[bkey].aux_idx
                expr += coef * R.w[aux_idx]
            expr += ci.constant
            if ci.lb is not None and ci.ub is not None:
                R.cons.add(pyo.inequality(ci.lb, expr, ci.ub))
            elif ci.ub is not None:
                R.cons.add(expr <= ci.ub)
            elif ci.lb is not None:
                R.cons.add(expr >= ci.lb)

        # ── Objective (linearised) ──
        obj_expr = sum(c * R.x[idx] for idx, c in pp.obj_linear.items())
        for bkey, coef in pp.obj_bilinear:
            aux_idx = pp.bilinear_registry[bkey].aux_idx
            obj_expr += coef * R.w[aux_idx]
        obj_expr += pp.obj_constant
        R.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        # ── UB cap   (Julia relax.jl:42) ──
        if _finite(current_UB):
            R.ub_cap = pyo.Constraint(expr=obj_expr <= current_UB)

        return R, R.x, R.w

    # ─── LP solve + iterative OA / αBB   (Julia bb.jl:707-806) ───

    def _solve_lp_with_cuts(self, R, var_r, aux_r,
                            max_oa_iters, mingap_abs, mingap_rel, UB):
        pp = self._pp
        result = dict(feasible=False, lp_relax_lb=float("-inf"),
                      n_oa_rounds=0, n_oa_cuts=0, n_abb_cuts=0,
                      lp_status="not_solved", lp_solution=None)

        # Gurobi  Method=1 (dual simplex)  (Julia bb.jl:710)
        opt = pyo.SolverFactory("gurobi")
        opt.options["Method"] = 1
        opt.options["Threads"] = 1
        opt.options["LogToConsole"] = 0
        opt.options["OutputFlag"] = 0

        try:
            sol = opt.solve(R, tee=False)
        except Exception as e:
            logger.warning(f"LP relaxation solve failed: {e}")
            result["lp_status"] = "error"
            return result

        tc = sol.solver.termination_condition
        if tc == pyo.TerminationCondition.infeasible:
            result["lp_status"] = "infeasible"
            result["lp_relax_lb"] = float("inf")
            return result
        elif tc != pyo.TerminationCondition.optimal:
            result["lp_status"] = str(tc)
            return result

        relaxed_LB = pyo.value(R.obj)
        result["feasible"] = True
        result["lp_relax_lb"] = relaxed_LB
        result["lp_status"] = "optimal"

        # Check gap
        if _finite(UB) and (UB - relaxed_LB <= mingap_abs or
                            (UB - relaxed_LB <= mingap_rel * min(abs(UB), abs(relaxed_LB)))):
            result["lp_solution"] = self._extract_solution(var_r, aux_r)
            return result

        # ── Iterative OA / αBB loop  (Julia bb.jl:747-798) ──
        no_change_count = 0
        total_oa = 0
        total_abb = 0
        for oa_round in range(max_oa_iters):
            n_oa = self._add_oa_cuts(R, var_r, aux_r)
            n_abb = self._add_abb_cuts(R, var_r, aux_r)
            total_oa += n_oa
            total_abb += n_abb

            if n_oa + n_abb == 0:
                break

            try:
                sol = opt.solve(R, tee=False)
            except Exception:
                break

            tc = sol.solver.termination_condition
            if tc == pyo.TerminationCondition.infeasible:
                result["lp_status"] = "infeasible"
                result["lp_relax_lb"] = float("inf")
                result["feasible"] = False  # LP became infeasible with cuts
                break
            elif tc != pyo.TerminationCondition.optimal:
                break

            new_LB = pyo.value(R.obj)
            improvement = new_LB - relaxed_LB
            relaxed_LB = new_LB
            result["lp_relax_lb"] = relaxed_LB

            if _finite(UB) and (UB - relaxed_LB <= mingap_abs or
                                (UB - relaxed_LB <= mingap_rel * min(abs(UB), abs(relaxed_LB)))):
                break

            if improvement < _LP_IMPROVE_TOL:
                no_change_count += 1
            else:
                no_change_count = 0

            if no_change_count >= 10:
                break

            result["n_oa_rounds"] = oa_round + 1

        result["n_oa_cuts"] = total_oa
        result["n_abb_cuts"] = total_abb
        result["lp_solution"] = self._extract_solution(var_r, aux_r)
        return result

    # ─── OA cuts   (Julia updaterelax.jl:370-561, scoped to bilinear/quadratic) ───

    def _add_oa_cuts(self, R, var_r, aux_r):
        pp = self._pp
        n_cuts = 0

        if not hasattr(R, 'oa_cuts'):
            R.oa_cuts = pyo.ConstraintList()

        # (A) Squared terms: tangent at current LP solution point
        for key, entry in pp.bilinear_registry.items():
            if not entry.is_squared:
                continue
            try:
                xv = pyo.value(var_r[entry.x_idx])
                wv = pyo.value(aux_r[entry.aux_idx])
            except Exception:
                continue
            if wv < xv * xv - _SIGMA_VIOLATION:
                # w >= 2*xv*x - xv²   (tangent line of x² at xv)
                R.oa_cuts.add(
                    aux_r[entry.aux_idx] >= 2.0 * xv * var_r[entry.x_idx] - xv * xv
                )
                n_cuts += 1

        # (B) Convex groups (pd == +1): supporting hyperplane
        for cg in pp.convex_groups:
            if cg.pd != 1 or len(cg.bilinear_keys) <= 1:
                continue
            constant = 0.0
            q_val = 0.0
            terms = []
            for bkey, coef in zip(cg.bilinear_keys, cg.coefficients):
                entry = pp.bilinear_registry[bkey]
                try:
                    xv = pyo.value(var_r[entry.x_idx])
                    yv = pyo.value(var_r[entry.y_idx])
                    wv = pyo.value(aux_r[entry.aux_idx])
                except Exception:
                    continue
                terms.append((entry, coef, xv, yv))
                constant += coef * xv * yv
                q_val += coef * wv

            if q_val <= constant - _SIGMA_VIOLATION:
                cut_expr = 0.0
                for entry, coef, xv, yv in terms:
                    cut_expr += coef * aux_r[entry.aux_idx]
                    cut_expr -= coef * yv * var_r[entry.x_idx]
                    cut_expr -= coef * xv * var_r[entry.y_idx]
                R.oa_cuts.add(cut_expr >= -constant)
                n_cuts += 1

        # (C) Concave groups (pd == -1): overestimator cut
        for cg in pp.convex_groups:
            if cg.pd != -1 or len(cg.bilinear_keys) <= 1:
                continue
            constant = 0.0
            q_val = 0.0
            terms = []
            for bkey, coef in zip(cg.bilinear_keys, cg.coefficients):
                entry = pp.bilinear_registry[bkey]
                try:
                    xv = pyo.value(var_r[entry.x_idx])
                    yv = pyo.value(var_r[entry.y_idx])
                    wv = pyo.value(aux_r[entry.aux_idx])
                except Exception:
                    continue
                terms.append((entry, coef, xv, yv))
                constant += coef * xv * yv
                q_val += coef * wv

            if q_val >= constant + _SIGMA_VIOLATION:
                cut_expr = 0.0
                for entry, coef, xv, yv in terms:
                    cut_expr += coef * aux_r[entry.aux_idx]
                    cut_expr -= coef * yv * var_r[entry.x_idx]
                    cut_expr -= coef * xv * var_r[entry.y_idx]
                R.oa_cuts.add(cut_expr <= -constant)
                n_cuts += 1

        return n_cuts

    # ─── αBB cuts   (Julia updaterelax.jl:288-346, scoped to bilinear/quadratic) ───

    def _add_abb_cuts(self, R, var_r, aux_r):
        pp = self._pp
        n_cuts = 0

        if not hasattr(R, 'abb_cuts'):
            R.abb_cuts = pyo.ConstraintList()

        for ag in pp.abb_groups:
            constant = 0.0
            cut_expr = 0.0

            # Bilinear terms
            for bkey, coef in zip(ag.bilinear_keys, ag.coefficients):
                entry = pp.bilinear_registry[bkey]
                try:
                    xv = pyo.value(var_r[entry.x_idx])
                    yv = pyo.value(var_r[entry.y_idx])
                except Exception:
                    continue
                cut_expr += coef * aux_r[entry.aux_idx]
                cut_expr -= coef * yv * var_r[entry.x_idx]
                cut_expr -= coef * xv * var_r[entry.y_idx]
                constant += coef * xv * yv

            # αBB diagonal shifts
            for diag_idx, alpha in zip(ag.diag_indices, ag.alpha_values):
                if alpha <= 0:
                    continue
                sq_key = (diag_idx, diag_idx)
                if sq_key not in pp.bilinear_registry:
                    continue
                sq_entry = pp.bilinear_registry[sq_key]
                try:
                    xv = pyo.value(var_r[diag_idx])
                except Exception:
                    continue
                cut_expr += alpha * aux_r[sq_entry.aux_idx]
                cut_expr -= 2.0 * alpha * xv * var_r[diag_idx]
                constant += alpha * xv * xv

            R.abb_cuts.add(cut_expr >= -constant)
            n_cuts += 1

        return n_cuts

    # ─── utility ───

    @staticmethod
    def _extract_solution(var_r, aux_r):
        """Extract current LP solution values."""
        sol = {}
        try:
            for idx in var_r:
                sol[idx] = pyo.value(var_r[idx], exception=False)
        except Exception:
            pass
        return sol
