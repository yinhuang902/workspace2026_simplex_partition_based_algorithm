
# simplex_specialstart.py
import csv, json, math, os, sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

# Ensure this file's directory is on sys.path so sibling modules are found
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import _safe_linalg
import numpy as np
import pyomo.environ as pyo   
from scipy.spatial import Delaunay
from time import perf_counter
from pyomo.opt import SolverStatus, TerminationCondition
from exact_opt import compute_exact_optima  # NEW: for validation points
from simplex_geometry import simplex_volume, barycentric_coordinates, snap_to_feature, vol_tolerance

from utils import SimplexTracker
from utils import (
    MIN_DIST, ACTIVE_TOL, MS_AGG, MS_CACHE_ENABLE, GAP_STOP_TOL,
    SMALL_VOL_TOL_ABS, ASPECT_BAD_TOL, EDGE_EPS,
    corners_from_var_bounds, evaluate_Q_at, tet_quality, min_dist_to_nodes,
    compute_edge_aspect,
    _print_candidates_table, plot_iteration_plotly, LAST_DEBUG,
    evaluate_true_expected_obj, append_ef_info, collect_ms_cs_issues,
)
from bundles import SurrogateLBBundle
from iter_logger import IterationLogger
from ef_upper_bounder import SimplexEFUb


# we have 2 setting of tol
# first one is tend to choose c_s point
# Tolerances for candidate-point feature snapping (ms/c_s selection)
# TOL_MS_C:    Switch to c_s if dist(ms, c_s) < TOL_MS_C.
#              1e3 is intentionally loose — effectively always allow c_s.
#              (Previous stricter value was 1e-3; override kept for current tuning.)
TOL_MS_C = 1e3
# TOL_C_VERTS: Only use c_s if min-dist(c_s, vertices) > TOL_C_VERTS.
#              1e-6 is intentionally tight — reject only if c_s is essentially on a vertex.
#              (Previous looser value was 1e2; override kept for current tuning.)
TOL_C_VERTS = 1e-6

# second one is tend to choose ms point

TOL_MS_C = 1e-3
TOL_C_VERTS = 1e2


# EXPERIMENTAL: set to True to also solve EF with Gurobi alongside ipopt (remove later)
ENABLE_EF_GUROBI = False

# DEBUG EXPERIMENT: set to True to clamp dual bounds to primal values.
# When True, ms_val = min(dual, primal) and c_val = min(dual, primal)
# for each per-scenario solve.  Purpose: test whether inflated dual
# bounds are causing LB > UB.  Default False — keep disabled in production.
# FIX-C: Formal safeguard — always clamp MS/CS dual bounds for non-optimal
# (time-limited / iteration-limited) solves.  When a MIP subproblem terminates
# early, the primal objective is an UPPER bound on the true optimum (for a
# minimization subproblem), and the dual bound is a LOWER bound.  Using the raw
# primal obj as ms_val can OVERESTIMATE the LB contribution (min_fverts + ms_total),
# leading to LB > UB violations.  The safe policy is:
#   used_bound = min(dual_bound, primal_obj)
# which is always <= true optimal, i.e. a valid underestimator.
# This was previously behind a DEBUG flag; it is now always enabled.
SAFEGUARD_CLAMP_NONOPT_BOUNDS = True


def _format_cs_failure_status(c_per_scene):
    """Return 'k/S' where k = number of missing/non-finite c values."""
    if c_per_scene is None:
        return "0/0"
    S = len(c_per_scene)
    k = sum(1 for v in c_per_scene
            if v is None or not math.isfinite(v))
    return f"{k}/{S}"


def _effective_c_scene(rec):
    """Merge native c_per_scene with c_fallback_per_scene for LB use.

    For each scenario: use native if finite, else fallback if finite,
    else keep original (None/-inf/etc.).
    """
    native = rec.get("c_per_scene", [])
    fallback = rec.get("c_fallback_per_scene")
    if fallback is None:
        return list(native)
    result = []
    for i in range(len(native)):
        v = native[i]
        if v is not None and math.isfinite(v):
            result.append(v)
        elif i < len(fallback) and fallback[i] is not None and math.isfinite(fallback[i]):
            result.append(fallback[i])
        else:
            result.append(v)  # keep original (None/-inf/etc.)
    return result


def _fmt_point(pt, prec=4):
    """Format a point of any dimension as '(x, y, ...)' string."""
    return "(" + ", ".join(f"{float(v):.{prec}f}" for v in pt) + ")"


def _find_any_simplex_containing_point(point, per_tet, tol_cascade=None):
    """Check whether *point* is inside any simplex in per_tet records.

    Uses barycentric coordinates from simplex_geometry.barycentric_coordinates.
    Tries a cascade of tolerances (tight first, then looser).

    Parameters
    ----------
    point : array-like, shape (d,)
    per_tet : list of dicts, each with "simplex_index" and "verts"
    tol_cascade : list of float, default [1e-8, 1e-6, 1e-4]

    Returns
    -------
    dict with keys:
        found : bool
        simplex_index : int or -1
        lambdas : np.ndarray or None
        tol_used : float or None
        n_checked : int
    """
    from simplex_geometry import barycentric_coordinates as _bary
    if tol_cascade is None:
        tol_cascade = [1e-8, 1e-6, 1e-4]

    pt = np.asarray(point, dtype=float)
    n_checked = len(per_tet)

    for tol in tol_cascade:
        for rec in per_tet:
            verts = np.asarray(rec["verts"], dtype=float)
            try:
                lam = _bary(pt, verts)
            except Exception:
                continue
            if np.all(lam >= -tol) and np.all(lam <= 1.0 + tol):
                return {
                    "found": True,
                    "simplex_index": rec["simplex_index"],
                    "lambdas": lam,
                    "tol_used": tol,
                    "n_checked": n_checked,
                }
    return {
        "found": False,
        "simplex_index": -1,
        "lambdas": None,
        "tol_used": None,
        "n_checked": n_checked,
    }

class SimplexMesh:
    """
    Maintain a tetrahedral mesh incrementally:
    - Initialize once from Delaunay(nodes).
    - Afterwards, update by (optionally) edge/face subdivision or full
      star-subdivision when new nodes are added.
    """

    def __init__(self, nodes, dim=None):
        pts = np.asarray(nodes, float)
        if dim is None:
            dim = pts.shape[1] if pts.ndim == 2 else 3
        self._dim = dim
        self.update_from_delaunay(nodes)
        self.last_split_kind = None   # record subdivision type

    def update_from_delaunay(self, nodes):
        pts = np.asarray(nodes, float)
        n_verts_required = self._dim + 1  # d+1
        if len(pts) < n_verts_required:
            self.tets = []
            return
        tri = Delaunay(pts)
        # each simplex is stored as a tuple of GLOBAL vertex indices
        self.tets = [tuple(map(int, simp)) for simp in tri.simplices]

    def iter_simplices(self):
        """
        Yield (local_index, vert_idx_list) for all tetrahedra.
        local_index is just the position in self.tets for this iteration.
        vert_idx_list contains GLOBAL node indices.
        """
        for k, idxs in enumerate(self.tets):
            yield k, list(idxs)

    # ---------- Original star subdivision: 4 sub-simplices ----------
    def subdivide(self, simplex_index: int, new_node_index: int):
        """
        Replace the simplex at simplex_index by (d+1) new simplices that
        all contain new_node_index (star subdivision).

        Assumes:
            - new_node_index is a valid index into the global nodes list.
            - The new point lies (approximately) inside the old simplex.
        """
        old = list(self.tets[simplex_index])
        d = self._dim
        n_verts = d + 1
        if len(old) != n_verts:
            raise ValueError(f"Expected a {n_verts}-vertex simplex, got {old}")

        # Star subdivision: replace each vertex in turn with the new point
        new_tets = []
        for k in range(n_verts):
            child = list(old)
            child[k] = new_node_index
            new_tets.append(tuple(child))

        # Remove old simplex and append children
        self.tets.pop(simplex_index)
        self.tets.extend(new_tets)
        self.last_split_kind = "interior"
        return n_verts  # number of children

    # ---------- New: point on edge => 2 sub-simplices ----------
    def subdivide_edge(self, simplex_index: int, new_node_index: int, edge_verts):
        """
        Subdivide a d-simplex when the new point lies on an edge.

        For a d-simplex with (d+1) vertices, an edge connects two of those
        vertices (a, b).  This method produces exactly **2** child simplices:

        - child 1: replace vertex *a* with new_node_index (keeps b)
        - child 2: replace vertex *b* with new_node_index (keeps a)

        Each child has (d+1) vertices. Works for any dimension d >= 2.

        Parameters
        ----------
        simplex_index : int
            Index into ``self.tets``.
        new_node_index : int
            Global index of the new node (already appended to the node list).
        edge_verts : tuple[int, int]
            Global indices of the two vertices that define the edge.

        Returns
        -------
        int
            Number of child simplices created (always 2).
        """
        old = list(self.tets.pop(simplex_index))

        a, b = edge_verts
        if a not in old or b not in old:
            raise ValueError(f"edge_verts {edge_verts} not subset of simplex {old}")

        n_others = len(old) - 2   # should be d - 1
        if n_others != self._dim - 1:
            raise ValueError(
                f"Expected {self._dim - 1} non-edge vertices, got {n_others}")

        # Child 1: replace edge endpoint a → new_node
        t1 = list(old)
        t1[old.index(a)] = new_node_index
        # Child 2: replace edge endpoint b → new_node
        t2 = list(old)
        t2[old.index(b)] = new_node_index

        self.tets.extend([tuple(t1), tuple(t2)])
        self.last_split_kind = "edge"
        return 2

    # ---------- New: point on face => 3 sub-simplices ----------
    def subdivide_face(self, simplex_index: int, new_node_index: int, face_verts):
        """
        Subdivide a d-simplex when the new point lies on a facet (codim-1 face).

        For a d-simplex with (d+1) vertices, a facet is a (d-1)-simplex
        defined by **d** of the (d+1) vertices.  This method produces exactly
        ``len(face_verts)`` child simplices:

        - For each vertex **v** in ``face_verts``, create a child by
          replacing **v** with ``new_node_index``.

        Each child has (d+1) vertices. Works for any dimension d >= 2.

        In 3D (tetrahedra), ``face_verts`` has 3 entries and we get 3 children
        — matching the original behaviour.

        Parameters
        ----------
        simplex_index : int
            Index into ``self.tets``.
        new_node_index : int
            Global index of the new node.
        face_verts : Sequence[int]
            Global indices of the vertices that define the facet.
            Length should be ``d`` (i.e., ``self._dim``) for a true facet.

        Returns
        -------
        int
            Number of child simplices created (``len(face_verts)``).
        """
        old = list(self.tets.pop(simplex_index))
        face_verts = list(face_verts)

        for v in face_verts:
            if v not in old:
                raise ValueError(f"face_verts {face_verts} not subset of simplex {old}")

        opp = [v for v in old if v not in face_verts]
        expected_opp = len(old) - len(face_verts)
        if len(opp) != expected_opp:
            raise ValueError(
                f"Expected {expected_opp} opposite vertex(es), got {len(opp)}")

        # Create one child per face vertex: replace that vertex with new_node
        new_tets = []
        for fv in face_verts:
            child = list(old)
            child[old.index(fv)] = new_node_index
            new_tets.append(tuple(child))

        self.tets.extend(new_tets)
        self.last_split_kind = "face"
        return len(face_verts)

    # ---------- Mode 2: split ALL simplices by axis-aligned hyperplane ----------
    def split_by_hyperplane(self, nodes, axis, value, tol=1e-10):
        """
        Cut every simplex that straddles the hyperplane x[axis] = value.

        For each straddling simplex, intersection points are computed on
        each edge that crosses the plane.  These new nodes are appended to
        *nodes* (mutated in-place).  The straddling simplex is then replaced
        by sub-simplices on each side of the plane.

        Parameters
        ----------
        nodes : list[tuple]
            Global node list (will be extended with new intersection nodes).
        axis : int
            Coordinate axis (0, 1, or 2 for 3-D).
        value : float
            Cut plane position: x[axis] = value.
        tol : float
            Tolerance for classifying a vertex as "on the plane".

        Returns
        -------
        dict
            ``new_node_indices`` : list[int] – global indices of added nodes.
            ``n_cut``            : int – number of simplices that were cut.
            ``n_children``       : int – total new simplices created.
        """
        pts = np.asarray(nodes, float)
        new_tets = []
        kept_tets = []
        new_node_indices = []

        # Cache for edge → new-node-index so shared edges are not duplicated
        edge_node_cache = {}  # (min_idx, max_idx) → new_node_global_index

        n_cut = 0

        for tet in self.tets:
            coords = np.array([nodes[v] for v in tet], float)  # (d+1, d)
            vals = coords[:, axis]

            below = vals < value - tol
            above = vals > value + tol
            on_plane = ~below & ~above  # within tol of the plane

            n_below = int(below.sum())
            n_above = int(above.sum())

            # If all on one side (or all on the plane), keep as-is
            if n_below == 0 or n_above == 0:
                kept_tets.append(tet)
                continue

            # This simplex straddles the plane — need to split it
            n_cut += 1

            # Find intersection nodes on each crossing edge
            tet_list = list(tet)
            n_verts = len(tet_list)
            crossing_nodes = []  # new node indices on the plane

            for i in range(n_verts):
                for j in range(i + 1, n_verts):
                    vi, vj = tet_list[i], tet_list[j]
                    vi_val, vj_val = vals[i], vals[j]

                    # Edge crosses if one is below and one is above
                    crosses = (below[i] and above[j]) or (above[i] and below[j])
                    if not crosses:
                        continue

                    edge_key = (min(vi, vj), max(vi, vj))
                    if edge_key in edge_node_cache:
                        crossing_nodes.append(edge_node_cache[edge_key])
                        continue

                    # Compute intersection: linear interpolation
                    t = (value - vi_val) / (vj_val - vi_val)
                    new_pt = tuple(
                        float((1 - t) * nodes[vi][d] + t * nodes[vj][d])
                        for d in range(len(nodes[vi]))
                    )
                    # Force exact coordinate on the cutting axis
                    new_pt_list = list(new_pt)
                    new_pt_list[axis] = value
                    new_pt = tuple(new_pt_list)

                    new_idx = len(nodes)
                    nodes.append(new_pt)
                    new_node_indices.append(new_idx)
                    edge_node_cache[edge_key] = new_idx
                    crossing_nodes.append(new_idx)

            # Also include vertices that are exactly on the plane
            on_plane_verts = [tet_list[i] for i in range(n_verts) if on_plane[i]]

            # Partition original vertices into below-set and above-set
            below_verts = [tet_list[i] for i in range(n_verts) if below[i]]
            above_verts = [tet_list[i] for i in range(n_verts) if above[i]]

            # Plane nodes = crossing intersection nodes + vertices on the plane
            plane_nodes = crossing_nodes + on_plane_verts

            # Build child simplices for each side:
            # Side A (below): below_verts + plane_nodes
            # Side B (above): above_verts + plane_nodes
            for side_verts in [below_verts, above_verts]:
                all_verts = side_verts + plane_nodes
                if len(all_verts) < self._dim + 1:
                    continue  # degenerate — should not happen

                if len(all_verts) == self._dim + 1:
                    # Exactly one simplex
                    new_tets.append(tuple(all_verts))
                else:
                    # Multiple points — need local Delaunay to tessellate
                    local_pts = np.array([nodes[v] for v in all_verts], float)
                    try:
                        from scipy.spatial import Delaunay as _Del
                        local_tri = _Del(local_pts)
                        for simp in local_tri.simplices:
                            child = tuple(all_verts[k] for k in simp)
                            new_tets.append(child)
                    except Exception:
                        # Fallback: if Delaunay fails (degenerate), just make
                        # one simplex from the first d+1 points
                        new_tets.append(tuple(all_verts[:self._dim + 1]))

        self.tets = kept_tets + new_tets
        self.last_split_kind = "hyperplane"
        return {
            "new_node_indices": new_node_indices,
            "n_cut": n_cut,
            "n_children": len(new_tets),
        }

    def as_delaunay_like(self):
        """
        Return a light-weight object with a 'simplices' attribute so that
        plotting code expecting scipy.spatial.Delaunay still works.
        """
        class _Dummy:
            pass

        obj = _Dummy()
        n_verts = self._dim + 1
        if self.tets:
            obj.simplices = np.asarray(self.tets, dtype=int)
        else:
            obj.simplices = np.zeros((0, n_verts), dtype=int)
        return obj




# small LP solver
# small LP solver (now backed by a persistent Gurobi solver)
_LB_BUNDLE = None   # global singleton, built once for given S

def solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene, terms_out=None):
    """
    fverts_per_scene: list of length S, each is length-4 list of As_s at 4 vertices
    ms_scene: length-S iterable of ms_s
    c_scene: length-S iterable of c_s (may contain -inf)

    Convention:
      - If some ms_s = +inf, it means ms subproblem failed for that scenario;
        we don't use As+ms condition, only use sum_s c_s as LB for this simplex.

    terms_out: optional dict to populate with LB decomposition terms:
        LB_linear, LB_const, fallback_LB, LB_sur_raw, LB_sur_final, lb_path
    """
    global _LB_BUNDLE

    S = len(ms_scene)

    # ---------- cheap fallback: LB_linear & LB_const ----------
    fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(len(fverts_per_scene[0]))]
    ms_scene_arr = np.asarray(ms_scene, float)
    c_scene_arr  = np.asarray(c_scene,  float)

    finite_c = c_scene_arr[np.isfinite(c_scene_arr)]

    # === Case 1: Exists ms_s = +inf -> treat ms as "unsolvable", use sum c_s as LB ===
    if not np.all(np.isfinite(ms_scene_arr)):
        _result = float(np.sum(finite_c)) if finite_c.size > 0 else float('inf')
        if terms_out is not None:
            terms_out["LB_linear"] = float('inf')   # MS not available
            terms_out["LB_const"] = float(np.sum(finite_c)) if finite_c.size > 0 else float('-inf')
            terms_out["fallback_LB"] = _result
            terms_out["LB_sur_raw"] = None           # surrogate LP not invoked
            terms_out["LB_sur_final"] = _result
            terms_out["lb_path"] = "case1_ms_inf"
        return _result

    # === Case 2: All ms_s finite -> keep original surrogate-LP logic ===
    ms_total   = float(np.sum(ms_scene_arr))
    LB_linear  = float(np.min(fverts_sum) + ms_total)

    if finite_c.size > 0:
        c_total     = float(np.sum(finite_c))
        LB_const    = c_total
        fallback_LB = max(LB_linear, LB_const)
    else:
        # all c_s = -inf, equivalent to only having As+ms part
        LB_const = float('-inf')
        fallback_LB = LB_linear

    # If all c_s are -inf, no need to solve LP, use fallback_LB directly
    if finite_c.size == 0:
        if terms_out is not None:
            terms_out["LB_linear"] = LB_linear
            terms_out["LB_const"] = LB_const
            terms_out["fallback_LB"] = fallback_LB
            terms_out["LB_sur_raw"] = None
            terms_out["LB_sur_final"] = fallback_LB
            terms_out["lb_path"] = "case2_no_finite_c"
        return fallback_LB

    # ---------- use dedicated persistent solver for this LP ----------
    if (_LB_BUNDLE is None) or (_LB_BUNDLE.S != S) or (_LB_BUNDLE._n_verts != len(fverts_per_scene[0])):
        _LB_BUNDLE = SurrogateLBBundle(S, n_verts=len(fverts_per_scene[0]))

    LB_sur = _LB_BUNDLE.compute_lb(
        fverts_per_scene=fverts_per_scene,
        ms_scene=ms_scene_arr,
        c_scene=c_scene_arr,
        fallback_LB=fallback_LB,
    )

    # Capture raw LP result from SurrogateLBBundle if available
    _raw_lp = getattr(_LB_BUNDLE, '_last_lb_terms', None)
    _lb_sur_raw = _raw_lp.get('raw_lp_obj') if _raw_lp else None

    if terms_out is not None:
        terms_out["LB_linear"] = LB_linear
        terms_out["LB_const"] = LB_const
        terms_out["fallback_LB"] = fallback_LB
        terms_out["LB_sur_raw"] = _lb_sur_raw
        terms_out["LB_sur_final"] = float(LB_sur)
        terms_out["lb_path"] = "case2_surrogate_lp"

    return float(LB_sur)


# ------------------------- Single tetra & scene: ms solve (persistent) -------------------------
def ms_on_tetra_for_scene(ms_bundle, tet_vertices, fverts_scene,
                          use_fbbt=False, use_obbt=False,
                          obbt_solver_name="gurobi", obbt_solver_opts=None,
                          obbt_tol=1e-1, max_obbt_rounds=3):
    """
    Solve ms and constant-cut for a single simplex(tetrahedron) in one scenario.

    Args:
        ms_bundle: Persistent model/bundle for the given scenario.
        tet_vertices (list[tuple[float]]): The d+1 vertex coordinates.
        fverts_scene (list[float]): Objective values at those vertices.
        use_fbbt (bool): Run FBBT before solve.
        use_obbt (bool): Run OBBT before solve.
        obbt_solver_name (str): Solver name for OBBT.
        obbt_solver_opts (dict|None): Solver options for OBBT.

    Returns:
        tuple of 5:
            ms_val (float): ms value; +inf if ms solve failed.
            new_pt_ms (tuple | None): Interpolated first-stage point.
            c_val (float): c_s = min_T Q_s(K), -inf if failed.
            c_pt (tuple | None): First-stage point corresponding to c_s.
            solve_state (str): one of "ok", "tightening_infeasible",
                               "ms_failed", "cs_failed", "both_failed".
    """
    ms_bundle.update_tetra(tet_vertices, fverts_scene)

    # ---- FBBT / OBBT bounds tightening (with snapshot/restore) ----
    _snap = None
    if use_fbbt or use_obbt:
        _snap = ms_bundle._snapshot_bounds()
        tighten_result = ms_bundle._tighten_current_simplex_bounds(
            use_fbbt=use_fbbt, use_obbt=use_obbt,
            obbt_solver_name=obbt_solver_name,
            obbt_solver_opts=obbt_solver_opts,
            obbt_tol=obbt_tol,
            max_obbt_rounds=max_obbt_rounds,
        )
        if not tighten_result["feasible"]:
            # Tightening infeasible: skip MS entirely, attempt CS on safe bounds.
            # Policy: MS infeasible => degrade (ms=+inf), do NOT kill simplex.
            ms_bundle._restore_bounds(_snap)
            _snap = None  # already restored — prevent double-restore in finally
            print(f"[ms_on_tetra] Tightening infeasible → skipping MS, "
                  f"attempting CS on restored safe bounds", flush=True)
            try:
                ok_c, c_val, c_pt = ms_bundle.solve_const_cut()
                if not ok_c:
                    c_val = float('-inf')
                    c_pt = None
                # Apply same SAFEGUARD_CLAMP logic to CS
                if (SAFEGUARD_CLAMP_NONOPT_BOUNDS and ok_c
                        and c_val is not None and math.isfinite(c_val)):
                    _cs_meta = getattr(ms_bundle, 'last_cs_meta', None)
                    if _cs_meta is not None:
                        _cs_status = _cs_meta.get('status', 'optimal')
                        _d = _cs_meta.get('dual_bound')
                        _p = _cs_meta.get('primal_obj')
                        if (_d is not None and _p is not None
                                and math.isfinite(_d) and math.isfinite(_p)
                                and _d > _p + 1e-8):
                            c_val = min(c_val, _p)
                        if _cs_status not in ('optimal',):
                            if (_d is not None and _p is not None
                                    and math.isfinite(_d) and math.isfinite(_p)):
                                c_val = min(_d, _p)
                            elif _p is not None and math.isfinite(_p):
                                c_val = min(c_val, _p)
            except Exception as _e_cs:
                print(f"[ms_on_tetra] CS on safe bounds also failed: {_e_cs}",
                      flush=True)
                c_val = float('-inf')
                c_pt = None
            _cs_ok = (c_val is not None and math.isfinite(c_val))
            _state = ("tightening_ms_skip_cs_ok" if _cs_ok
                      else "tightening_ms_skip_cs_failed")
            return float('inf'), None, c_val, c_pt, _state

    try:
        ok_ms = ms_bundle.solve()
        if ok_ms:
            ms_val, lam_star, new_pt_ms = ms_bundle.get_ms_and_point()
        else:
            ms_val = float('inf')
            lam_star = None
            new_pt_ms = None

        # FIX-C: Formal safeguard — for non-optimal MS solves,
        # clamp ms_val to min(ms_val, primal_obj) so the LB contribution is safe.
        if SAFEGUARD_CLAMP_NONOPT_BOUNDS and ok_ms:
            _ms_meta = getattr(ms_bundle, 'last_solve_meta', None)
            if _ms_meta is not None:
                _ms_status = _ms_meta.get('status', 'optimal')
                _d = _ms_meta.get('dual_bound')
                _p = _ms_meta.get('primal_obj')
                if (_d is not None and _p is not None
                        and math.isfinite(_d) and math.isfinite(_p)
                        and _d > _p + 1e-8):
                    ms_val = min(ms_val, _p)
                if _ms_status not in ('optimal',):
                    if (_d is not None and _p is not None
                            and math.isfinite(_d) and math.isfinite(_p)):
                        ms_val = min(_d, _p)
                    elif _p is not None and math.isfinite(_p):
                        pass

        ok_c, c_val, c_pt = ms_bundle.solve_const_cut()
        if not ok_c:
            print("c_s solve wrong")
            c_val = float('-inf')
            c_pt = None

        # FIX-C: Formal safeguard — for non-optimal CS solves, clamp c_val
        if SAFEGUARD_CLAMP_NONOPT_BOUNDS and ok_c and c_val is not None and math.isfinite(c_val):
            _cs_meta = getattr(ms_bundle, 'last_cs_meta', None)
            if _cs_meta is not None:
                _cs_status = _cs_meta.get('status', 'optimal')
                _d = _cs_meta.get('dual_bound')
                _p = _cs_meta.get('primal_obj')
                if (_d is not None and _p is not None
                        and math.isfinite(_d) and math.isfinite(_p)
                        and _d > _p + 1e-8):
                    c_val = min(c_val, _p)
                if _cs_status not in ('optimal',):
                    if (_d is not None and _p is not None
                            and math.isfinite(_d) and math.isfinite(_p)):
                        c_val = min(_d, _p)
                    elif _p is not None and math.isfinite(_p):
                        c_val = min(c_val, _p)

        # Determine solve_state
        _ms_ok = (ms_val is not None and math.isfinite(ms_val))
        _cs_ok = (c_val is not None and math.isfinite(c_val))
        if _ms_ok and _cs_ok:
            solve_state = "ok"
        elif _ms_ok and not _cs_ok:
            solve_state = "cs_failed"
        elif not _ms_ok and _cs_ok:
            solve_state = "ms_failed"
        else:
            solve_state = "both_failed"

        return ms_val, new_pt_ms, c_val, c_pt, solve_state

    finally:
        # ---- Restore bounds after solve if we tightened ----
        if _snap is not None:
            ms_bundle._restore_bounds(_snap)


# ---- CS-only solve for a single simplex & scenario (skips MS) ----
def cs_only_on_tetra_for_scene(ms_bundle, tet_vertices, fverts_scene,
                               use_fbbt=False, use_obbt=False,
                               obbt_solver_name="gurobi", obbt_solver_opts=None,
                               obbt_tol=1e-1, max_obbt_rounds=3):
    """
    Solve ONLY constant-cut for a single simplex in one scenario.
    Skips the expensive ms_bundle.solve() entirely.

    Returns the same 5-tuple as ms_on_tetra_for_scene:
        (ms_val=+inf, new_pt_ms=None, c_val, c_pt, solve_state)
    """
    ms_bundle.update_tetra(tet_vertices, fverts_scene)

    # ---- FBBT / OBBT bounds tightening (with snapshot/restore) ----
    _snap = None
    if use_fbbt or use_obbt:
        _snap = ms_bundle._snapshot_bounds()
        tighten_result = ms_bundle._tighten_current_simplex_bounds(
            use_fbbt=use_fbbt, use_obbt=use_obbt,
            obbt_solver_name=obbt_solver_name,
            obbt_solver_opts=obbt_solver_opts,
            obbt_tol=obbt_tol,
            max_obbt_rounds=max_obbt_rounds,
        )
        if not tighten_result["feasible"]:
            ms_bundle._restore_bounds(_snap)
            return float('inf'), None, float('-inf'), None, "tightening_infeasible"

    try:
        ok_c, c_val, c_pt = ms_bundle.solve_const_cut()
        if not ok_c:
            c_val = float('-inf')
            c_pt = None

        # FIX-C: Formal safeguard — for non-optimal CS-only solves, clamp c_val
        if SAFEGUARD_CLAMP_NONOPT_BOUNDS and ok_c and c_val is not None and math.isfinite(c_val):
            _cs_meta = getattr(ms_bundle, 'last_cs_meta', None)
            if _cs_meta is not None:
                _cs_status = _cs_meta.get('status', 'optimal')
                _d = _cs_meta.get('dual_bound')
                _p = _cs_meta.get('primal_obj')
                if (_d is not None and _p is not None
                        and math.isfinite(_d) and math.isfinite(_p)
                        and _d > _p + 1e-8):
                    c_val = min(c_val, _p)
                if _cs_status not in ('optimal',):
                    if (_d is not None and _p is not None
                            and math.isfinite(_d) and math.isfinite(_p)):
                        c_val = min(_d, _p)
                    elif _p is not None and math.isfinite(_p):
                        c_val = min(c_val, _p)

        _cs_ok = (c_val is not None and math.isfinite(c_val))
        solve_state = "ok" if _cs_ok else "cs_failed"
        return float('inf'), None, c_val, c_pt, solve_state

    finally:
        # ---- Restore bounds after solve if we tightened ----
        if _snap is not None:
            ms_bundle._restore_bounds(_snap)


# ---- Cache status helpers ----
# Cache entries are 5-tuples: (status, ms_val, new_pt_ms, c_val, c_pt)
# status in {"full", "cs_only", "ms_only", "ms_failed", "cs_failed",
#            "both_failed", "tightening_infeasible",
#            "tightening_ms_skip_cs_ok", "tightening_ms_skip_cs_failed"}

def _cache_status_has_cs(status):
    """Return True if this cache entry has usable CS data."""
    return status in ("full", "cs_only", "ms_failed",
                      "tightening_ms_skip_cs_ok")

def _cache_status_has_ms(status):
    """Return True if this cache entry has usable MS data."""
    return status in ("full", "ms_only", "cs_failed")
    # NOTE: "tightening_ms_skip_*" statuses do NOT have usable MS

def _cache_status_is_tightening_infeasible(status):
    """Return True if this cache entry was marked infeasible by FBBT/OBBT.
    Includes both the old CS-side kill status and the new MS-skip statuses."""
    return status in ("tightening_infeasible",
                      "tightening_ms_skip_cs_ok",
                      "tightening_ms_skip_cs_failed")

def _cache_status_is_tightening_ms_skip(status):
    """Return True if tightening failed but MS was skipped AND CS is available.
    This is the harmless case: simplex stays alive with CS-only support."""
    return status == "tightening_ms_skip_cs_ok"

def _cache_status_is_cs_side_kill(status):
    """Return True if this cache entry represents a CS-side kill (simplex death).
    Includes:
      - old 'tightening_infeasible' (used by CS-only paths)
      - 'tightening_ms_skip_cs_failed' (MS skipped, fallback CS also failed)"""
    return status in ("tightening_infeasible",
                      "tightening_ms_skip_cs_failed")

def _upgrade_cache_entry_old_format(entry):
    """Upgrade a legacy 4-tuple cache entry to 5-tuple format."""
    if len(entry) == 5:
        return entry  # already new format
    # Old format: (ms_val, new_pt_ms, c_val, c_pt)
    ms_val, new_pt_ms, c_val, c_pt = entry
    ms_ok = (ms_val is not None and math.isfinite(ms_val))
    cs_ok = (c_val is not None and math.isfinite(c_val))
    if ms_ok and cs_ok:
        status = "full"
    elif ms_ok and not cs_ok:
        status = "ms_only"
    elif not ms_ok and cs_ok:
        status = "ms_failed"
    else:
        status = "both_failed"
    return (status, ms_val, new_pt_ms, c_val, c_pt)


# ---- Simplex containment helper ----
def _find_simplex_containing_point(point, per_tet, tol=1e-8):
    """
    Find the first simplex record in *per_tet* that geometrically contains *point*.

    Uses barycentric coordinates: a point is inside if all bary coords >= -tol
    AND abs(sum(bary) - 1) <= tol.  Skips simplices with non-positive volume.
    Returns the index into per_tet (not the simplex_index field), or -1 if none.
    Deterministic: first match wins.
    """
    from simplex_geometry import barycentric_coordinates
    p = np.asarray(point, dtype=float)
    for idx, rec in enumerate(per_tet):
        # Skip degenerate / missing-volume simplices
        vol = rec.get("volume", None)
        if vol is None or not math.isfinite(vol) or vol <= 0:
            continue
        V = np.array(rec["verts"], dtype=float)
        try:
            bary = barycentric_coordinates(p, V)
            if np.all(bary >= -tol) and abs(float(np.sum(bary)) - 1.0) <= tol:
                return idx
        except Exception:
            continue
    return -1


def _build_active_mask(records, ub_global, active_tol, q_cut,
                       return_diagnostics=False):
    """Build the active-simplex mask for a set of simplex records.

    INVARIANT: after any in-place LB/status update, the LB simplex must be
    re-selected from a freshly rebuilt active set.

    A simplex is active iff ALL of the following hold:
      1. not permanently inactive (``inactive`` flag)
      2. passes the active-gap test  (LB <= ub_global + active_tol)
      3. passes the quality filter   (tet_quality >= q_cut)
      4. passes the bad-shape filter  (not small-vol + high-aspect)

    When *return_diagnostics* is False (default), returns ``dict {sid: bool}``.
    When True, returns ``(mask, diag)`` where *diag* is a dict with:
      - bad_quality_count:  int
      - bad_shape_count:    int
      - shape_checked:      int   (simplices with vol < threshold)
      - bad_shape_details:  list of dicts with per-simplex shape info
    This is a PURE function — no logging, no side-effects on records.
    """
    mask = {
        r["simplex_index"]: (not r.get("inactive", False)
                             and r["LB"] <= ub_global + active_tol)
        for r in records
    }

    bad_quality_count = 0
    # Quality filter
    for r in records:
        sid = r["simplex_index"]
        if not mask.get(sid, False):
            continue
        q = tet_quality(r["verts"])
        if q < q_cut:
            mask[sid] = False
            bad_quality_count += 1

    # Bad-shape filter: small volume + high aspect ratio
    bad_shape_count = 0
    shape_checked = 0
    bad_shape_details = []  # [{sid, vol, max_e, min_e, aspect}, ...]
    for r in records:
        sid = r["simplex_index"]
        if not mask.get(sid, False):
            continue
        vol = r.get("volume")
        if vol is None or vol >= SMALL_VOL_TOL_ABS:
            continue
        shape_checked += 1
        try:
            max_e, min_e, aspect = compute_edge_aspect(r["verts"])
            if aspect >= ASPECT_BAD_TOL:
                mask[sid] = False
                bad_shape_count += 1
                bad_shape_details.append({
                    "sid": sid, "vol": vol,
                    "max_e": max_e, "min_e": min_e, "aspect": aspect,
                })
        except Exception:
            pass

    if not return_diagnostics:
        return mask
    return mask, {
        "bad_quality_count": bad_quality_count,
        "bad_shape_count": bad_shape_count,
        "shape_checked": shape_checked,
        "bad_shape_details": bad_shape_details,
    }


# ---- Iteration-0 box-level CS-only evaluator ----
def evaluate_iter0_box_cs_only(nodes, scen_values, ms_bundles, first_vars_list,
                               ms_cache=None, cache_on=True, tracker=None,
                               tet_mesh: SimplexMesh | None = None,
                               lb_sur_cache=None,
                               dbg_timelimit_path=None,
                               dbg_cs_timing_path=None,
                               dbg_ms_timing_path=None,
                               iter_num=None,
                               _nonopt_buf=None,
                               _debug_buf=None,
                               dbg_debug_path=None,
                               dbg_lb_split_path=None,
                               base_bundles=None):
    """
    Iteration-0 box-level CS-only evaluator.

    Instead of solving per-simplex CS, solves ONE box-level constant-cut per
    scenario over the full initial box.  All simplex records receive the SAME
    shared box-level c_s values, c_s points, and LB.

    Returns
    -------
    tri : Delaunay-like
    per_tet : list[dict]
    box_c_vals : list[float]       per-scenario c_s values
    box_c_pts  : list[tuple|None]  per-scenario c_s points
    box_LB     : float             aggregated box-level LB
    """
    print("[DBG] evaluate_iter0_box_cs_only: start", flush=True)
    pts = np.asarray(nodes, dtype=float)
    d = pts.shape[1] if pts.ndim == 2 else 3
    n_verts = d + 1
    if len(pts) < n_verts:
        return None, [], [], [], float('-inf')

    if tet_mesh is not None:
        simplices = [list(t) for t in tet_mesh.tets]
        tri = tet_mesh.as_delaunay_like()
    else:
        tri = Delaunay(pts)
        simplices = tri.simplices
    print(f"[DBG] evaluate_iter0_box_cs_only: {len(simplices)} simplices", flush=True)

    S = len(ms_bundles)

    # ---- Step 1: Solve ONE box-level CS per scenario ----
    box_c_vals = []
    box_c_pts = []
    n_box_ok = 0
    print(f"[Iter {iter_num}] Box-level CS-only mode: solving 1 CS per scenario "
          f"over full box ({S} scenarios)", flush=True)

    for w in range(S):
        ok_box, c_val_box, c_pt_box = ms_bundles[w].solve_const_cut_box()
        if not ok_box:
            c_val_box = float('-inf')
            c_pt_box = None
        else:
            n_box_ok += 1
        box_c_vals.append(c_val_box)
        box_c_pts.append(c_pt_box)
        _pt_str = _fmt_point(c_pt_box) if c_pt_box is not None else "None"
        print(f"  scen {w}: c_val={c_val_box:.6e}, c_pt={_pt_str}", flush=True)

        # Buffer non-optimal events
        _cs_meta = getattr(ms_bundles[w], 'last_cs_meta', None)
        if _nonopt_buf is not None and _cs_meta is not None:
            _st = _cs_meta.get("status", "")
            if _st != "optimal":
                _nonopt_buf.append(
                    f"  iter={iter_num} box_cs scen={w} type=box_cs "
                    f"status={_st} "
                    f"term={_cs_meta.get('termination_condition','?')} "
                    f"ok={_cs_meta.get('ok','?')} "
                    f"val={_cs_meta.get('dual_bound','?')} "
                    f"time={_cs_meta.get('time_sec',0.0):.4f}s\n"
                )

    # Collect box-level CS metadata for each scenario (shared across all simplices).
    # Using deepcopy so each simplex record gets an independent copy that won't
    # be mutated by later solves.  The metadata includes box_level=True.
    from copy import deepcopy
    box_cs_meta_per_scene = [
        deepcopy(getattr(ms_bundles[w], 'last_cs_meta', None))
        for w in range(S)
    ]

    # ---- Step 2: Aggregate box-level LB ----
    finite_box_cs = [c for c in box_c_vals if math.isfinite(c)]
    if MS_AGG == "sum":
        box_LB = float(np.sum(finite_box_cs)) if finite_box_cs else float('-inf')
    elif MS_AGG == "mean":
        box_LB = float(np.mean(finite_box_cs)) if finite_box_cs else float('-inf')
    else:
        box_LB = float(np.sum(finite_box_cs)) if finite_box_cs else float('-inf')

    print(f"[Iter {iter_num}] Box-level LB = {box_LB:.6e}  "
          f"({n_box_ok}/{S} scenarios OK)", flush=True)

    # ---- [DIAG-A/B] Iter-0 box-CS provenance trace ----
    _prov_lines = []
    _prov_lines.append(f"\n{'=' * 80}")
    _prov_lines.append(f"[Iter {iter_num}] BOX-CS PROVENANCE TRACE  (MS_AGG={MS_AGG})")
    _prov_lines.append(f"{'=' * 80}")
    _prov_lines.append(f"  {'scen':>5s} {'status':>12s} {'term':>18s} {'dual_bound':>16s}"
                       f" {'primal_obj':>16s} {'selected':>16s} {'source':>16s} {'valid_lb':>8s}")
    _prov_lines.append(f"  {'-'*5:>5s} {'-'*12:>12s} {'-'*18:>18s} {'-'*16:>16s}"
                       f" {'-'*16:>16s} {'-'*16:>16s} {'-'*16:>16s} {'-'*8:>8s}")
    for _pw in range(S):
        _m = box_cs_meta_per_scene[_pw] if box_cs_meta_per_scene[_pw] is not None else {}
        _stat = _m.get('status', '?')
        _term = _m.get('termination_condition', '?')
        _dual = _m.get('dual_bound', None)
        _prim = _m.get('primal_obj', None)
        _sel = box_c_vals[_pw]
        # Determine source label
        if not math.isfinite(_sel):
            _src = "FAILED(-inf)"
        elif _dual is not None and math.isfinite(_dual) and abs(_sel - _dual) < 1e-12:
            _src = "dual_bound"
        elif _prim is not None and math.isfinite(_prim) and abs(_sel - _prim) < 1e-12:
            _src = "primal_obj"
        else:
            _src = "dual_bound"  # solve_const_cut_box returns dual_bound as c_val
        _valid = math.isfinite(_sel)
        _prov_lines.append(
            f"  {_pw:>5d} {_stat:>12s} {str(_term)[:18]:>18s}"
            f" {(_dual if _dual is not None else float('nan')):>16.9f}"
            f" {(_prim if _prim is not None else float('nan')):>16.9f}"
            f" {_sel:>16.9f} {_src:>16s} {str(_valid):>8s}"
        )
    # Aggregation summary
    _n_finite = len(finite_box_cs)
    _n_nonfinite = S - _n_finite
    _prov_lines.append(f"\n  --- Aggregation ---")
    _prov_lines.append(f"  Finite CS values:    {_n_finite}/{S}")
    _prov_lines.append(f"  Non-finite (skipped): {_n_nonfinite}")
    _prov_lines.append(f"  Aggregation mode:    MS_AGG = {MS_AGG!r}")
    if MS_AGG == 'sum':
        _prov_lines.append(f"  Formula:             box_LB = sum(finite_cs_vals)")
    elif MS_AGG == 'mean':
        _prov_lines.append(f"  Formula:             box_LB = mean(finite_cs_vals)")
    else:
        _prov_lines.append(f"  Formula:             box_LB = sum(finite_cs_vals) [default]")
    _prov_lines.append(f"  box_LB (sum):        {box_LB:.12f}")
    _prov_lines.append(f"  box_LB (per-scen):   {box_LB / S:.12f}")
    _prov_lines.append(f"  NOTE: box_LB = sum of per-scenario dual bounds from solve_const_cut_box.")
    _prov_lines.append(f"         Vertex Q-values (fverts) are NOT included in this LB.")
    _prov_lines.append(f"{'=' * 80}\n")
    for _pl in _prov_lines:
        print(_pl, flush=True)
    sys.stdout.flush()

    # ---- Write provenance diagnostics directly to debug files ----
    _prov_block = "\n".join([
        "=== ITER-0 BOX-CS PROVENANCE START ===",
        *_prov_lines,
        "=== ITER-0 BOX-CS PROVENANCE END ===",
        "",
    ])
    if dbg_debug_path is not None:
        try:
            with open(dbg_debug_path, "a", encoding="utf-8") as _f:
                _f.write(_prov_block)
                _f.flush()
        except Exception:
            pass
    if dbg_lb_split_path is not None:
        try:
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _f:
                _f.write(_prov_block)
                _f.flush()
        except Exception:
            pass

    # ---- [DIAG-C] Iter0 Box-CS Lower-Bound Check ----
    _check_lines = []
    _check_lines.append(f"\n{'=' * 80}")
    _check_lines.append(f"=== ITER0 BOX-CS LOWER-BOUND CHECK START ===")
    
    _init_bounds = [(float(v.lb), float(v.ub)) for v in first_vars_list[0]]
    for w in range(S):
        _bun = ms_bundles[w]
        _check_lines.append(f"\n--- Scenario {w} ---")
        
        # [A] Compare bounds
        _orig = getattr(_bun, "_orig_bounds", None)
        _check_lines.append(f"Initial simplex bounds (first_vars_list[0]): {_init_bounds}")
        _check_lines.append(f"Bundle _orig_bounds used in solve_const_cut_box: {_orig}")
        if _orig == _init_bounds:
            _check_lines.append("Bounds match EXACTLY: True")
        else:
            _check_lines.append("Bounds match EXACTLY: False")
            _check_lines.append("Mismatch details:")
            if _orig is None:
                _check_lines.append("  _orig_bounds is None")
            else:
                for _i, (_v, _o) in enumerate(zip(_init_bounds, _orig)):
                    if _v != _o:
                        _v_name = getattr(first_vars_list[0][_i], "name", f"var_{_i}")
                        _check_lines.append(f"  {_v_name}: init={_v}, bundle={_o}")

        # [B] objective
        _act_obj = None
        _obj_expr_str = "unavailable"
        if hasattr(_bun, "obj_const"):
            _act_obj = "obj_const"
            try:
                _obj_expr_str = str(_bun.obj_const.expr)[:150]
            except Exception:
                pass
        _check_lines.append(f"Active objective in solve_const_cut_box: {_act_obj}")
        _check_lines.append("Note: this should represent the surrogate/constant-cut form of Q_s.")
        _check_lines.append(f"Expression (truncated): {_obj_expr_str}")

        # [C] re-evaluate Q_s(cand_pt)
        _c_pt = box_c_pts[w]
        _sel_cs = box_c_vals[w]
        _m = box_cs_meta_per_scene[w] if box_cs_meta_per_scene[w] is not None else {}
        _db = _m.get('dual_bound', None)
        _pr = _m.get('primal_obj', None)
        
        _check_lines.append(f"cand_pt returned: {_c_pt}")
        _check_lines.append(f"dual_bound: {_db}")
        _check_lines.append(f"primal_obj: {_pr}")
        _check_lines.append(f"selected c_s: {_sel_cs}")
        
        if _c_pt is not None and base_bundles is not None:
            try:
                # evaluate_Q_at is imported from utils
                _true_q = evaluate_Q_at(base_bundles[w], first_vars_list[w], _c_pt)
                _check_lines.append(f"true_Q_at_cand = Q_s(cand_pt) = {_true_q}")
                
                _diff_sel = _sel_cs - _true_q if math.isfinite(_sel_cs) else float('nan')
                _check_lines.append(f"selected_minus_trueQ: {_diff_sel}")
                
                if _db is not None and math.isfinite(_db):
                    _check_lines.append(f"dual_minus_trueQ: {_db - _true_q}")
                
                if _pr is not None and math.isfinite(_pr):
                    _check_lines.append(f"primal_minus_trueQ: {_pr - _true_q}")
                
                _sel_le = bool(_diff_sel <= 1e-6)
                _check_lines.append(f"selected_le_trueQ: {_sel_le}")
                
                _db_le = bool(_db <= _true_q + 1e-6) if (_db is not None and math.isfinite(_db)) else None
                _check_lines.append(f"dual_le_trueQ: {_db_le}")
                
                _pr_match = bool(abs(_pr - _true_q) <= 1e-6) if (_pr is not None and math.isfinite(_pr)) else None
                _check_lines.append(f"primal_matches_trueQ: {_pr_match}")
                
                if not _sel_le:
                    _check_lines.append(f"WARNING: selected_c_s > true_Q_at_cand by {_diff_sel}! Lower bound invalid.")
            except Exception as e:
                _check_lines.append(f"Error evaluating true Q: {e}")
        else:
            _check_lines.append("Could not evaluate true_Q_at_cand (cand_pt is None or base_bundles missing).")

    _check_lines.append(f"=== ITER0 BOX-CS LOWER-BOUND CHECK END ===")
    _check_lines.append(f"{'=' * 80}\n")
    
    for _dl in _check_lines:
        print(_dl, flush=True)

    _check_block = "\n".join(_check_lines) + "\n"
    if dbg_debug_path is not None:
        try:
            with open(dbg_debug_path, "a", encoding="utf-8") as _f:
                _f.write(_check_block)
                _f.flush()
        except Exception:
            pass
    if dbg_lb_split_path is not None:
        try:
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _f:
                _f.write(_check_block)
                _f.flush()
        except Exception:
            pass



    # ---- Step 3: Build per_tet records with shared box-level data ----
    per_tet = []
    for k, simp in enumerate(simplices):
        idxs = list(map(int, simp))
        verts = [tuple(pts[i]) for i in idxs]

        V_arr = np.array(verts, dtype=float)
        try:
            vol = simplex_volume(V_arr)
        except Exception:
            vol = 0.0
        _vol_tol = vol_tolerance(pts, d)
        if vol < _vol_tol:
            continue

        simplex_id = tuple(sorted(idxs))
        if tracker is not None:
            tracker.note_created(simplex_id)

        fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]
        fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(n_verts)]

        # Infeasible vertex count
        n_infeas_verts = 0
        for j in range(n_verts):
            for s in range(S):
                if fverts_per_scene[s][j] >= 1e5 - 1e-9:
                    n_infeas_verts += 1
                    break

        # c_agg is the same box_LB for all simplices
        if MS_AGG == "sum":
            c_total = float(np.sum(finite_box_cs)) if finite_box_cs else float('-inf')
        elif MS_AGG == "mean":
            c_total = float(np.mean(finite_box_cs)) if finite_box_cs else float('-inf')
        else:
            c_total = float(np.sum(finite_box_cs)) if finite_box_cs else float('-inf')

        # LB terms
        min_f = float(np.min(fverts_sum))
        c_arr = np.asarray(box_c_vals, float)
        finite_c_mask = np.isfinite(c_arr)
        c_total_finite = float(np.sum(c_arr[finite_c_mask])) if np.any(finite_c_mask) else float('nan')
        c_total_all = float(np.sum(c_arr))

        lb_terms = {
            "min_fverts_sum": min_f,
            "LB_linear": float('inf'),  # MS not computed
            "ms_total": float('inf'),
            "c_total_finite": c_total_finite,
            "c_total_all": c_total_all,
            "lb_case": "box_cs_iter0",
        }

        # Readiness flags
        cs_status = "full" if n_box_ok == S else ("partial" if n_box_ok > 0 else "none")

        # ---- Iteration-0 placeholder values ----
        # Because iter-0 uses shared box-level CS only:
        #   ms = inf, UB = inf, best_scene = None, x_ms_best_scene = None
        # are intentional placeholders.  MS and simplex-specific UB are
        # deferred and computed lazily by ensure_ms_for_simplex() when
        # the selected simplex needs them.
        #
        # NOTE: box-level CS data is NOT written to ms_cache under any
        # simplex key.  Full-box c_s values must not be confused with
        # simplex-specific c_s cache entries.  From iteration 1 onward,
        # per-simplex solves populate the cache fresh.

        per_tet.append({
            "simplex_index": k,
            "vert_idx": idxs,
            "verts": verts,
            "fverts_sum": fverts_sum,
            "ms_per_scene": [float('inf')] * S,        # placeholder: MS not computed
            "xms_per_scene": [None] * S,                # placeholder: no MS points
            "c_per_scene": list(box_c_vals),             # SHARED box-level c_s
            "c_point_per_scene": list(box_c_pts),        # SHARED box-level c_s points
            "ms": float('inf'),                          # placeholder: MS not computed
            "c_agg": c_total,
            "LB": box_LB,                                # SAME for all simplices
            "UB": float('inf'),                          # placeholder: no per-simplex UB
            "x_ms_best_scene": None,                     # placeholder: deferred
            "best_scene": None,                          # placeholder: deferred
            "volume": vol,
            "n_infeas_verts": n_infeas_verts,
            "ms_meta_per_scene": [None] * S,             # MS metadata: not computed yet
            "cs_meta_per_scene": deepcopy(box_cs_meta_per_scene),  # SHARED box-level CS metadata
            "LB_terms": lb_terms,
            "cs_status": cs_status,
            "ms_status": "none",                         # explicitly "not computed yet"
            "c_fallback_per_scene": list(box_c_vals),     # box CS as initial fallback
            "c_fallback_source_per_scene": ["box"] * S,
            # ---- INACTIVE fields (iter-0 records are always active) ----
            "inactive": False,
            "inactive_reason": None,
        })

    return tri, per_tet, box_c_vals, box_c_pts, box_LB


# ---- CS-only evaluation of all tetrahedra ----
def evaluate_all_tetra_cs_only(nodes, scen_values, ms_bundles, first_vars_list,
                               ms_cache=None, cache_on=True, tracker=None,
                               tet_mesh: SimplexMesh | None = None,
                               lb_sur_cache=None,
                               dbg_timelimit_path=None,
                               dbg_cs_timing_path=None,
                               dbg_ms_timing_path=None,
                               iter_num=None,
                               _nonopt_buf=None,
                               _debug_buf=None):
    """
    CS-only version of evaluate_all_tetra.
    Computes ONLY constant-cut (c_s) for each simplex×scenario; skips MS.
    Records have ms_status="none", cs_status="full"/"partial".
    LB is computed via sum(finite_c_s) — valid lower bound.
    """
    print("[DBG] evaluate_all_tetra_cs_only: start", flush=True)
    pts = np.asarray(nodes, dtype=float)
    d = pts.shape[1] if pts.ndim == 2 else 3
    n_verts = d + 1
    if len(pts) < n_verts:
        return None, []

    if tet_mesh is not None:
        simplices = [list(t) for t in tet_mesh.tets]
        tri = tet_mesh.as_delaunay_like()
    else:
        tri = Delaunay(pts)
        simplices = tri.simplices
    print(f"[DBG] evaluate_all_tetra_cs_only: {len(simplices)} simplices", flush=True)

    S = len(ms_bundles)

    per_tet = []
    for k, simp in enumerate(simplices):
        idxs = list(map(int, simp))
        verts = [tuple(pts[i]) for i in idxs]

        V_arr = np.array(verts, dtype=float)
        try:
            vol = simplex_volume(V_arr)
        except Exception:
            vol = 0.0
        _vol_tol = vol_tolerance(pts, d)
        if vol < _vol_tol:
            continue

        simplex_id = tuple(sorted(idxs))
        if tracker is not None:
            tracker.note_created(simplex_id)

        fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]
        fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(n_verts)]

        key_base = tuple(sorted(idxs))
        ms_scene = []
        xms_scene = []
        c_scene = []
        cpts_scene = []
        ms_meta_per_scene = []
        cs_meta_per_scene = []
        n_cs_ok = 0

        for w in range(S):
            cache_key = (int(w), key_base)
            hit = False
            if cache_on and ms_cache is not None and cache_key in ms_cache:
                entry = ms_cache[cache_key]
                entry = _upgrade_cache_entry_old_format(entry)
                if _cache_status_has_cs(entry[0]):
                    # Reuse CS from cache
                    _, _, _, c_val, c_pt = entry
                    hit = True

            if hit:
                ms_val = float('inf')
                new_pt_ms = None
                ms_meta_per_scene.append(None)
                cs_meta_per_scene.append(None)
            else:
                ms_val, new_pt_ms, c_val, c_pt, _solve_state = cs_only_on_tetra_for_scene(
                    ms_bundles[w], verts, fverts_per_scene[w]
                )
                _cs_meta = getattr(ms_bundles[w], 'last_cs_meta', None)
                ms_meta_per_scene.append(None)  # MS not attempted
                cs_meta_per_scene.append(_cs_meta)

                # Buffer non-optimal CS events
                if _nonopt_buf is not None and _cs_meta is not None:
                    _st = _cs_meta.get("status", "")
                    if _st == "time_limit" or _st != "optimal":
                        _nonopt_buf.append(
                            f"  iter={iter_num} simp=T{k} scen={w} type=cs "
                            f"status={_st} "
                            f"term={_cs_meta.get('termination_condition','?')} "
                            f"ok={_cs_meta.get('ok','?')} "
                            f"val={_cs_meta.get('dual_bound','?')} "
                            f"time={_cs_meta.get('time_sec',0.0):.4f}s\n"
                        )

                # Write to cache as cs_only
                cs_ok = (c_val is not None and math.isfinite(c_val))
                cache_status = "cs_only" if cs_ok else "both_failed"
                if cache_on and ms_cache is not None:
                    ms_cache[cache_key] = (cache_status, float('inf'), None, c_val, c_pt)

                # Note: in CS-only path, do NOT call tracker.note_ms_recomputed
                # because MS was not computed

            if c_val is not None and math.isfinite(c_val):
                n_cs_ok += 1

            ms_scene.append(float('inf'))
            xms_scene.append(None)
            c_scene.append(c_val)
            cpts_scene.append(c_pt)

        # Aggregates
        if MS_AGG == "sum":
            ms_total = float('inf')
            c_total = float(np.sum([c for c in c_scene if math.isfinite(c)])) if any(math.isfinite(c) for c in c_scene) else float('-inf')
        elif MS_AGG == "mean":
            ms_total = float('inf')
            finite_cs = [c for c in c_scene if math.isfinite(c)]
            c_total = float(np.mean(finite_cs)) if finite_cs else float('-inf')
        else:
            raise ValueError("MS_AGG must be 'sum' or 'mean'")

        UB = float('inf')  # No MS → UB undefined

        # LB via solve_surrogate_lb_for_tet — handles ms=+inf by returning sum(finite_c_s)
        if lb_sur_cache is not None and key_base in lb_sur_cache:
            LB_sur = lb_sur_cache[key_base]
        else:
            LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene)
            if lb_sur_cache is not None:
                lb_sur_cache[key_base] = LB_sur

        # Infeasible vertex count
        n_infeas_verts = 0
        for j in range(n_verts):
            for s in range(S):
                if fverts_per_scene[s][j] >= 1e5 - 1e-9:
                    n_infeas_verts += 1
                    break

        # LB terms
        min_f = float(np.min(fverts_sum))
        c_arr = np.asarray(c_scene, float)
        finite_c_mask = np.isfinite(c_arr)
        c_total_finite = float(np.sum(c_arr[finite_c_mask])) if np.any(finite_c_mask) else float('nan')
        c_total_all = float(np.sum(c_arr))

        lb_terms = {
            "min_fverts_sum": min_f,
            "LB_linear": float('inf'),  # MS not computed
            "ms_total": float('inf'),
            "c_total_finite": c_total_finite,
            "c_total_all": c_total_all,
            "lb_case": "ms_not_attempted",
        }

        # Readiness flags
        cs_status = "full" if n_cs_ok == S else ("partial" if n_cs_ok > 0 else "none")

        per_tet.append({
            "simplex_index": k,
            "vert_idx": idxs,
            "verts": verts,
            "fverts_sum": fverts_sum,
            "ms_per_scene": ms_scene,
            "xms_per_scene": xms_scene,
            "c_per_scene": c_scene,
            "c_point_per_scene": cpts_scene,
            "ms": ms_total,
            "c_agg": c_total,
            "LB": LB_sur,
            "UB": UB,
            "x_ms_best_scene": None,
            "best_scene": None,
            "volume": vol,
            "n_infeas_verts": n_infeas_verts,
            "ms_meta_per_scene": ms_meta_per_scene,
            "cs_meta_per_scene": cs_meta_per_scene,
            "LB_terms": lb_terms,
            # Readiness flags
            "cs_status": cs_status,
            "ms_status": "none",
            "c_fallback_per_scene": list(c_scene),        # native CS = fallback initially
            "c_fallback_source_per_scene": ["self" if (c_scene[s] is not None and math.isfinite(c_scene[s])) else "none" for s in range(S)],
            # ---- INACTIVE fields (CS-only iter-0 records are always active) ----
            "inactive": False,
            "inactive_reason": None,
        })

    return tri, per_tet


def _normalize_inactive_record(rec, S, reason):
    """Normalize all status/bound fields on a record being marked inactive."""
    rec["inactive"] = True
    rec["inactive_reason"] = reason
    rec["ms_status"] = "none"
    rec["cs_status"] = "none"
    rec["LB"] = float('inf')
    rec["UB"] = float('inf')
    rec["ms"] = float('inf')
    rec["c_agg"] = float('-inf')
    rec["ms_per_scene"] = [float('inf')] * S
    rec["xms_per_scene"] = [None] * S
    rec["c_per_scene"] = [float('-inf')] * S
    rec["c_point_per_scene"] = [None] * S
    rec["c_fallback_per_scene"] = [float('-inf')] * S
    rec["c_fallback_source_per_scene"] = ["none"] * S
    if "LB_terms" not in rec or not isinstance(rec.get("LB_terms"), dict):
        rec["LB_terms"] = {}
    rec["LB_terms"]["lb_case"] = "tightening_infeasible"
    rec["x_ms_best_scene"] = None
    rec["best_scene"] = None


# ---- Lazy ensure helpers ----
def ensure_ms_for_simplex(rec, ms_bundles, nodes, scen_values, ms_cache,
                          lb_sur_cache, S, verbose=False, iter_num=None,
                          use_fbbt=False, use_obbt=False,
                          obbt_solver_name="gurobi", obbt_solver_opts=None,
                          obbt_tol=1e-1, max_obbt_rounds=3,
                          _debug_buf=None):
    """
    Lazily compute MS for a simplex record that may only have CS.
    Updates rec in-place. Per-scenario cache entries are updated individually.
    """
    if rec.get("ms_status", "full") == "full":
        return  # Already fully evaluated
    if rec.get("inactive", False):
        return  # Skip inactive simplices — they cannot be re-materialized

    pts = np.asarray(nodes, dtype=float)
    idxs = rec["vert_idx"]
    verts = rec["verts"]
    key_base = tuple(sorted(idxs))
    fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]

    ms_scene = list(rec.get("ms_per_scene", [float('inf')] * S))
    xms_scene = list(rec.get("xms_per_scene", [None] * S))
    c_scene = list(rec.get("c_per_scene", [float('-inf')] * S))
    cpts_scene = list(rec.get("c_point_per_scene", [None] * S))
    ms_meta_per_scene = list(rec.get("ms_meta_per_scene", [None] * S))
    cs_meta_per_scene = list(rec.get("cs_meta_per_scene", [None] * S))

    n_ms_ok = 0
    n_cs_ok = 0

    for w in range(S):
        cache_key = (int(w), key_base)

        # Check cache for tightening_infeasible
        already_has_ms = False
        if ms_cache is not None and cache_key in ms_cache:
            entry = _upgrade_cache_entry_old_format(ms_cache[cache_key])
            if _cache_status_is_tightening_ms_skip(entry[0]):
                # MS was previously infeasible by tightening, but CS is available.
                # Skip MS, keep simplex alive with CS-only support.
                ms_scene[w] = float('inf')
                xms_scene[w] = None
                # Reuse CS from cache (guaranteed available for ms_skip)
                if _cache_status_has_cs(entry[0]):
                    c_scene[w] = entry[3]
                    cpts_scene[w] = entry[4]
                already_has_ms = True  # no fresh solve needed
                if _debug_buf is not None:
                    _debug_buf.append(
                        f"  simp=T{rec['simplex_index']} scen={w} type=ms "
                        f"issue=tightening_ms_skipped_cached ok=False "
                        f"status={entry[0]} term=tightening "
                        f"val=inf dual=None time=0.0000s\n"
                    )
                print(f"[ensure_ms] T{rec['simplex_index']} scen {w}: "
                      f"tightening (cached) → MS skipped, CS available")
            elif _cache_status_is_cs_side_kill(entry[0]):
                # CS-side kill: tightening_infeasible or tightening_ms_skip_cs_failed
                # → normalize inactive and return.
                _normalize_inactive_record(rec, S,
                    f"cs_side_kill_scene_{w}_{entry[0]}")
                if _debug_buf is not None:
                    _debug_buf.append(
                        f"  simp=T{rec['simplex_index']} scen={w} type=cs "
                        f"issue={entry[0]}_cached ok=False "
                        f"status={entry[0]} term=tightening "
                        f"val=-inf dual=None time=0.0000s\n"
                    )
                print(f"[ensure_ms] T{rec['simplex_index']} scen {w}: "
                      f"{entry[0]} (cached) → CS unavailable, simplex inactive")
                return
            elif _cache_status_has_ms(entry[0]):
                # Reuse MS from cache
                _, ms_val, new_pt_ms, c_v, c_p = entry
                ms_scene[w] = ms_val
                xms_scene[w] = new_pt_ms
                # Also pick up CS if better than what we have
                if _cache_status_has_cs(entry[0]):
                    c_scene[w] = c_v
                    cpts_scene[w] = c_p
                already_has_ms = True

        if not already_has_ms:
            # Fresh solve: ms_on_tetra_for_scene does both MS and CS
            ms_val, new_pt_ms, c_val, c_pt, solve_state = ms_on_tetra_for_scene(
                ms_bundles[w], verts, fverts_per_scene[w],
                use_fbbt=use_fbbt, use_obbt=use_obbt,
                obbt_solver_name=obbt_solver_name,
                obbt_solver_opts=obbt_solver_opts,
                obbt_tol=obbt_tol,
                max_obbt_rounds=max_obbt_rounds,
            )

            # Tightening infeasible: ms_on_tetra_for_scene returns
            # "tightening_ms_skip_cs_ok" or "tightening_ms_skip_cs_failed".
            if solve_state == "tightening_ms_skip_cs_ok":
                # MS unavailable, but fallback CS succeeded → keep alive
                if ms_cache is not None:
                    ms_cache[cache_key] = (solve_state,
                                           float('inf'), None, c_val, c_pt)
                if _debug_buf is not None:
                    _debug_buf.append(
                        f"  simp=T{rec['simplex_index']} scen={w} type=ms "
                        f"issue={solve_state} ok=False "
                        f"status={solve_state} term=tightening "
                        f"val=inf dual=None time=0.0000s\n"
                    )
                print(f"[ensure_ms] T{rec['simplex_index']} scen {w}: "
                      f"{solve_state} → MS skipped, fallback CS available")
                # Fall through: ms_val/c_val already set correctly

            elif solve_state == "tightening_ms_skip_cs_failed":
                # MS unavailable AND fallback CS failed → CS-side kill
                if ms_cache is not None:
                    ms_cache[cache_key] = (solve_state,
                                           float('inf'), None, float('-inf'), None)
                _normalize_inactive_record(rec, S,
                    f"tightening_ms_skip_cs_failed_scene_{w}")
                if _debug_buf is not None:
                    _debug_buf.append(
                        f"  simp=T{rec['simplex_index']} scen={w} type=cs "
                        f"issue={solve_state} ok=False "
                        f"status={solve_state} term=tightening "
                        f"val=-inf dual=None time=0.0000s\n"
                    )
                print(f"[ensure_ms] T{rec['simplex_index']} scen {w}: "
                      f"{solve_state} → CS unavailable, simplex inactive")
                return

            ms_scene[w] = ms_val
            xms_scene[w] = new_pt_ms
            c_scene[w] = c_val
            cpts_scene[w] = c_pt

            _ms_meta = getattr(ms_bundles[w], 'last_solve_meta', None)
            _cs_meta = getattr(ms_bundles[w], 'last_cs_meta', None)
            ms_meta_per_scene[w] = _ms_meta
            cs_meta_per_scene[w] = _cs_meta

            # --- Log failed MS/CS to debug buffer ---
            # Failure = dual bound not obtainable.
            # Time-limit with valid dual bound is NOT a failure.
            if _debug_buf is not None:
                _simp_idx = rec['simplex_index']
                for _tag, _meta, _val in [("ms", _ms_meta, ms_val), ("cs", _cs_meta, c_val)]:
                    if _meta is None:
                        continue
                    _dual = _meta.get("dual_bound")
                    _dual_ok = (_dual is not None
                                and isinstance(_dual, (int, float))
                                and math.isfinite(_dual))
                    if _tag == "cs":
                        _should_log = not _dual_ok
                        _issue = "no_dual_bound"
                    else:
                        _st = _meta.get("status", "")
                        _ok = _meta.get("ok", True)
                        _tm = _meta.get("termination_condition", "")
                        _should_log = (not _ok
                                       or _st in ("time_limit", "error", "infeasible", "unbounded")
                                       or _tm not in ("optimal", "locallyOptimal", ""))
                        _issue = "solve_fail"
                    if _should_log:
                        _debug_buf.append(
                            f"  simp=T{_simp_idx} scen={w} type={_tag} issue={_issue} "
                            f"ok={_meta.get('ok','?')} status={_meta.get('status','')} "
                            f"term={_meta.get('termination_condition','')} "
                            f"val={_val} dual={_dual} "
                            f"time={_meta.get('time_sec',0.0):.4f}s\n"
                        )

            # Per-scenario cache update with correct status
            ms_ok = (ms_val is not None and math.isfinite(ms_val))
            cs_ok = (c_val is not None and math.isfinite(c_val))
            if ms_ok and cs_ok:
                cache_st = "full"
            elif ms_ok and not cs_ok:
                cache_st = "ms_only"
            elif not ms_ok and cs_ok:
                cache_st = "ms_failed"
            else:
                cache_st = "both_failed"

            if ms_cache is not None:
                ms_cache[cache_key] = (cache_st, ms_val, new_pt_ms, c_val, c_pt)

        # Count successes
        if math.isfinite(ms_scene[w]):
            n_ms_ok += 1
        if math.isfinite(c_scene[w]):
            n_cs_ok += 1

    # Update record in-place
    rec["ms_per_scene"] = ms_scene
    rec["xms_per_scene"] = xms_scene
    rec["c_per_scene"] = c_scene
    rec["c_point_per_scene"] = cpts_scene
    rec["ms_meta_per_scene"] = ms_meta_per_scene
    rec["cs_meta_per_scene"] = cs_meta_per_scene

    # Update c_fallback_per_scene: native CS success updates fallback
    fb = list(rec.get("c_fallback_per_scene", [None] * S))
    fb_src = list(rec.get("c_fallback_source_per_scene", ["none"] * S))
    for w in range(S):
        if c_scene[w] is not None and math.isfinite(c_scene[w]):
            fb[w] = c_scene[w]
            fb_src[w] = "self"
    rec["c_fallback_per_scene"] = fb
    rec["c_fallback_source_per_scene"] = fb_src

    # Effective c for LB: native if finite, else fallback if finite
    eff_c_scene = _effective_c_scene(rec)

    # Aggregates (use effective c for c_agg)
    if MS_AGG == "sum":
        rec["ms"] = float(np.sum(ms_scene))
        finite_cs = [c for c in eff_c_scene if math.isfinite(c)]
        rec["c_agg"] = float(np.sum(finite_cs)) if finite_cs else float('-inf')
    elif MS_AGG == "mean":
        rec["ms"] = float(np.mean(ms_scene))
        finite_cs = [c for c in eff_c_scene if math.isfinite(c)]
        rec["c_agg"] = float(np.mean(finite_cs)) if finite_cs else float('-inf')

    # best_scene / x_ms_best_scene
    finite_ms_indices = [w for w in range(S) if math.isfinite(ms_scene[w])]
    if finite_ms_indices:
        best_w = min(finite_ms_indices, key=lambda w: ms_scene[w])
        rec["best_scene"] = best_w
        rec["x_ms_best_scene"] = xms_scene[best_w]
    else:
        rec["best_scene"] = None
        rec["x_ms_best_scene"] = None

    # UB
    fverts_sum = rec.get("fverts_sum", [])
    if fverts_sum and math.isfinite(rec["ms"]):
        rec["UB"] = float(np.max(fverts_sum) + rec["ms"])
    else:
        rec["UB"] = float('inf')

    # Recompute LB (using effective c)
    fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]
    LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, eff_c_scene)
    rec["LB"] = LB_sur

    # Invalidate lb_sur_cache
    if lb_sur_cache is not None and key_base in lb_sur_cache:
        lb_sur_cache[key_base] = LB_sur

    # LB_terms update (using effective c for consistency)
    min_f = float(np.min(fverts_sum)) if fverts_sum else float('inf')
    ms_total = rec["ms"]
    c_arr = np.asarray(eff_c_scene, float)
    finite_c_mask = np.isfinite(c_arr)
    ms_arr = np.asarray(ms_scene, float)
    all_ms_finite = bool(np.all(np.isfinite(ms_arr)))
    any_finite_c = bool(np.any(finite_c_mask))
    if all_ms_finite and any_finite_c:
        lb_case = "all_ms_finite"
    elif all_ms_finite and not any_finite_c:
        lb_case = "all_ms_finite_no_c"
    elif not all_ms_finite and any_finite_c:
        lb_case = "some_ms_inf_use_c_only"
    else:
        lb_case = "all_fail"
    rec["LB_terms"] = {
        "min_fverts_sum": min_f,
        "LB_linear": min_f + ms_total if math.isfinite(ms_total) else float('inf'),
        "ms_total": ms_total,
        "c_total_finite": float(np.sum(c_arr[finite_c_mask])) if any_finite_c else float('nan'),
        "c_total_all": float(np.sum(c_arr)),
        "lb_case": lb_case,
    }

    # Readiness flags
    rec["ms_status"] = "full" if n_ms_ok == S else ("partial" if n_ms_ok > 0 else "none")
    rec["cs_status"] = "full" if n_cs_ok == S else ("partial" if n_cs_ok > 0 else "none")

    if verbose:
        print(f"[ensure_ms] simplex T{rec['simplex_index']}: "
              f"ms_status={rec['ms_status']}, cs_status={rec['cs_status']}, "
              f"ms={rec['ms']:.6e}, LB={rec['LB']:.6e}")


def ensure_cs_for_simplex(rec, ms_bundles, nodes, scen_values, ms_cache,
                          lb_sur_cache, S, verbose=False, iter_num=None,
                          use_fbbt=False, use_obbt=False,
                          obbt_solver_name="gurobi", obbt_solver_opts=None,
                          obbt_tol=1e-1, max_obbt_rounds=3):
    """
    Lazily compute CS for a simplex record that may be missing CS.
    Symmetric to ensure_ms_for_simplex. Updates rec in-place.
    """
    if rec.get("cs_status", "full") == "full":
        return
    if rec.get("inactive", False):
        return  # Skip inactive simplices

    pts = np.asarray(nodes, dtype=float)
    idxs = rec["vert_idx"]
    verts = rec["verts"]
    key_base = tuple(sorted(idxs))
    fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]

    c_scene = list(rec.get("c_per_scene", [float('-inf')] * S))
    cpts_scene = list(rec.get("c_point_per_scene", [None] * S))
    cs_meta_per_scene = list(rec.get("cs_meta_per_scene", [None] * S))

    n_cs_ok = 0
    for w in range(S):
        cache_key = (int(w), key_base)

        already_has_cs = False
        if ms_cache is not None and cache_key in ms_cache:
            entry = _upgrade_cache_entry_old_format(ms_cache[cache_key])
            if _cache_status_is_cs_side_kill(entry[0]):
                # CS-side kill (tightening_infeasible or tightening_ms_skip_cs_failed)
                _normalize_inactive_record(rec, S, f"cs_side_kill_scene_{w}_{entry[0]}")
                return
            if _cache_status_is_tightening_ms_skip(entry[0]):
                # MS-side tightening skip, CS available — reuse CS from cache
                if _cache_status_has_cs(entry[0]):
                    c_scene[w] = entry[3]
                    cpts_scene[w] = entry[4]
                    already_has_cs = True
                # else: should not happen for cs_ok, but fall through to fresh solve
            if _cache_status_has_cs(entry[0]):
                _, _, _, c_val, c_pt = entry
                c_scene[w] = c_val
                cpts_scene[w] = c_pt
                already_has_cs = True

        if not already_has_cs:
            _, _, c_val, c_pt, solve_state = cs_only_on_tetra_for_scene(
                ms_bundles[w], verts, fverts_per_scene[w],
                use_fbbt=use_fbbt, use_obbt=use_obbt,
                obbt_solver_name=obbt_solver_name,
                obbt_solver_opts=obbt_solver_opts,
                obbt_tol=obbt_tol,
                max_obbt_rounds=max_obbt_rounds,
            )

            if solve_state == "tightening_infeasible":
                if ms_cache is not None:
                    ms_cache[cache_key] = ("tightening_infeasible",
                                           float('inf'), None, float('-inf'), None)
                _normalize_inactive_record(rec, S, f"tightening_infeasible_scene_{w}")
                return

            c_scene[w] = c_val
            cpts_scene[w] = c_pt
            _cs_meta = getattr(ms_bundles[w], 'last_cs_meta', None)
            cs_meta_per_scene[w] = _cs_meta

            # Update cache: merge with existing MS if available
            cs_ok = (c_val is not None and math.isfinite(c_val))
            if ms_cache is not None:
                old_entry = ms_cache.get(cache_key)
                if old_entry is not None:
                    old_entry = _upgrade_cache_entry_old_format(old_entry)
                    ms_ok = _cache_status_has_ms(old_entry[0])
                    if ms_ok and cs_ok:
                        cache_st = "full"
                    elif ms_ok and not cs_ok:
                        cache_st = "cs_failed"
                    elif not ms_ok and cs_ok:
                        cache_st = "cs_only"
                    else:
                        cache_st = "both_failed"
                    ms_cache[cache_key] = (cache_st, old_entry[1], old_entry[2], c_val, c_pt)
                else:
                    cache_st = "cs_only" if cs_ok else "both_failed"
                    ms_cache[cache_key] = (cache_st, float('inf'), None, c_val, c_pt)

        if math.isfinite(c_scene[w]):
            n_cs_ok += 1

    rec["c_per_scene"] = c_scene
    rec["c_point_per_scene"] = cpts_scene
    rec["cs_meta_per_scene"] = cs_meta_per_scene

    # Update c_fallback_per_scene: native CS success updates fallback
    fb = list(rec.get("c_fallback_per_scene", [None] * S))
    fb_src = list(rec.get("c_fallback_source_per_scene", ["none"] * S))
    for w in range(S):
        if c_scene[w] is not None and math.isfinite(c_scene[w]):
            fb[w] = c_scene[w]
            fb_src[w] = "self"
    rec["c_fallback_per_scene"] = fb
    rec["c_fallback_source_per_scene"] = fb_src

    # Effective c for LB: native if finite, else fallback if finite
    eff_c_scene = _effective_c_scene(rec)

    # Recompute c_agg (using effective c)
    finite_cs = [c for c in eff_c_scene if math.isfinite(c)]
    if MS_AGG == "sum":
        rec["c_agg"] = float(sum(finite_cs)) if finite_cs else float('-inf')
    else:
        rec["c_agg"] = float(np.mean(finite_cs)) if finite_cs else float('-inf')

    # Recompute LB (using effective c)
    ms_scene = rec.get("ms_per_scene", [float('inf')] * S)
    fverts_per_scene2 = [[scen_values[s][i] for i in idxs] for s in range(S)]
    LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene2, ms_scene, eff_c_scene)
    rec["LB"] = LB_sur
    if lb_sur_cache is not None:
        lb_sur_cache[key_base] = LB_sur

    # LB_terms update (using effective c for consistency with LB)
    fverts_sum = rec.get("fverts_sum", [])
    min_f = float(np.min(fverts_sum)) if fverts_sum else float('inf')
    ms_total = rec.get("ms", float('inf'))
    c_arr = np.asarray(eff_c_scene, float)
    finite_c_mask = np.isfinite(c_arr)
    ms_arr = np.asarray(ms_scene, float)
    all_ms_finite = bool(np.all(np.isfinite(ms_arr)))
    any_finite_c = bool(np.any(finite_c_mask))
    if all_ms_finite and any_finite_c:
        lb_case = "all_ms_finite"
    elif all_ms_finite and not any_finite_c:
        lb_case = "all_ms_finite_no_c"
    elif not all_ms_finite and any_finite_c:
        lb_case = "some_ms_inf_use_c_only"
    else:
        lb_case = "all_fail"
    rec["LB_terms"] = {
        "min_fverts_sum": min_f,
        "LB_linear": min_f + ms_total if math.isfinite(ms_total) else float('inf'),
        "ms_total": ms_total,
        "c_total_finite": float(np.sum(c_arr[finite_c_mask])) if any_finite_c else float('nan'),
        "c_total_all": float(np.sum(c_arr)),
        "lb_case": lb_case,
    }

    rec["cs_status"] = "full" if n_cs_ok == S else ("partial" if n_cs_ok > 0 else "none")

    if verbose:
        print(f"[ensure_cs] simplex T{rec['simplex_index']}: "
              f"cs_status={rec['cs_status']}, LB={rec['LB']:.6e}")


# ------------------------- Evaluate all tetrahedra (per-scene) -------------------------
def evaluate_all_tetra(nodes, scen_values, ms_bundles, first_vars_list,
                       ms_cache=None, cache_on=True, tracker=None,
                       tet_mesh: SimplexMesh | None = None,
                       lb_sur_cache=None,  # LB surrogate cache
                       dbg_timelimit_path=None,  # path for timeout log
                       dbg_cs_timing_path=None,  # path for CS (Q-value) timing log
                       dbg_ms_timing_path=None,  # path for MS timing log
                       iter_num=None,  # iteration number for logging
                       _nonopt_buf=None,  # buffer for non-optimal events (list, append-only)
                       _debug_buf=None,  # buffer for simplex_debug.txt (list, append-only)
                       use_fbbt=False,  # FBBT before each scenario solve
                       use_obbt=False,  # OBBT before each scenario solve
                       obbt_solver_name="gurobi",
                       obbt_solver_opts=None,
                       obbt_tol=1e-1,
                       max_obbt_rounds=3):
    """
    Evaluate all Delaunay simplex formed by the node set
    across all scenarios, computing their ms values,
    lower/upper bounds, and candidate points.

    For each simplex(tetrahedron):
        - It gathers objective values at the four vertices for each scenario.
        - Solves the ms subproblem per scenario (with caching to skip repeats).
        - Aggregates per-scene ms values into a single ms (via MS_AGG).
        - Computes LB/UB and identifies the best scene and candidate point.

    Parameters
    ----------
    nodes : list[tuple[float]]
        Current first-stage points (x1, x2, ..., xd).
    scen_values : list[list[float]]
        Cached Q evaluations for each scenario s at each node i.
        Shape: [S][N].
    ms_bundles : list[MSBundle]
        Scenario-specific persistent ms solvers.


    first_vars_list : list[list[pyo.Var]]
        Corresponding first-stage Pyomo variables for each scenario.
    ms_cache : dict, optional
        Cache {(scene_idx, sorted(vert_idx)) -> (ms_val, new_point)}.
    cache_on : bool, default=True
        Whether to use and update ms_cache.
    tracker : SimplexTracker, optional
        Records bookkeeping events (created simplex, ms recomputed, etc.).

    Returns
    -------
    tri : scipy.spatial.Delaunay
        The Delaunay triangulation of current nodes.
    per_tet : list[dict]
        List of simplex records containing vertices, ms results,
        LB/UB values, best scene, candidate point, and volume.
    """
    print("[DBG] evaluate_all_tetra: start", flush=True)
    pts = np.asarray(nodes, dtype=float)
    d = pts.shape[1] if pts.ndim == 2 else 3
    n_verts = d + 1
    if len(pts) < n_verts:
        return None, []

    print("[DBG] evaluate_all_tetra: building mesh ...", flush=True)
    if tet_mesh is not None:
        simplices = [list(t) for t in tet_mesh.tets]
        tri = tet_mesh.as_delaunay_like()
    else:
        tri = Delaunay(pts)  # divide into several non-overlapping simplex from pts
        simplices = tri.simplices
    print(f"[DBG] evaluate_all_tetra: {len(simplices)} simplices, entering loop ...", flush=True)

    S = len(ms_bundles)

    mins = pts.min(axis=0)

    maxs = pts.max(axis=0)
    diam = float(np.linalg.norm(maxs - mins))
    vol_tol = 1e-12 * max(diam**3, 1.0)

    # per_tet stores information for every simplex in the current iteration
    per_tet = []
    for k, simp in enumerate(simplices):
        idxs = list(map(int, simp))
        verts = [tuple(pts[i]) for i in idxs]

        # Generic d-dim volume calculation
        V_arr = np.array(verts, dtype=float)  # (d+1, d)
        try:
            vol = simplex_volume(V_arr)
        except Exception as _e_vol:
            vol = 0.0
        vol_tol = vol_tolerance(pts, d)
        if vol < vol_tol:
            continue

        # Use the ordered tuple of vertex index as the unique ID of the simplex
        simplex_id = tuple(sorted(idxs))
        if tracker is not None:
            tracker.note_created(simplex_id)

        fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]
        fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(n_verts)]

        # ==========  per-scene ms + constant-cut solve with cache ==========
        key_base = tuple(sorted(idxs))
        ms_scene = []
        xms_scene = []
        c_scene = []          # c_{T,s}
        cpts_scene = []       # Point corresponding to c_s
        # For logging: store solve metadata per scene
        ms_meta_per_scene = []
        cs_meta_per_scene = []

        _did_fresh_ms = False  # simplex-level: did any scenario need fresh MS solve?
        _simplex_tightening_infeasible = False
        _tighten_infeas_scene = None

        for w in range(S):
            cache_key = (int(w), key_base)
            # Cache hit: only reuse if the entry has usable MS data.
            # After CS-only iter-0, many entries are "cs_only" — those must NOT
            # be treated as full hits, or they'd silently inject ms_val=+inf.
            _cache_entry_raw = None
            hit = False
            if cache_on and (ms_cache is not None) and (cache_key in ms_cache):
                _cache_entry_raw = _upgrade_cache_entry_old_format(ms_cache[cache_key])
                # --- Cached tightening: split by CS availability ---
                if _cache_status_is_tightening_ms_skip(_cache_entry_raw[0]):
                    # MS unavailable, but fallback CS succeeded → keep alive
                    _ms_val_skip = float('inf')
                    _xms_skip = None
                    _c_val_skip = float('-inf')
                    _c_pt_skip = None
                    if _cache_status_has_cs(_cache_entry_raw[0]):
                        _c_val_skip = _cache_entry_raw[3]
                        _c_pt_skip = _cache_entry_raw[4]
                    ms_scene.append(_ms_val_skip)
                    xms_scene.append(_xms_skip)
                    c_scene.append(_c_val_skip)
                    cpts_scene.append(_c_pt_skip)
                    ms_meta_per_scene.append(None)
                    cs_meta_per_scene.append(None)
                    print(f"[eval_all] T{k} scen {w}: {_cache_entry_raw[0]} (cached) "
                          f"→ MS skipped, fallback CS available")
                    continue  # next scenario — simplex stays alive
                if _cache_status_is_cs_side_kill(_cache_entry_raw[0]):
                    # CS unavailable (tightening_infeasible or ms_skip_cs_failed)
                    # → kill this simplex
                    _simplex_tightening_infeasible = True
                    _tighten_infeas_scene = w
                    print(f"[eval_all] T{k} scen {w}: {_cache_entry_raw[0]} (cached) "
                          f"→ CS unavailable, simplex will be killed")
                    break
                if _cache_status_has_ms(_cache_entry_raw[0]):
                    hit = True  # Full hit: MS is usable

            if hit:
                _, ms_val, new_pt_ms, c_val, c_pt = _cache_entry_raw
                # Cached: no fresh metadata, use None
                ms_meta_per_scene.append(None)
                cs_meta_per_scene.append(None)
            else:
                ms_val, new_pt_ms, c_val, c_pt, solve_state = ms_on_tetra_for_scene(
                    ms_bundles[w], verts, fverts_per_scene[w],
                    use_fbbt=use_fbbt, use_obbt=use_obbt,
                    obbt_solver_name=obbt_solver_name,
                    obbt_solver_opts=obbt_solver_opts,
                    obbt_tol=obbt_tol,
                    max_obbt_rounds=max_obbt_rounds,
                )

                # ---- Tightening MS-skip: split by CS availability ----
                if solve_state == "tightening_ms_skip_cs_ok":
                    # MS unavailable, fallback CS succeeded → keep alive
                    if cache_on and (ms_cache is not None):
                        ms_cache[cache_key] = (solve_state,
                                               float('inf'), None, c_val, c_pt)
                    print(f"[eval_all] T{k} scen {w}: {solve_state} "
                          f"→ MS skipped, fallback CS available")
                    ms_meta_per_scene.append(None)
                    cs_meta_per_scene.append(
                        getattr(ms_bundles[w], 'last_cs_meta', None))
                    ms_scene.append(ms_val)   # inf
                    xms_scene.append(new_pt_ms)  # None
                    c_scene.append(c_val)
                    cpts_scene.append(c_pt)
                    _did_fresh_ms = True
                    continue  # next scenario — simplex stays alive

                if solve_state == "tightening_ms_skip_cs_failed":
                    # MS unavailable AND fallback CS failed → CS-side kill
                    if cache_on and (ms_cache is not None):
                        ms_cache[cache_key] = (solve_state,
                                               float('inf'), None, float('-inf'), None)
                    _simplex_tightening_infeasible = True
                    _tighten_infeas_scene = w
                    print(f"[eval_all] T{k} scen {w}: {solve_state} "
                          f"→ CS unavailable, simplex will be killed")
                    break  # stop processing scenarios for this simplex

                # Capture solve metadata from the bundle
                _ms_meta = getattr(ms_bundles[w], 'last_solve_meta', None)
                _cs_meta = getattr(ms_bundles[w], 'last_cs_meta', None)
                ms_meta_per_scene.append(_ms_meta)
                cs_meta_per_scene.append(_cs_meta)

                # --- Buffer non-optimal / time-limit events for end-of-iter write ---
                if _nonopt_buf is not None:
                    for _tag, _meta in [("ms", _ms_meta), ("cs", _cs_meta)]:
                        if _meta is None:
                            continue
                        _st = _meta.get("status", "")
                        if _st == "time_limit" or _st != "optimal":
                            _nonopt_buf.append(
                                f"  iter={iter_num} simp=T{k} scen={w} type={_tag} "
                                f"status={_st} "
                                f"term={_meta.get('termination_condition','?')} "
                                f"ok={_meta.get('ok','?')} "
                                f"val={_meta.get('dual_bound','?')} "
                                f"time={_meta.get('time_sec',0.0):.4f}s\n"
                            )

                # --- Buffer debug events for simplex_debug.txt ---
                if _debug_buf is not None:
                    for _tag, _meta, _val in [("ms", _ms_meta, ms_val), ("cs", _cs_meta, c_val)]:
                        if _meta is None:
                            continue
                        _st = _meta.get("status", "")
                        _ok = _meta.get("ok", True)
                        _tm = _meta.get("termination_condition", "")
                        _dual = _meta.get("dual_bound")
                        _dual_ok = (_dual is not None
                                    and isinstance(_dual, (int, float))
                                    and math.isfinite(_dual))
                        # For CS: failure = dual bound not obtainable
                        # (time-limit with valid dual bound is NOT a failure)
                        # For MS: keep broader check (not ok, or bad status/term)
                        if _tag == "cs":
                            _should_log = not _dual_ok
                            _issue = "no_dual_bound"
                        else:
                            _is_bad = (not _ok
                                       or _st in ("time_limit", "error", "infeasible", "unbounded")
                                       or _tm not in ("optimal", "locallyOptimal", ""))
                            _should_log = _is_bad
                            _issue = "solve_fail"
                        if _should_log:
                            _debug_buf.append(
                                f"  simp=T{k} scen={w} type={_tag} issue={_issue} "
                                f"ok={_ok} status={_st} term={_tm} "
                                f"val={_val} dual={_dual} "
                                f"time={_meta.get('time_sec',0.0):.4f}s\n"
                            )

                if cache_on and (ms_cache is not None):
                    _ms_ok = (ms_val is not None and math.isfinite(ms_val))
                    _cs_ok = (c_val is not None and math.isfinite(c_val))
                    if _ms_ok and _cs_ok:    _cst = "full"
                    elif _ms_ok:             _cst = "ms_only"
                    elif _cs_ok:             _cst = "ms_failed"
                    else:                    _cst = "both_failed"
                    ms_cache[cache_key] = (_cst, ms_val, new_pt_ms, c_val, c_pt)

                _did_fresh_ms = True  # at least one scenario needed fresh solve

            ms_scene.append(ms_val)
            xms_scene.append(new_pt_ms)
            c_scene.append(c_val)
            cpts_scene.append(c_pt)
        # ============================================

        # ---- If tightening + fallback CS both failed → CS-side kill ----
        # This fires only for tightening_ms_skip_cs_failed (CS unavailable).
        # Pure MS-only failure does NOT reach here.
        if _simplex_tightening_infeasible:
            per_tet.append({
                "simplex_index": k,
                "vert_idx": idxs,
                "verts": verts,
                "fverts_sum": fverts_sum,
                "ms_per_scene": [float('inf')] * S,
                "xms_per_scene": [None] * S,
                "c_per_scene": [float('-inf')] * S,
                "c_point_per_scene": [None] * S,
                "ms": float('inf'),
                "c_agg": float('-inf'),
                "LB": float('inf'),
                "UB": float('inf'),
                "x_ms_best_scene": None,
                "best_scene": None,
                "volume": vol,
                "n_infeas_verts": n_verts,
                "ms_meta_per_scene": [None] * S,
                "cs_meta_per_scene": [None] * S,
                "LB_terms": {"lb_case": "tightening_cs_failed"},
                "cs_status": "none",
                "ms_status": "none",
                "c_fallback_per_scene": [float('-inf')] * S,
                "c_fallback_source_per_scene": ["none"] * S,
                # ---- INACTIVE fields ----
                "inactive": True,
                "inactive_reason": f"tightening_ms_skip_cs_failed_scene_{_tighten_infeas_scene}",
            })
            continue

        # Simplex-level tracker: one call per simplex if any scenario did fresh MS
        if _did_fresh_ms and tracker is not None:
            tracker.note_ms_recomputed(simplex_id)


        if MS_AGG == "sum":
            ms_total = float(np.sum(ms_scene))
            _finite_c_vals = [c for c in c_scene if math.isfinite(c)]
            c_total = float(sum(_finite_c_vals)) if _finite_c_vals else float('-inf')
        elif MS_AGG == "mean":
            ms_total = float(np.mean(ms_scene))
            _finite_c_vals = [c for c in c_scene if math.isfinite(c)]
            c_total = float(np.mean(_finite_c_vals)) if _finite_c_vals else float('-inf')
        else:
            raise ValueError("MS_AGG must be 'sum' or 'mean'")

        UB = float(np.max(fverts_sum) + ms_total)

        # solve true surrogate LB (with cache) 
        # LB surrogate cache: check cache first to avoid redundant Gurobi solves
        _lb_terms_out = {}  # capture LB decomposition terms
        if lb_sur_cache is not None and key_base in lb_sur_cache:
            LB_sur = lb_sur_cache[key_base]
        else:
            LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene,
                                                 terms_out=_lb_terms_out)
            # LB surrogate cache: store result for future iterations
            if lb_sur_cache is not None:
                lb_sur_cache[key_base] = LB_sur

        # best_scene: pick argmin of finite MS values; None if all +inf
        _finite_ms_indices = [w for w in range(S) if math.isfinite(ms_scene[w])]
        if _finite_ms_indices:
            best_scene = min(_finite_ms_indices, key=lambda w: ms_scene[w])
            x_ms_best = xms_scene[best_scene]
        else:
            best_scene = None
            x_ms_best = None

        #  count infeasible vertices (Q >= 1e5 in any scenario) 
        n_infeas_verts = 0
        for j in range(n_verts):
            is_infeas = False
            for s in range(S):
                if fverts_per_scene[s][j] >= 1e5 - 1e-9:
                    is_infeas = True
                    break
            if is_infeas:
                n_infeas_verts += 1

        # record LB construction components
        min_f = float(np.min(fverts_sum))
        lb_linear = min_f + ms_total

        c_arr = np.asarray(c_scene, float)
        finite_c_mask = np.isfinite(c_arr)
        c_total_all = float(np.sum(c_arr))
        c_total_finite = float(np.sum(c_arr[finite_c_mask])) if np.any(finite_c_mask) else float('nan')

        ms_arr = np.asarray(ms_scene, float)
        all_ms_finite = bool(np.all(np.isfinite(ms_arr)))
        any_finite_c  = bool(np.any(finite_c_mask))

        if all_ms_finite and any_finite_c:
            lb_case = "all_ms_finite"
        elif all_ms_finite and not any_finite_c:
            lb_case = "all_ms_finite_no_c"
        elif not all_ms_finite and any_finite_c:
            lb_case = "some_ms_inf_use_c_only"
        else:
            lb_case = "all_fail"

        lb_terms = {
            "min_fverts_sum": min_f,
            "LB_linear": lb_linear,
            "ms_total": ms_total,
            "c_total_finite": c_total_finite,
            "c_total_all": c_total_all,
            "lb_case": lb_case,
        }
        if _lb_terms_out:
            lb_terms["fallback_LB"] = _lb_terms_out.get("fallback_LB")
            lb_terms["LB_sur_raw"] = _lb_terms_out.get("LB_sur_raw")
            lb_terms["LB_sur_final"] = _lb_terms_out.get("LB_sur_final")
            lb_terms["lb_path"] = _lb_terms_out.get("lb_path")

        _n_ms_ok = sum(1 for m in ms_scene if m is not None and math.isfinite(m))
        _n_cs_ok = sum(1 for c in c_scene if c is not None and math.isfinite(c))
        _ms_status = "full" if _n_ms_ok == S else ("partial" if _n_ms_ok > 0 else "none")
        _cs_status = "full" if _n_cs_ok == S else ("partial" if _n_cs_ok > 0 else "none")

        per_tet.append({
            "simplex_index": k,
            "vert_idx": idxs,
            "verts": verts,
            "fverts_sum": fverts_sum,
            "ms_per_scene": ms_scene,
            "xms_per_scene": xms_scene,
            "c_per_scene":  c_scene,
            "c_point_per_scene": cpts_scene,   
            "ms": ms_total,
            "c_agg": c_total,
            "LB": LB_sur,
            "UB": UB,
            "x_ms_best_scene": x_ms_best,
            "best_scene": best_scene,
            "volume": vol,
            "n_infeas_verts": n_infeas_verts,
            "ms_meta_per_scene": ms_meta_per_scene,
            "cs_meta_per_scene": cs_meta_per_scene,
            "LB_terms": lb_terms,
            "cs_status": _cs_status,
            "ms_status": _ms_status,
            "c_fallback_per_scene": list(c_scene),
            "c_fallback_source_per_scene": ["self" if (c_scene[s] is not None and math.isfinite(c_scene[s])) else "none" for s in range(S)],
            # ---- INACTIVE fields (normal records are active) ----
            "inactive": False,
            "inactive_reason": None,
        })


    return tri, per_tet

# ------------------------- MAIN LOOP -------------------------
def run_pid_simplex_3d(base_bundles, ms_bundles, model_list, first_vars_list,
                       target_nodes=30, min_dist=MIN_DIST, active_tol=ACTIVE_TOL, verbose=True,
                       agg_bundle=None, gap_stop_tol=GAP_STOP_TOL, tracker: SimplexTracker | None = None,
                       enable_3d_plot: Optional[bool] = None,
                       plot_every: int | None = None,
                       use_exact_opt: bool = False,
                       exact_solver_name: str = "gurobi",
                       exact_solver_opts: dict | None = None,
                       time_limit: float | None = None,
                       enable_ef_ub: bool = True,
                       ef_time_ub: float = 60.0,
                       initial_nodes: list | None = None,
                       output_csv_path: str | None = None,
                       split_mode: int = 1,
                       plot_output_dir: str | None = None,
                       axis_labels: tuple | None = None,
                       use_fbbt: bool = False,
                       use_obbt: bool = False,
                       obbt_solver_name: str = "gurobi",
                       obbt_solver_opts: dict | None = None,
                       obbt_tol: float = 1e-1,
                       max_obbt_rounds: int = 3):
    """
    Starting from the 8 corner nodes, in each iteration:
        - Compute global UB from current nodes (sum over scenarios)
        - Evaluate all simplex(tetrahedra) by evaluate_all_tetra
        - Identify active simplices near the current UB.
        - Determine global LB = UB + ms_b (from best active simplex).
        - Select a new candidate node minimizing ms, subject to min_dist.
        - Update scenario evaluations, nodes, and gap.
        - Stop when UB - LB ≤ gap_stop_tol or candidate collision occurs.

    * if you see verbose, ignore it, it just prints more things...

    Parameters
    ----------
    base_bundles : list[BaseBundle]
        Scenario-specific models for true Q evaluation.
    ms_bundles : list[MSBundle]
        Scenario-specific persistent solvers for ms subproblems.
    model_list : list[pyo.ConcreteModel]
        Original Pyomo models (one per scenario).
    first_vars_list : list[list[pyo.Var]]
        First-stage variable lists for each scenario.
    target_nodes : int, 
        Maximum number of nodes to generate.
    min_dist : float, 
        Minimum allowed distance between nodes.
    active_tol : float, 
        Relaxation tolerance for active simplex filtering.
    verbose : bool, default=True
        Whether to print iteration details.
    agg_bundle : 
        Reserved for aggregated ms solving.
    gap_stop_tol : float, 
        Convergence threshold for optimal rel-gap.
    tracker : SimplexTracker, 
        Tracks created/active simplices and ms recomputations.
    enable_3d_plot : Optional[bool], default=None
        Master switch for all 3D plotting. None = auto (True only when d==3).
    plot_every : int | None, optional
        Draw the 3D plot every n iterations (only used if enable_3d_plot=True).

    Returns
    -------
    dict
        History and results including nodes, LB/UB/ms traces,
        added nodes, and active simplex ratios.

    Notes
    -----
    split_mode:
        1 = Standard Mode 1: start from corner nodes, add one ms point per iteration.
        2 = Custom initial points: start from user-supplied initial_nodes
            (which should include all desired kink/inflection points),
            Delaunay-tessellate them, then run Mode 1 splitting.
    """

    iter_q_times_detail = []
    per_iter_q_counts = []
    iter_ms_times_detail = []   # [iter][scene] -> list of dt
    per_iter_ms_counts   = []   # [iter] -> int, total ms call count this round


    if tracker is None:
        tracker = SimplexTracker()
    global LAST_DEBUG

    LB_hist, UB_hist, ms_hist, node_count = [], [], [], []
    UB_node_hist, add_node_hist = [], []
    ms_a_hist, ms_b_hist = [], []
    active_ratio_hist = []
    split_kind_hist = []
    selection_reason_hist = [] # NEW: history of selection reasons

    # NEW: per-iteration summary info
    iter_time_hist = []          # Cumulative time (seconds)
    simplex_hist = []            # Total simplices per round
    active_simplex_hist = []     # Active simplices per round
    t_start = perf_counter()  # NEW: Total start time

    ms_ub_active_per_iter = []
    c_hist_per_iter = []

    # NEW: give c for the lb-simplex
    lb_c_agg_hist = []        # Scalars (e.g., sum_s c_{T,s}, only finite ones are considered)
    lb_c_per_scene_hist = []  # Each round generates a list containing the c_per_scene of that singularity.

    # timing info
    timing = {
        "init_Q_time": 0.0,
        "iter_total_time": [],
        "iter_ms_time": [],
        "iter_Q_new_time": [],
        "iter_ms_time_per_scene": [],
        "iter_ms_calls_per_scene": [],
    }

    # === NEW: Monotonic bound tracking ===
    # best_lb_ever: Track the best (highest) LB ever seen - ensures LB is non-decreasing
    best_lb_ever = float('-inf')
    # ub_candidate_library: Store (point, ub_value) tuples from c_s candidates that improved UB
    # This ensures we never "forget" a good UB candidate from previous rounds
    ub_candidate_library = []  # list of (point_tuple, ub_value)
    best_ub_ever = float('inf')  # Track best UB ever seen
    
    S = len(model_list)

    # === TIMING INSTRUMENTATION: pre-loop phases ===
    _phase_times_preloop = {}

    # === Generate initial simplex vertices ===
    _t_phase = perf_counter()
    if initial_nodes is not None:
        nodes = [tuple(float(c) for c in pt) for pt in initial_nodes]
        if verbose:
            print(f"[Init] Using {len(nodes)} user-supplied initial nodes")
            if split_mode == 2:
                print(f"[Init] split_mode=2: custom initial point set → Delaunay tessellation → Mode 1 splitting")
    else:
        nodes = corners_from_var_bounds(first_vars_list[0])

    # === Dimension auto-detection and enable_3d_plot gating ===
    d = len(first_vars_list[0])
    if enable_3d_plot is None:
        enable_3d_plot = (d == 3)
    if d != 3:
        enable_3d_plot = False
    if axis_labels is None:
        axis_labels = tuple(f"x{i+1}" for i in range(d))

    # === Preload the exact optimal solution for plotting===
    true_opt_points = None
    if enable_3d_plot:
        try:
            csv_path = "exact_opt_precomputed.csv"
            if os.path.exists(csv_path):
                rows = []
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            s_idx = int(float(row.get("scenario_index", "0")))
                        except Exception:
                            continue
                        # Only plot the scenarios currently in use 
                        
                        if s_idx < S:
                            try:
                                kp = float(row["Kp"])
                                ki = float(row["Ki"])
                                kd = float(row["Kd"])
                            except Exception:
                                continue
                            rows.append((kp, ki, kd))
                if rows:
                    true_opt_points = np.asarray(rows, float)
                    if verbose:
                        print(f"[Precompute] Loaded {len(rows)} exact optima from {csv_path} for plotting.")
            else:
                if verbose:
                    print(f"[Precompute] File {csv_path} not found; true optima will not be plotted.")
        except Exception as e:
            if verbose:
                print(f"[Precompute] Failed to load exact optima for plotting: {e}")
            true_opt_points = None


    # === NEW: build initial tetrahedral mesh once using Delaunay ===
    tet_mesh = SimplexMesh(nodes)
    _phase_times_preloop["corners_and_mesh"] = perf_counter() - _t_phase

    # All first-stage variables must have finite bounds; the simplex method cannot
    # operate over an unbounded domain.
    _vars0 = first_vars_list[0]
    _unbounded = [v.name for v in _vars0 if v.lb is None or v.ub is None]
    if _unbounded:
        raise ValueError(
            f"run_simplex: first-stage variable(s) have no finite bound and cannot be used — "
            f"please add explicit lb/ub for: {_unbounded}"
        )
    bounds_arr = np.array([[float(v.lb), float(v.ub)] for v in _vars0], float)
    diam = float(np.linalg.norm(bounds_arr[:,1] - bounds_arr[:,0]))  # diameter of the first-stage domain (for shape quality)
    min_dist = float(min_dist)

    # cache f_ω(node_i)
    _t_phase = perf_counter()
    scen_values = [[None]*len(nodes) for _ in range(S)]
    t_init_q0 = perf_counter()
    for i, node in enumerate(nodes):
        for ω in range(S):
            scen_values[ω][i] = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], node)
    timing["init_Q_time"] = perf_counter() - t_init_q0
    _phase_times_preloop["vertex_Q_evals"] = perf_counter() - _t_phase
    if verbose:
        print(f"[TIMING] Pre-loop vertex Q evals: {_phase_times_preloop['vertex_Q_evals']:.2f}s  ({len(nodes)} nodes x {S} scenarios = {len(nodes)*S} solves)")

    # ==== NEW: exact optima for validation / plotting ====
    _t_phase = perf_counter()
    exact_points_per_scen = None
    exact_point_agg = None
    if use_exact_opt:
        print("[ExactOpt] Solving per-scenario exact optima (no aggregate model) ...")
        exact_info = compute_exact_optima(
            model_list,
            first_vars_list,
            solver_name=exact_solver_name,
            solver_opts=exact_solver_opts,
        )
        exact_points_per_scen = [rec["K"] for rec in exact_info["per_scenario"]]
        exact_point_agg = exact_info["aggregate"]["K"]  # will be None
        print("[ExactOpt] Done.")
    _phase_times_preloop["exact_optima"] = perf_counter() - _t_phase



    it = 0
    stop_due_to_collision = False
    ms_cache = {}   # (scene_idx, sorted(vert_idx)) -> (ms_val, new_pt_ms, c_val, c_pt)
    lb_sur_cache = {}  # LB surrogate cache: sorted(vert_idx) -> LB_sur (float)

    # === Helper to nudge candidate point slightly into simplex interior (if necessary) ===
    # Delegates to the dimension-general snap_to_feature from simplex_geometry.
    def _snap_feature(cand_pt, rec,
                    tol_vertex=1e-3,
                    tol_edge=1e-3,
                    tol_face=1e-3):
        if rec is None:
            return tuple(map(float, cand_pt)), "interior", None

        verts = np.asarray(rec["verts"], float)       # (d+1, d)
        vert_idx = list(rec["vert_idx"])
        return snap_to_feature(cand_pt, verts, vert_idx,
                               tol_vertex=tol_vertex,
                               tol_edge=tol_edge,
                               tol_face=tol_face)


    t_start = perf_counter()   # NEW: Total start time
    cum_time = 0.0             # NEW: Cumulative time

    # === Create CSV file for incremental logging ===
    # LB/UB per user definition: main columns use per-iteration values, extra columns for monotonic envelopes
    csv_path = output_csv_path if output_csv_path else "simplex_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "# Nodes", "LB", "UB", "Rel. Gap", "Abs. Gap", "best_LB_ever", "best_UB_ever", "LB_in_split", "UB_in_split", "UB_status", "cs_status_choosen", "cs_status_all", "cs_fallback"])

    # === Determine output directory from csv_path (used for all logs) ===
    _csv_output_dir = os.path.dirname(csv_path) if csv_path else "."
    if _csv_output_dir:
        os.makedirs(_csv_output_dir, exist_ok=True)
    else:
        _csv_output_dir = "."

    # === Create debug log folder and initialize log files (all overwritten on each run) ===
    debug_log_dir = os.path.join(_csv_output_dir, "simplex_method_debug_log")
    os.makedirs(debug_log_dir, exist_ok=True)

    # File 1: Gurobi TimeLimit subproblems
    dbg_timelimit_path = os.path.join(debug_log_dir, "debug_timelimit.txt")
    with open(dbg_timelimit_path, "w", encoding="utf-8") as f:
        f.write("# debug_timelimit.txt — subproblems that hit Gurobi TimeLimit\n")
        f.write("# Format: [Iter N] [MS/CS TimeLimit] simplex_idx=..., verts=..., scenario=..., dual_bound=..., primal_obj=..., dual<primal=...\n\n")

    # File: Candidate debug log
    dbg_candidates_path = os.path.join(debug_log_dir, "debug_candidates.txt")
    with open(dbg_candidates_path, "w", encoding="utf-8") as f:
        f.write("# debug_candidates.txt — per-iteration ms/c_s candidate details\n")
        f.write("# Records: simplex vertices, ms points, c_s points, distances to existing nodes\n\n")

    # File 2: Dual > Primal violations
    dbg_dual_gt_obj_path = os.path.join(debug_log_dir, "debug_dual_gt_obj.txt")
    with open(dbg_dual_gt_obj_path, "w", encoding="utf-8") as f:
        f.write("# debug_dual_gt_obj.txt — subproblems where dual_bound > primal_obj\n")
        f.write("# Format: [Iter N] [MS/CS] simplex_idx=..., verts=..., scenario=..., dual_bound=..., primal_obj=..., gap=...\n\n")

    # File 3: EF (ipopt) solver details
    dbg_ef_solver_path = os.path.join(debug_log_dir, "debug_ef_solver.txt")
    with open(dbg_ef_solver_path, "w", encoding="utf-8") as f:
        f.write("# debug_ef_solver.txt — EF solver details per iteration (dual: ipopt + gurobi)\n")
        f.write("# Each iteration has per-solver lines + winner line\n\n")

    # File 4: LB after split (parent LB, child LBs, INSIDE/OUTSIDE)
    dbg_lb_split_path = os.path.join(debug_log_dir, "debug_lb_after_split.txt")
    with open(dbg_lb_split_path, "w", encoding="utf-8") as f:
        f.write("# debug_lb_after_split.txt — LB analysis after simplex split\n")
        f.write("# Shows: selected simplex LB, all child LBs, new global LB, INSIDE/OUTSIDE\n\n")

    # File 5: CS (Q-value) solve timing
    dbg_cs_timing_path = os.path.join(debug_log_dir, "debug_cs_timing.txt")
    with open(dbg_cs_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_cs_timing.txt — NON-OPTIMAL c_s (Q-value) solves only\n")
        f.write("# Format: [Iter N] scenario=..., simplex_idx=..., verts=..., time=...s, status=..., dual_bound=..., primal_obj=...\n\n")

    # File 6: MS (ms_on_tetra) solve timing — non-optimal only
    dbg_ms_timing_path = os.path.join(debug_log_dir, "debug_ms_timing.txt")
    with open(dbg_ms_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_ms_timing.txt — NON-OPTIMAL MS (ms_on_tetra) solves only\n")
        f.write("# Format: [Iter N] scenario=..., simplex_idx=..., verts=..., time=...s, status=..., dual_bound=..., primal_obj=...\n\n")

    # File 7: Q-value (evaluate_Q_at / BaseBundle.eval_at) solve timing
    dbg_q_timing_path = os.path.join(debug_log_dir, "debug_q_timing.txt")
    with open(dbg_q_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_q_timing.txt — NON-OPTIMAL Q-value (evaluate_Q_at) solves only\n")
        f.write("# Format: [Iter N] scenario=..., point=..., time=...s, status=..., term=..., obj=...\n\n")

    # File 8: Iter-0 summary (all simplices overview for first round)
    dbg_iter0_summary_path = os.path.join(debug_log_dir, "debug_lIter0_summary.txt")
    with open(dbg_iter0_summary_path, "w", encoding="utf-8") as f:
        f.write("# debug_lIter0_summary.txt — Iteration 0 summary\n")
        f.write("# Per-simplex: LB, avg dual bound, avg c_s Q-value, IPOPT result\n")
        f.write("# Plus: current UB, UB source, next chosen simplex\n\n")

    # File: Per-simplex evaluation status per iteration (lazy tracking)
    dbg_simplex_status_path = os.path.join(debug_log_dir, "debug_simplex_status.txt")
    with open(dbg_simplex_status_path, "w", encoding="utf-8") as f:
        f.write("# debug_simplex_status.txt — per-simplex evaluation status per iteration\n")
        f.write("# Shows: tet_key, LB/S, ms_status, cs_status, is_inherited_placeholder\n")
        f.write("# Counts: total, inherited, materialized(full), newly_materialized\n\n")

    # File 9a: Per-simplex LB decomposition when LB > UB
    dbg_lb_gt_ub_path = os.path.join(debug_log_dir, "debug_lb_gt_ub.txt")
    with open(dbg_lb_gt_ub_path, "w", encoding="utf-8") as f:
        f.write("# debug_lb_gt_ub.txt — per-simplex LB decomposition for simplices where LB > UB\n")
        f.write("# Written after post-split evaluate_all_tetra passes\n\n")

    # File 9b: All-simplex dual > primal violations (not just selected simplex)
    dbg_dual_gt_obj_all_path = os.path.join(debug_log_dir, "debug_dual_gt_obj_all.txt")
    with open(dbg_dual_gt_obj_all_path, "w", encoding="utf-8") as f:
        f.write("# debug_dual_gt_obj_all.txt — dual > primal violations for ALL simplices and scenarios\n")
        f.write("# Scans every simplex after full evaluation, not just the selected one\n\n")

    # File 10a: UB mesh membership diagnostic
    dbg_ub_mesh_path = os.path.join(debug_log_dir, "debug_ub_mesh_membership.txt")
    with open(dbg_ub_mesh_path, "w", encoding="utf-8") as f:
        f.write("# debug_ub_mesh_membership.txt — UB point mesh containment check\n")
        f.write("# Tests whether UB_node lies inside any simplex of the current mesh\n")
        f.write("# 'inside' = barycentric coords all >= -tol, all <= 1+tol\n\n")

    # File 9: Per-iteration split record (in CSV output folder for easy access)

    # simplex_debug.txt — diagnostic log for ms/cs/Q failures
    debug_path = os.path.join(_csv_output_dir, "simplex_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("# simplex_debug.txt — ms/cs/Q subproblem failures\n")
        f.write("# Fields: simp, scen, type, issue, ok, status, term, val, dual, time\n\n")
    split_log_path = os.path.join(_csv_output_dir, "simplex_record_split.txt")
    with open(split_log_path, "w", encoding="utf-8") as f:
        f.write("# simplex_record_split.txt — per-iteration split record\n")
        f.write("# selected_simplex LB_pre, UB_pre, children LBs, end LB/UB\n\n")

    # File 10: Per-iteration subproblem runtime record (in CSV output folder)
    runtime_log_path = os.path.join(_csv_output_dir, "simplex_record_subproblem_runtime.txt")
    with open(runtime_log_path, "w", encoding="utf-8") as f:
        f.write("# simplex_record_subproblem_runtime.txt — per-iteration subproblem runtime\n")
        f.write("# ms/cs/Q timing, non-optimal subproblem detection\n\n")

    # File 11: Per-iteration ms/cs detail log (per-scenario ms/cs values, points, barycentric)
    ms_cs_detail_path = os.path.join(_csv_output_dir, "simplex_record_ms_cs_detail.txt")
    with open(ms_cs_detail_path, "w", encoding="utf-8") as f:
        f.write("# simplex_record_ms_cs_detail.txt — per-scenario ms/cs values and points\n")
        f.write("# For the selected (LB) simplex each iteration\n")
        f.write("# ms_val, ms_pt (coordinates), ms_bary (barycentric)\n")
        f.write("# cs_val, cs_pt (coordinates), cs_bary (barycentric)\n\n")

    # === Initialize per-iteration diagnostic logger ===
    iter_logger = IterationLogger(path="simplex_result.txt")
    # UB provenance tracking state (note: logger also stores this, but we track here for clarity)
    ub_updated_this_iter = False
    ub_source_current = "init"           # FIX-A: start as 'init', not 'unknown'
    ub_simplex_id_current = None
    ub_origin_iter_current = -1           # FIX-A: consistent with origin_iter=-1 below
    
    # LB/UB per user definition: incumbent UB (not reset per iteration)
    UB_incumbent = float("inf")
    UB_node_incumbent = None              # FIX-A: persistent incumbent point
    
    # LB/UB per user definition: initialize selected_simplex_id for logging
    selected_simplex_id = None

    # === EF Upper Bounder initialization ===
    _t_phase = perf_counter()
    # EXPERIMENTAL: dual-solver EF — ipopt (local) + gurobi (global)
    ef_ub_ipopt = None
    ef_ub_gurobi = None
    if enable_ef_ub:
        # --- ipopt instance (original) ---
        try:
            ef_ub_ipopt = SimplexEFUb(
                model_list, first_vars_list,
                time_ub=ef_time_ub,
                solver_name="ipopt",
            )
            if verbose:
                print(f"[EF-UB] Initialized ipopt EF with {S} scenarios, time_ub={ef_time_ub}s")
        except Exception as e:
            print(f"[EF-UB] WARNING: Failed to build ipopt EF model: {e}")
            ef_ub_ipopt = None
        # --- gurobi instance (experimental, controlled by ENABLE_EF_GUROBI) ---
        if ENABLE_EF_GUROBI:
            try:
                ef_ub_gurobi = SimplexEFUb(
                    model_list, first_vars_list,
                    time_ub=ef_time_ub,
                    solver_name="gurobi",
                )
                if verbose:
                    print(f"[EF-UB] Initialized gurobi EF with {S} scenarios, time_ub={ef_time_ub}s")
            except Exception as e:
                print(f"[EF-UB] WARNING: Failed to build gurobi EF model: {e}")
                ef_ub_gurobi = None
        if ef_ub_ipopt is None and ef_ub_gurobi is None:
            enable_ef_ub = False
    _phase_times_preloop["ef_init"] = perf_counter() - _t_phase

    if verbose:
        print("\n" + "="*70)
        print("[TIMING] === Pre-loop phase summary ===")
        for _pname, _ptime in _phase_times_preloop.items():
            print(f"  {_pname:30s} : {_ptime:10.2f}s")
        print(f"  {'TOTAL':30s} : {sum(_phase_times_preloop.values()):10.2f}s")
        print("="*70 + "\n")

    # LB/UB per user definition: compute initial UB from mesh vertices BEFORE the loop
    # This initializes UB_incumbent from the initial vertex evaluations
    if len(nodes) >= d + 1:
        f_sum_initial = [
            sum(scen_values[s][i] for s in range(S))
            for i in range(len(nodes))
        ]
        _init_ub_idx = int(np.argmin(f_sum_initial))
        UB_incumbent = float(f_sum_initial[_init_ub_idx])
        UB_node_incumbent = tuple(nodes[_init_ub_idx])  # FIX-A: persist the incumbent point
        # Provenance logging: initial UB from vertices before loop
        iter_logger.update_ub_provenance(updated=True, source="init", simplex_id=None, origin_iter=-1)


    # === TIMING INSTRUMENTATION: per-iteration phase dict ===
    _iter_phase_times = []  # list of dicts, one per iteration

    # === Strict 3-stage iteration: state carried from iter 0 to iter 1 ===
    _deferred_first_split = False   # True during iter 1 when iter 0 was box-CS only
    _saved_box_c_pts = None         # c_s solution points from iter-0 box-CS
    _saved_box_c_vals = None        # c_s values from iter-0 box-CS
    _saved_box_LB = None            # box-level LB from iter-0

    termination_reason = "max_nodes"  # default if loop ends naturally

    while len(nodes) < target_nodes:
        t_iter0 = perf_counter()
        _phases = {}  # timing for this iteration
        tracker.start_iter(it)

        # initialize timing slots for this iteration
        timing["iter_total_time"].append(0.0)
        timing["iter_ms_time"].append(0.0)
        timing["iter_Q_new_time"].append(0.0)
        timing_idx = len(timing["iter_total_time"]) - 1

        ms_prev_calls = [len(b.solve_time_hist) for b in ms_bundles]

        # === Reset iteration-specific tracking ===
        ub_updated_this_iter = False
        ub_source_this_iter = None  # will track source if UB updates
        ub_simplex_id_this_iter = None
        
        # LB/UB per user definition: track simplex count before split
        n_simplices_before = len(tet_mesh.tets)

        # 1) Global UB — FIX-A: derive UB_node from incumbent, not from argmin vertex
        _t_phase = perf_counter()
        f_sum_per_node = [
            sum(scen_values[s][i] for s in range(S))
            for i in range(len(nodes))
        ]
        ub_idx = int(np.argmin(f_sum_per_node))
        UB_vertex = float(f_sum_per_node[ub_idx])
        # FIX-A: Only update incumbent if the vertex UB is genuinely better
        if UB_vertex < UB_incumbent:
            UB_incumbent = UB_vertex
            UB_node_incumbent = tuple(nodes[ub_idx])
            ub_source_current = "vertex"
            ub_origin_iter_current = it
            ub_simplex_id_current = None
            iter_logger.update_ub_provenance(
                updated=True, source="vertex", simplex_id=None, origin_iter=it
            )
            if verbose:
                print(f"[Iter {it}] Vertex UB improved incumbent: "
                      f"{UB_vertex/S:.9f} (per-scen) at node #{ub_idx}")
        # FIX-A: UB_global and UB_node always come from the persistent incumbent
        UB_global = UB_incumbent
        UB_node = UB_node_incumbent
        
        # Track vertex as initial UB source for this iteration (debug only)
        vertex_ub_simplex_id = None  # vertex is not tied to a specific simplex for now

        _phases["1_vertex_UB"] = perf_counter() - _t_phase

        # 2) Evaluate all tetrahedrons (single scene)
        _t_phase = perf_counter()

        # === Iter-0 MIPGap override DISABLED: use externally provided MIPGap ===
        _iter0_mipgap_overridden = False
        _iter0_orig_mipgaps = []

        print("[DBG] entering evaluate_all_tetra ...", flush=True)
        t_ms0 = perf_counter()
        _iter_nonopt_buf = []  # non-optimal events buffer for this iteration
        _iter_debug_lines = []  # debug events buffer for simplex_debug.txt
        _box_cs_iter = (it == 0 and initial_nodes is None)  # box start: box-level CS-only first round
        _box_c_vals = None   # set only during box-CS iter
        _box_c_pts = None
        _box_LB = None
        try:
            if _box_cs_iter:
                # Box-level CS-only evaluation for first iteration (box start)
                if verbose:
                    print(f"[Iter {it}] Box-level CS-only evaluation (box start): "
                          f"solving 1 CS per scenario over the full box")
                tri, per_tet, _box_c_vals, _box_c_pts, _box_LB = evaluate_iter0_box_cs_only(
                    nodes, scen_values, ms_bundles, first_vars_list,
                    ms_cache=ms_cache, cache_on=True, tracker=tracker,
                    tet_mesh=tet_mesh,
                    lb_sur_cache=lb_sur_cache,
                    dbg_timelimit_path=dbg_timelimit_path,
                    dbg_cs_timing_path=dbg_cs_timing_path,
                    dbg_ms_timing_path=dbg_ms_timing_path,
                    iter_num=it,
                    _nonopt_buf=_iter_nonopt_buf,
                    dbg_debug_path=debug_path,
                    dbg_lb_split_path=dbg_lb_split_path,
                    base_bundles=base_bundles,
                )
            else:
                # Iter >= 1: carry forward per_tet_end from previous iteration
                # (lazy evaluation — only the materialize-before-split loop
                #  will evaluate simplices on demand)
                per_tet = per_tet_end  # type: ignore[possibly-undefined]
                tri = None  # not used in lazy path
        except Exception as _e_eval:
            import traceback as _tb
            _crash_path = Path(__file__).resolve().parent / "crash_eval_tetra.txt"
            with open(_crash_path, "w", encoding="utf-8") as _f:
                _f.write(f"Crash in evaluate_all_tetra at iter {it}\n")
                _f.write(_tb.format_exc())
            print(f"[CRASH] evaluate_all_tetra failed: {_e_eval}")
            sys.stdout.flush()
            raise
        t_ms = perf_counter() - t_ms0
        timing["iter_ms_time"][timing_idx] = t_ms


        # === Restore original MIPGap after iter-0 ===
        if _iter0_mipgap_overridden:
            for b, orig_gap in zip(ms_bundles, _iter0_orig_mipgaps):
                b.gp.set_gurobi_param('MIPGap', orig_gap)
            if verbose:
                print(f"[Iter 0] MIPGap restored to {_iter0_orig_mipgaps[0] if _iter0_orig_mipgaps else 'N/A'}")

        # ---- Write per-simplex status to debug_simplex_status.txt ----
        _n_total = len(per_tet)
        _n_inherited = sum(1 for r in per_tet if r.get("is_inherited_placeholder", False))
        _n_full = sum(1 for r in per_tet if r.get("ms_status") == "full")
        _n_other = _n_total - _n_inherited - _n_full
        with open(dbg_simplex_status_path, "a", encoding="utf-8") as _f_ss:
            _f_ss.write(f"[Iter {it}] {_n_total} simplices: "
                        f"{_n_full} full, {_n_inherited} inherited, {_n_other} other\n")
            for r in sorted(per_tet, key=lambda x: x["simplex_index"]):
                _tet_key = tuple(sorted(r["vert_idx"]))
                _lb_s = r["LB"] / S if S > 0 else r["LB"]
                _ms = r.get("ms_status", "?")
                _cs = r.get("cs_status", "?")
                _inh = r.get("is_inherited_placeholder", False)
                _tag = "[inherited]" if _inh else f"[{_ms}]"
                _f_ss.write(f"  T{r['simplex_index']} {_tet_key}: "
                            f"LB/S={_lb_s:.9f}  {_tag}  ms={_ms}  cs={_cs}\n")
            _f_ss.write("\n")

        # NEW: record c_per_scene
        iter_c_records = []
        for r in per_tet:
            c_list = [float(x) if x is not None else None for x in r.get("c_per_scene", [])]
            iter_c_records.append({
                "simplex_index": int(r["simplex_index"]),
                "vert_idx": list(map(int, r["vert_idx"])),
                "c_per_scene": c_list,
            })
        c_hist_per_iter.append(iter_c_records)

        # —— new: record ms.solve() time and call times ——
        per_scene_times = []
        per_scene_calls = []
        for s, b in enumerate(ms_bundles):
            new_times = b.solve_time_hist[ms_prev_calls[s]:]
            per_scene_times.append(sum(new_times))
            per_scene_calls.append(len(new_times))
        timing["iter_ms_time_per_scene"].append(per_scene_times)
        timing["iter_ms_calls_per_scene"].append(per_scene_calls)

        # Extra record: list of "time per ms call" for each scenario this round
        iter_ms_times_detail.append([
            list(ms_bundles[s].solve_time_hist[ms_prev_calls[s]:])
            for s in range(len(ms_bundles))
        ])
        per_iter_ms_counts.append(sum(per_scene_calls))


        _phases["2_evaluate_all_tetra"] = perf_counter() - _t_phase

        # 3) active mask (Filter by UB + Shape Quality)
        #    Inactive simplices (from FBBT/OBBT) are ALWAYS excluded.
        _t_phase = perf_counter()
        q_cut = 0
        active_mask, _mask_diag = _build_active_mask(
            per_tet, UB_global, active_tol, q_cut, return_diagnostics=True)
        bad_quality_count = _mask_diag["bad_quality_count"]
        bad_shape_count   = _mask_diag["bad_shape_count"]
        _shape_checked    = _mask_diag["shape_checked"]

        # Store per-record quality values for downstream diagnostics
        for r in per_tet:
            sid = r["simplex_index"]
            r["quality"] = tet_quality(r["verts"])

        # --- Bad-shape filter diagnostics: store per-record info and log ---
        for _bsd in _mask_diag["bad_shape_details"]:
            _bsd_sid = _bsd["sid"]
            # Find the matching record and store diagnostics on it
            for r in per_tet:
                if r["simplex_index"] == _bsd_sid:
                    r["inactive_reason"] = "bad_shape_aspect"
                    r["max_edge"] = _bsd["max_e"]
                    r["min_edge"] = _bsd["min_e"]
                    r["aspect"]   = _bsd["aspect"]
                    break
            # Log per-simplex detail to debug_lb_after_split.txt
            _inact_line = (
                f"  Inactivate simplex {_bsd_sid}: vol={_bsd['vol']:.3e} < 1e-4, "
                f"max_edge={_bsd['max_e']:.3e}, min_edge={_bsd['min_e']:.3e}, "
                f"aspect={_bsd['aspect']:.3e}\n"
            )
            try:
                with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                    _fbs.write(_inact_line)
            except Exception:
                pass
            if verbose:
                print(f"[Iter {it}] {_inact_line.strip()}")

        # --- Write per-round shape-check summary to debug_lb_after_split.txt ---
        try:
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                _fbs.write(
                    f"[Iter {it}] Shape check: triggered={_shape_checked}, "
                    f"inactivated={bad_shape_count}, "
                    f"passed={_shape_checked - bad_shape_count}\n"
                )
        except Exception:
            pass

        # === Count simplices and active ones this round ===
        n_tets = len(per_tet)
        n_active = sum(1 for r in per_tet if active_mask.get(r["simplex_index"], False))
        n_tighten_inactive = sum(1 for r in per_tet if r.get("inactive", False))
        if verbose:
            print(f"[Iter {it}] #simplices = {n_tets}, #active = {n_active}, "
                  f"bad-quality cut = {bad_quality_count}, bad-shape cut = {bad_shape_count}, "
                  f"tighten_inactive = {n_tighten_inactive}")

        # === Guard: all simplices inactive → terminate cleanly ===
        if n_active == 0:
            # ---- Detailed diagnostic: WHY is n_active == 0? ----
            n_perm_inactive = sum(1 for r in per_tet if r.get("inactive", False))
            n_bad_quality = bad_quality_count
            n_bad_shape = bad_shape_count
            # Simplices that are NOT permanently inactive, NOT bad-quality,
            # NOT bad-shape, but still failed the active-gap test
            n_gap_filtered = 0
            _gap_threshold = UB_global + active_tol
            for r in per_tet:
                sid = r["simplex_index"]
                if r.get("inactive", False):
                    continue  # already counted as perm-inactive
                if r.get("inactive_reason") == "bad_shape_aspect":
                    continue  # already counted as bad-shape
                # Check if it was killed by quality filter
                q = r.get("quality", None)
                if q is not None and q < q_cut:
                    continue  # already counted as bad-quality
                # If we get here, the simplex was only excluded by gap test
                if r["LB"] > _gap_threshold:
                    n_gap_filtered += 1

            # Determine precise reason
            if n_perm_inactive == n_tets:
                termination_reason = "all_simplices_permanently_inactive"
            elif n_gap_filtered > 0 and n_perm_inactive == 0 and n_bad_quality == 0 and n_bad_shape == 0:
                termination_reason = "all_simplices_filtered_by_active_gap_test"
            elif n_gap_filtered > 0:
                termination_reason = "all_simplices_inactive_mixed_reasons"
            else:
                termination_reason = "all_simplices_inactive_after_tightening"

            _diag_lines = []
            _diag_lines.append(f"\n{'=' * 70}")
            _diag_lines.append(f"[Iter {it}] TERMINATION: {termination_reason}")
            _diag_lines.append(f"  Total simplices:          {n_tets}")
            _diag_lines.append(f"  Permanently inactive:     {n_perm_inactive}")
            _diag_lines.append(f"  Bad-quality cut:          {n_bad_quality}")
            _diag_lines.append(f"  Bad-shape cut:            {n_bad_shape}")
            _diag_lines.append(f"  Failed active-gap test:   {n_gap_filtered}")
            _diag_lines.append(f"  UB_global (sum):          {UB_global:.12f}")
            _diag_lines.append(f"  UB_global (per-scen):     {UB_global / S:.12f}")
            _diag_lines.append(f"  active_tol:               {active_tol:.6e}")
            _diag_lines.append(f"  Threshold (UB + tol):     {_gap_threshold:.12f}")
            _diag_lines.append(f"  Threshold (per-scen):     {_gap_threshold / S:.12f}")

            # Per-simplex sample (first 10 simplices sorted by LB)
            _sorted_recs = sorted(per_tet, key=lambda r: r["LB"])
            _sample_n = min(10, len(_sorted_recs))
            _diag_lines.append(f"\n  --- Per-simplex detail (first {_sample_n} by LB) ---")
            _diag_lines.append(f"  {'idx':>5s} {'inactive':>8s} {'LB/S':>16s} {'LB<=thresh':>10s} {'reason':>24s}")
            for _si, r in enumerate(_sorted_recs[:_sample_n]):
                _sid = r["simplex_index"]
                _is_inact = r.get("inactive", False)
                _lb_s = r["LB"] / S
                _passes = r["LB"] <= _gap_threshold
                _reason = "perm_inactive" if _is_inact else ""
                if not _reason and r.get("inactive_reason") == "bad_shape_aspect":
                    _reason = "bad_shape"
                if not _reason:
                    _q = r.get("quality", None)
                    if _q is not None and _q < q_cut:
                        _reason = "bad_quality"
                if not _reason:
                    _reason = "gap_fail" if not _passes else "PASS"
                _diag_lines.append(
                    f"  T{_sid:>4d} {str(_is_inact):>8s} {_lb_s:>16.9f} {str(_passes):>10s} {_reason:>24s}"
                )

            _diag_lines.append(f"{'=' * 70}\n")

            # Print to console
            for _dl in _diag_lines:
                print(_dl)
            sys.stdout.flush()

            # Write to debug log file (always, even if _iter_debug_lines is empty)
            try:
                with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                    _fbs.write(f"[Iter {it}] TERMINATED: {termination_reason}\n")
                    for _dl in _diag_lines:
                        _fbs.write(_dl + "\n")
                    _fbs.flush()
            except Exception:
                pass

            # Also write to simplex_debug.txt
            try:
                with open(debug_path, "a", encoding="utf-8") as _fdbg:
                    _fdbg.write(f"=== Iter {it} (early termination) ===\n")
                    for _dl in _diag_lines:
                        _fdbg.write(_dl + "\n")
                    if _iter_debug_lines:
                        _fdbg.writelines(_iter_debug_lines)
                    _fdbg.flush()
            except Exception:
                pass
            break
            
        # === Per-iter tables: active vs inactive simplices (LB, UB, sum ms, sum c_s) ===
        if verbose:
            def _print_lb_ub_ms_c_table(records, title):
                print(f"== [Iter {it}] {title} simplices: LB / UB / sum(ms_s) / sum(c_s) (per-scenario) ==")
                if not records:
                    print("  (none)")
                    return
                # Sort by simplex_index for easy comparison
                records = sorted(records, key=lambda rr: rr["simplex_index"])
                header = ["simp", "LB", "UB", "sum_ms", "sum_c_s", "Range(As)", "Range(As+ms)", "#InfV"]
                colw = [8, 18, 18, 18, 18, 25, 25, 6]

                def fmt_row(cols):
                    return (
                        f"{str(cols[0]).ljust(colw[0])}"
                        f"{str(cols[1]).rjust(colw[1])}"
                        f"{str(cols[2]).rjust(colw[2])}"
                        f"{str(cols[3]).rjust(colw[3])}"
                        f"{str(cols[4]).rjust(colw[4])}"
                        f"{str(cols[5]).rjust(colw[5])}"
                        f"{str(cols[6]).rjust(colw[6])}"
                        f"{str(cols[7]).rjust(colw[7])}"
                    )

                print(fmt_row(header))
                print("-" * sum(colw))

                for rr in records:
                    simp_id = f"T{int(rr['simplex_index'])}"
                    # Divide by number of scenarios
                    LB_val  = float(rr["LB"]) / S
                    UB_val  = float(rr["UB"]) / S
                    ms_val  = float(rr.get("ms", float("nan")))
                    c_agg   = float(rr.get("c_agg", float("nan")))

                    fverts_sum = rr.get("fverts_sum", [])
                    if fverts_sum:
                        min_As = min(fverts_sum)
                        max_As = max(fverts_sum)
                        rng_As = f"[{min_As:.2e}, {max_As:.2e}]"

                        min_As_ms = min_As + ms_val
                        max_As_ms = max_As + ms_val
                        rng_As_ms = f"[{min_As_ms:.2e}, {max_As_ms:.2e}]"
                    else:
                        rng_As = "N/A"
                        rng_As_ms = "N/A"
                    
                    n_inf = rr.get("n_infeas_verts", 0)

                    row = [
                        simp_id,
                        f"{LB_val:.6e}",
                        f"{UB_val:.6e}",
                        f"{ms_val:.6e}",
                        f"{c_agg:.6e}",
                        rng_As,
                        rng_As_ms,
                        n_inf,
                    ]
                    print(fmt_row(row))
                print()

            active_recs = [r for r in per_tet if active_mask.get(r["simplex_index"], False)]
            inactive_recs = [r for r in per_tet if not active_mask.get(r["simplex_index"], False)]

            _print_lb_ub_ms_c_table(active_recs,   "ACTIVE")
            _print_lb_ub_ms_c_table(inactive_recs, "INACTIVE")




        # 4) active ratio
        total_vol = sum(r["volume"] for r in per_tet)
        active_vol = sum(r["volume"] for r in per_tet if active_mask[r["simplex_index"]])
        active_ratio = active_vol / total_vol if total_vol > 0 else 0.0

        # NEW: Number of simplices and active simplices this round
        num_simplices = len(per_tet)
        num_active_simplices = sum(
            1 for r in per_tet if active_mask.get(r["simplex_index"], False)
        )

        # collect statistics of active simplices (active / active+UB)
        for r in per_tet:
            is_active = active_mask.get(r["simplex_index"], False)
            if not is_active:
                continue
            simplex_id = tuple(sorted(r["vert_idx"]))
            has_ub = (ub_idx in r["vert_idx"])
            tracker.note_active(simplex_id, has_ub=has_ub)

        # print iteration statistics immediately
        tracker.end_iter()

        _phases["3_active_filter_print"] = perf_counter() - _t_phase

        # 5) LB_global
        _t_phase = perf_counter()
        active_LBs = [r["LB"] for r in per_tet if active_mask.get(r["simplex_index"], False)]
        if active_LBs:
            LB_global = float(min(active_LBs))
            lb_simp_rec = min(
                (r for r in per_tet if active_mask[r["simplex_index"]]),
                key=lambda r: r["LB"]
            )
        else:
            _fallback_recs = [r for r in per_tet if not r.get("inactive", False)]
            if _fallback_recs:
                LB_global = float(min(r["LB"] for r in _fallback_recs))
                lb_simp_rec = min(_fallback_recs, key=lambda r: r["LB"])
            else:
                LB_global = float(min(r["LB"] for r in per_tet))
                lb_simp_rec = min(per_tet, key=lambda r: r["LB"])

        # === ITER-0 EARLY EXIT: pure box-CS only, defer split to iter 1 ===
        if _box_cs_iter:
            # Save box-level c_s info for iteration 1's simplex selection
            _saved_box_c_pts = _box_c_pts
            _saved_box_c_vals = _box_c_vals
            _saved_box_LB = _box_LB
            _deferred_first_split = True

            # ---- [DIAG-C/D] Iter-0 UB provenance + LB vs UB consistency ----
            _iter0_diag = []
            _iter0_diag.append(f"\n{'=' * 80}")
            _iter0_diag.append(f"[Iter {it}] ITER-0 LB vs UB PROVENANCE CHECK")
            _iter0_diag.append(f"{'=' * 80}")
            _iter0_diag.append(f"  box_LB (sum):        {_box_LB:.12f}")
            _iter0_diag.append(f"  box_LB (per-scen):   {_box_LB / S:.12f}")
            _iter0_diag.append(f"  LB_global (sum):     {LB_global:.12f}")
            _iter0_diag.append(f"  LB_global (per-scen):{LB_global / S:.12f}")
            _iter0_diag.append(f"")
            _iter0_diag.append(f"  UB_global (sum):     {UB_global:.12f}")
            _iter0_diag.append(f"  UB_global (per-scen):{UB_global / S:.12f}")
            _ub_src = ub_source_current if ub_source_current else "unknown"
            _iter0_diag.append(f"  UB source:           {_ub_src}")
            if UB_node_incumbent is not None:
                _iter0_diag.append(f"  UB point:            {_fmt_point(UB_node_incumbent)}")
            else:
                _iter0_diag.append(f"  UB point:            None")
            _iter0_diag.append(f"  UB origin iter:      {ub_origin_iter_current}")
            # Per-scenario UB breakdown (vertex Q-values at UB point)
            if UB_node_incumbent is not None:
                _iter0_diag.append(f"\n  --- Per-scenario UB (vertex Q-values at UB point) ---")
                _iter0_diag.append(f"  {'scen':>5s} {'Q_val':>16s}")
                _ub_q_sum = 0.0
                for _us in range(min(S, 10)):
                    _ub_node_idx = None
                    for _ni, _nd in enumerate(nodes):
                        if tuple(_nd) == UB_node_incumbent:
                            _ub_node_idx = _ni
                            break
                    if _ub_node_idx is not None:
                        _uq = scen_values[_us][_ub_node_idx]
                    else:
                        _uq = float('nan')
                    _ub_q_sum += _uq if math.isfinite(_uq) else 0.0
                    _iter0_diag.append(f"  {_us:>5d} {_uq:>16.9f}")
                if S > 10:
                    _iter0_diag.append(f"  ... ({S - 10} more scenarios)")
            _iter0_diag.append(f"")
            # Consistency check
            _lb_ub_diff = _box_LB - UB_global
            _lb_ub_diff_ps = _lb_ub_diff / S if S > 0 else _lb_ub_diff
            _iter0_diag.append(f"  --- Consistency Check ---")
            _iter0_diag.append(f"  box_LB - UB_global (sum):     {_lb_ub_diff:.12f}")
            _iter0_diag.append(f"  box_LB - UB_global (per-scen):{_lb_ub_diff_ps:.12f}")
            _small_tol = 1e-6
            if _box_LB > UB_global + _small_tol:
                _iter0_diag.append(f"  *** WARNING: iter-0 box_LB EXCEEDS current UB_global ***")
                _iter0_diag.append(f"      box_LB (sum)    = {_box_LB:.12f}")
                _iter0_diag.append(f"      UB_global (sum) = {UB_global:.12f}")
                _iter0_diag.append(f"      difference      = {_lb_ub_diff:.12f}  (per-scen: {_lb_ub_diff_ps:.12f})")
                _iter0_diag.append(f"      This means the box-CS 'lower bound' is NOT a valid LB")
                _iter0_diag.append(f"      relative to the current UB. Possible causes:")
                _iter0_diag.append(f"        - LB and UB are computed over different feasible regions")
                _iter0_diag.append(f"        - Solver non-optimality / MIPGap in box-CS dual bounds")
                _iter0_diag.append(f"        - Aggregation mismatch (probabilities / weighting)")
            else:
                _iter0_diag.append(f"  OK: box_LB <= UB_global (valid lower bound)")
            _iter0_diag.append(f"  active_tol:          {active_tol:.6e}")
            _iter0_diag.append(f"  Threshold (UB+tol):  {UB_global + active_tol:.12f}")
            _will_pass = _box_LB <= UB_global + active_tol
            _iter0_diag.append(f"  box_LB <= threshold: {_will_pass}")
            if not _will_pass:
                _iter0_diag.append(f"  => ALL simplices will fail active-gap test (inherit same box_LB)")

            # ---- [DIAG-NEW] Per-scenario box-CS vs true Q_s(cand_pt) comparison ----
            _iter0_diag.append(f"\n{'=' * 80}")
            _iter0_diag.append(f"[Iter {it}] BOX-CS vs TRUE Q COMPARISON  (evaluate_Q_at on original models)")
            _iter0_diag.append(f"{'=' * 80}")
            _iter0_diag.append(f"  {'scen':>5s} {'dual_bnd':>16s} {'primal_obj':>16s} {'selected':>16s}"
                               f" {'trueQ(cpt)':>16s} {'sel-trueQ':>12s} {'dual-trueQ':>12s}"
                               f" {'sel<=trueQ':>10s} {'dual<=trueQ':>11s} {'prim~trueQ':>10s}")
            _iter0_diag.append(f"  {'-'*5:>5s} {'-'*16:>16s} {'-'*16:>16s} {'-'*16:>16s}"
                               f" {'-'*16:>16s} {'-'*12:>12s} {'-'*12:>12s}"
                               f" {'-'*10:>10s} {'-'*11:>11s} {'-'*10:>10s}")

            _trueQ_list = []
            _sel_list = []
            _n_sel_gt_trueQ = 0
            _n_dual_gt_trueQ = 0
            _diag_tol = 1e-6

            # Retrieve box_cs_meta from the per_tet records (shared, so use first record)
            _box_meta_list = []
            if per_tet:
                _box_meta_list = per_tet[0].get("cs_meta_per_scene", [None] * S)
            else:
                _box_meta_list = [None] * S

            for _ws in range(S):
                _sel = _box_c_vals[_ws] if _box_c_vals is not None else float('nan')
                _cpt = _box_c_pts[_ws] if _box_c_pts is not None else None
                _meta = _box_meta_list[_ws] if _ws < len(_box_meta_list) else None
                _meta = _meta if _meta is not None else {}
                _dual = _meta.get('dual_bound', None)
                _prim = _meta.get('primal_obj', None)

                # Evaluate true Q_s(cand_pt) using the ORIGINAL scenario model
                _trueQ = float('nan')
                if _cpt is not None and all(math.isfinite(v) for v in _cpt):
                    try:
                        _trueQ = float(evaluate_Q_at(
                            base_bundles[_ws], first_vars_list[_ws], _cpt
                        ))
                    except Exception as _eq_e:
                        _trueQ = float('nan')
                        _iter0_diag.append(f"  # scen {_ws}: evaluate_Q_at raised: {_eq_e}")

                _trueQ_list.append(_trueQ)
                _sel_list.append(_sel)

                # Comparison deltas
                _sel_minus_tq = (_sel - _trueQ) if (math.isfinite(_sel) and math.isfinite(_trueQ)) else float('nan')
                _dual_f = _dual if _dual is not None else float('nan')
                _dual_minus_tq = (_dual_f - _trueQ) if (math.isfinite(_dual_f) and math.isfinite(_trueQ)) else float('nan')

                # Flags
                _sel_le_tq = ""
                if math.isfinite(_sel) and math.isfinite(_trueQ):
                    _sel_le_tq = str(_sel <= _trueQ + _diag_tol)
                    if _sel > _trueQ + _diag_tol:
                        _n_sel_gt_trueQ += 1
                _dual_le_tq = ""
                if math.isfinite(_dual_f) and math.isfinite(_trueQ):
                    _dual_le_tq = str(_dual_f <= _trueQ + _diag_tol)
                    if _dual_f > _trueQ + _diag_tol:
                        _n_dual_gt_trueQ += 1
                _prim_f = _prim if _prim is not None else float('nan')
                _prim_match_tq = ""
                if math.isfinite(_prim_f) and math.isfinite(_trueQ):
                    _prim_match_tq = str(abs(_prim_f - _trueQ) < _diag_tol)

                _iter0_diag.append(
                    f"  {_ws:>5d} {_dual_f:>16.9f} {_prim_f:>16.9f} {_sel:>16.9f}"
                    f" {_trueQ:>16.9f} {_sel_minus_tq:>12.6f} {_dual_minus_tq:>12.6f}"
                    f" {_sel_le_tq:>10s} {_dual_le_tq:>11s} {_prim_match_tq:>10s}"
                )

            # Aggregation summary with trueQ comparison
            _sum_sel = sum(v for v in _sel_list if math.isfinite(v))
            _sum_trueQ = sum(v for v in _trueQ_list if math.isfinite(v))
            _n_trueQ_finite = sum(1 for v in _trueQ_list if math.isfinite(v))
            _iter0_diag.append(f"\n  --- Box-CS vs True Q Aggregation Summary ---")
            _iter0_diag.append(f"  sum(selected)   = {_sum_sel:.12f}  (per-scen: {_sum_sel / S:.12f})")
            _iter0_diag.append(f"  sum(trueQ)      = {_sum_trueQ:.12f}  (per-scen: {_sum_trueQ / S:.12f})  [{_n_trueQ_finite}/{S} finite]")
            _iter0_diag.append(f"  UB_global       = {UB_global:.12f}  (per-scen: {UB_global / S:.12f})")
            _iter0_diag.append(f"  box_LB          = {_box_LB:.12f}  (per-scen: {_box_LB / S:.12f})")
            _iter0_diag.append(f"  box_LB - UB     = {_box_LB - UB_global:.12f}  (per-scen: {(_box_LB - UB_global) / S:.12f})")
            _iter0_diag.append(f"  sum(sel) - sum(trueQ) = {_sum_sel - _sum_trueQ:.12f}")
            _iter0_diag.append(f"  # scenarios where selected > trueQ (by > {_diag_tol:.0e}): {_n_sel_gt_trueQ}/{S}")
            _iter0_diag.append(f"  # scenarios where dual_bnd > trueQ (by > {_diag_tol:.0e}): {_n_dual_gt_trueQ}/{S}")
            if _n_sel_gt_trueQ > 0:
                _iter0_diag.append(f"  *** WARNING: {_n_sel_gt_trueQ} scenario(s) have selected > trueQ(cand_pt) ***")
                _iter0_diag.append(f"      The box-CS 'dual bound' is NOT a valid lower bound for those scenarios.")
                _iter0_diag.append(f"      This explains why box_LB > UB_global.")
            elif _n_trueQ_finite == S:
                _iter0_diag.append(f"  OK: all selected values <= trueQ(cand_pt) — box-CS dual bounds look valid per-scenario.")
                if _sum_sel > UB_global + _diag_tol:
                    _iter0_diag.append(f"      BUT sum(selected) > UB_global — the issue may be in aggregation, not per-scenario bounds.")
            _iter0_diag.append(f"{'=' * 80}\n")

            # Print to console
            for _dl in _iter0_diag:
                print(_dl)
            sys.stdout.flush()

            # Write to debug files
            try:
                with open(debug_path, "a", encoding="utf-8") as _fdbg:
                    _fdbg.write(f"=== Iter {it} (box-CS provenance) ===\n")
                    for _dl in _iter0_diag:
                        _fdbg.write(_dl + "\n")
                    _fdbg.flush()
            except Exception:
                pass
            try:
                with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                    for _dl in _iter0_diag:
                        _fbs.write(_dl + "\n")
                    _fbs.flush()
            except Exception:
                pass

            # Record iteration 0 history (pure box-level CS, no split)
            LB_hist.append(LB_global)
            UB_hist.append(UB_global)
            ms_hist.append(float('inf'))
            node_count.append(len(nodes))
            UB_node_hist.append(UB_node)
            ms_a_hist.append(float('inf'))
            ms_b_hist.append(float('inf'))
            active_ratio_hist.append(active_ratio)
            now = perf_counter()
            iter_time_hist.append(now - t_start)
            simplex_hist.append(len(per_tet))
            active_simplex_hist.append(num_active_simplices)
            split_kind_hist.append("none")
            lb_c_agg_hist.append(float(_box_LB) if _box_LB is not None else float('-inf'))
            lb_c_per_scene_hist.append(
                [float(v) if v is not None else None for v in _box_c_vals]
                if _box_c_vals is not None else []
            )
            add_node_hist.append(None)
            selection_reason_hist.append("none_box_cs_only")

            # No split: per_tet_end = per_tet for next iteration
            per_tet_end = per_tet

            # Update monotonic envelopes
            best_lb_ever = max(best_lb_ever, LB_global)
            best_ub_ever = min(best_ub_ever, UB_global)

            # Write CSV row for iteration 0
            iter_time = iter_time_hist[-1]
            n_nodes_csv = node_count[-1]
            lb_val_csv = LB_global / S
            ub_val_csv = UB_global / S
            abs_gap_csv = (UB_global - LB_global) / S
            rel_gap_csv = abs_gap_csv / (abs(ub_val_csv) + 1e-16)
            lb_ever_csv = best_lb_ever / S
            ub_ever_csv = best_ub_ever / S
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                _cs_chosen_iter0 = _format_cs_failure_status(_box_c_vals) if _box_c_vals is not None else "0/0"
                writer.writerow([
                    f"{iter_time:.3f}", n_nodes_csv,
                    f"{lb_val_csv:.9f}", f"{ub_val_csv:.9f}",
                    f"{rel_gap_csv*100:.7f}%", f"{abs_gap_csv:.5f}",
                    f"{lb_ever_csv:.9f}", f"{ub_ever_csv:.9f}",
                    "N/A", "N/A", "N/A",
                    _cs_chosen_iter0, "none", "none"
                ])

            # Timing
            t_iter = perf_counter() - t_iter0
            _phases["TOTAL"] = t_iter
            _iter_phase_times.append(_phases)
            timing["iter_total_time"][timing_idx] = t_iter

            if verbose:
                print(f"[Iter {it}] Box-CS only iteration complete (strict 3-stage). "
                      f"LB={LB_global/S:.9f}, UB={UB_global/S:.9f}. "
                      f"Split deferred to iter 1.")

            it += 1
            continue  # skip to iteration 1

        # === Iteration-0 box-CS override: select simplex containing averaged c_s point ===
        # This override runs AFTER the standard min-LB selection above, and
        # BEFORE ensure_ms_for_simplex() below.  This ordering is critical:
        #   1. Standard min-LB selection (fallback baseline)
        #   2. Iter-0 averaged-c_s containment override (this block)
        #   3. lb_simp_idx = int(lb_simp_rec["simplex_index"]) — always derived from lb_simp_rec
        #   4. ensure_ms_for_simplex() on the final selected simplex
        _iter0_selection_fallback = False
        if _deferred_first_split and _saved_box_c_pts is not None and per_tet:
            # Compute the average of all VALID box-level c_s points
            valid_box_pts = [
                pt for pt in _saved_box_c_pts
                if pt is not None and all(math.isfinite(v) for v in pt)
            ]
            if valid_box_pts:
                avg_cs_pt = tuple(float(x) for x in np.mean(valid_box_pts, axis=0))
                _avg_str = _fmt_point(avg_cs_pt)
                if verbose:
                    print(f"[Iter {it}] Averaged c_s point = {_avg_str} "
                          f"({len(valid_box_pts)}/{S} valid points)")

                # INVARIANT: the LB simplex must be selected from the active set.
                # Search only active simplices for containment.
                _active_per_tet_iter0 = [r for r in per_tet if active_mask.get(r["simplex_index"], False)]
                contain_idx = _find_simplex_containing_point(avg_cs_pt, _active_per_tet_iter0, tol=1e-8)
                if contain_idx >= 0:
                    lb_simp_rec = _active_per_tet_iter0[contain_idx]
                    if verbose:
                        print(f"[Iter {it}] Selected simplex T{lb_simp_rec['simplex_index']} "
                              f"(contains averaged c_s point)")
                else:
                    # Try with looser tolerance
                    contain_idx = _find_simplex_containing_point(avg_cs_pt, _active_per_tet_iter0, tol=1e-4)
                    if contain_idx >= 0:
                        lb_simp_rec = _active_per_tet_iter0[contain_idx]
                        if verbose:
                            print(f"[Iter {it}] Selected simplex T{lb_simp_rec['simplex_index']} "
                                  f"(contains averaged c_s point, loose tol)")
                    else:
                        _iter0_selection_fallback = True
                        if verbose:
                            print(f"[Iter {it}] FALLBACK: no simplex contains averaged c_s point "
                                  f"{_avg_str}, using default min-LB selection "
                                  f"(T{lb_simp_rec['simplex_index']})")
            else:
                _iter0_selection_fallback = True
                if verbose:
                    print(f"[Iter {it}] FALLBACK: no valid box-level c_s points, "
                          f"using default min-LB selection (T{lb_simp_rec['simplex_index']})")
        # LB/UB per user definition: best_lb_ever update disabled here, done AFTER end-of-iter
        # best_lb_ever = max(best_lb_ever, LB_global)  # DISABLED

        # LB/UB per user definition: DISABLED - UB candidates only from NEW simplices (end-of-iter)
        # The following avg_cs and library reuse blocks are disabled per user spec.
        # UB is now computed ONLY from new simplices after split.
        
        # # NEW: avg c_s UB candidate (DISABLED per user spec)
        # # Collect finite c_pts from lb_simp_rec, compute average, evaluate true objective
        # c_pts_list = lb_simp_rec.get("c_point_per_scene", [])
        # valid_c_pts = [
        #     pt for pt in c_pts_list
        #     if pt is not None and all(math.isfinite(v) for v in pt)
        # ]
        # if valid_c_pts:
        #     x_avg = tuple(np.mean(valid_c_pts, axis=0))
        #     # Evaluate TRUE objective at x_avg for each scenario
        #     ub_avg_vals = [
        #         evaluate_Q_at(base_bundles[s], first_vars_list[s], x_avg)
        #         for s in range(S)
        #     ]
        #     ub_avg_val = float(sum(ub_avg_vals))  # same aggregation as UB_global
        #     # Update UB if this improves it AND doesn't violate best_lb_ever
        #     if ub_avg_val < UB_global and ub_avg_val >= best_lb_ever - 1e-8:
        #         UB_global = ub_avg_val
        #         UB_node = x_avg
        #         ub_idx = None  # Mark that UB is not from a mesh node
        #         # === NEW: Add this candidate to the library so it's never forgotten ===
        #         ub_candidate_library.append((x_avg, ub_avg_val))
        #         # Track UB provenance
        #         ub_updated_this_iter = True
        #         ub_source_this_iter = "avg_cs"
        #         ub_simplex_id_this_iter = tuple(sorted(lb_simp_rec["vert_idx"]))
        #         if verbose:
        #             print(f"[Iter {it}] NEW: avg c_s UB candidate improved UB: {ub_avg_val:.6e} (added to library)")

        # # === NEW: Check all candidates in the library to ensure UB never increases === (DISABLED per user spec)
        # # IMPORTANT: Only use library entries that are still valid (>= best_lb_ever)
        # for lib_pt, lib_ub_val in ub_candidate_library:
        #     if lib_ub_val < UB_global and lib_ub_val >= best_lb_ever - 1e-8:
        #         UB_global = lib_ub_val
        #         UB_node = lib_pt
        #         ub_idx = None
        #         # Track UB provenance (library reuse)
        #         ub_updated_this_iter = True
        #         ub_source_this_iter = "library_reuse"
        #         ub_simplex_id_this_iter = None

        # LB/UB per user definition: best_ub_ever update disabled here, done AFTER end-of-iter
        # best_ub_ever = min(best_ub_ever, UB_global)  # DISABLED

        # LB/UB per user definition: UB provenance updated ONLY at end-of-iter, not here
        # (Removed pre-split provenance updates per user spec)

        # ======= Consistency check: selected simplex LB(=LB_global) should not exceed UB_global =======
        # Theoretically surrogate is underestimator, so LB_global <= UB_global should always hold
        # Add small tolerance to prevent pure numerical error
        if LB_global > UB_global + 1e-8:
            print(
                f"[WARNING iter {it}] LB exceeds UB! "
                f"simplex={lb_simp_rec['simplex_index']}, "
                f"LB={LB_global:.6e}, UB={UB_global:.6e}, "
                f"gap={LB_global - UB_global:.6e}  (continuing for diagnostics)"
            )
        # =======================================================================


        lb_simp_idx = int(lb_simp_rec["simplex_index"])

        if not _deferred_first_split:
            # ---- Materialize-before-split loop (iter >= 2, or iter >= 1 when not deferred) ----
            # Any not-fully-evaluated simplex must be materialized and then
            # global min-LB re-checked before committing to split it.
            _max_mat = len(per_tet)
            for _mat_round in range(_max_mat):
                lb_simp_idx = int(lb_simp_rec["simplex_index"])
                if lb_simp_rec.get("ms_status", "full") != "full":
                    _was_placeholder = lb_simp_rec.get("is_inherited_placeholder", False)
                    if verbose:
                        print(f"[Iter {it}] Materializing T{lb_simp_idx} "
                              f"(ms_status={lb_simp_rec.get('ms_status')}, "
                              f"placeholder={_was_placeholder}, "
                              f"inherited LB/S={lb_simp_rec['LB']/S:.9f}) ...")
                    ensure_ms_for_simplex(
                        lb_simp_rec, ms_bundles, nodes, scen_values,
                        ms_cache, lb_sur_cache, S,
                        verbose=verbose, iter_num=it,
                        use_fbbt=use_fbbt, use_obbt=use_obbt,
                        obbt_solver_name=obbt_solver_name,
                        obbt_solver_opts=obbt_solver_opts,
                        obbt_tol=obbt_tol,
                        max_obbt_rounds=max_obbt_rounds,
                        _debug_buf=_iter_debug_lines,
                    )
                    lb_simp_rec.pop("is_inherited_placeholder", None)
                    if verbose:
                        print(f"[Iter {it}] Materialized T{lb_simp_idx}: "
                              f"LB/S={lb_simp_rec['LB']/S:.9f}, "
                              f"ms_status={lb_simp_rec['ms_status']}")

                    # INVARIANT: after any in-place LB/status update, the LB
                    # simplex must be re-selected from a FRESHLY REBUILT active
                    # set, because ensure_ms_for_simplex() may have changed LB,
                    # ms_status, cs_status, or inactive fields.
                    active_mask = _build_active_mask(per_tet, UB_global, active_tol, q_cut)
                    _active_per_tet = [r for r in per_tet if active_mask.get(r["simplex_index"], False)]
                    _new_min_rec = min(_active_per_tet, key=lambda r: r["LB"]) if _active_per_tet else lb_simp_rec
                    if _new_min_rec["simplex_index"] != lb_simp_idx:
                        if verbose:
                            print(f"[Iter {it}] Min-LB shifted: "
                                  f"T{lb_simp_idx} -> T{_new_min_rec['simplex_index']} "
                                  f"(LB/S={_new_min_rec['LB']/S:.9f})")
                        lb_simp_rec = _new_min_rec
                        LB_global = float(_new_min_rec["LB"])
                        continue  # re-check this new candidate
                    else:
                        LB_global = float(lb_simp_rec["LB"])
                        break  # same simplex is still min-LB
                else:
                    break  # already fully evaluated
        else:
            # Iter 0: existing single ensure_ms call (no re-check needed,
            # because avg_cs_pt override intentionally selects the simplex)
            if lb_simp_rec.get("ms_status", "full") != "full":
                if verbose:
                    print(f"[Iter {it}] Computing MS for selected simplex T{lb_simp_idx} "
                          f"(ms_status={lb_simp_rec.get('ms_status','?')}) ...")
                ensure_ms_for_simplex(
                    lb_simp_rec, ms_bundles, nodes, scen_values,
                    ms_cache, lb_sur_cache, S, verbose=verbose, iter_num=it,
                    use_fbbt=use_fbbt, use_obbt=use_obbt,
                    obbt_solver_name=obbt_solver_name,
                    obbt_solver_opts=obbt_solver_opts,
                    obbt_tol=obbt_tol,
                    max_obbt_rounds=max_obbt_rounds,
                    _debug_buf=_iter_debug_lines,
                )
                if verbose:
                    print(f"[Iter {it}] MS computed: ms_status={lb_simp_rec['ms_status']}, "
                          f"ms={lb_simp_rec.get('ms', float('inf')):.6e}, "
                          f"LB={lb_simp_rec['LB']:.6e}")

        lb_simp_idx = int(lb_simp_rec["simplex_index"])
        ms_b      = float(lb_simp_rec.get("ms", float('inf')))
        ms_b_simp = lb_simp_idx


        # ---- Post-materialize: log which simplices are now full ----
        _n_full_after = sum(1 for r in per_tet if r.get("ms_status") == "full")
        _n_inh_after = sum(1 for r in per_tet if r.get("is_inherited_placeholder", False))
        _newly_mat = _n_full_after - _n_full  # how many were materialized this round
        with open(dbg_simplex_status_path, "a", encoding="utf-8") as _f_ss:
            _f_ss.write(f"  >> After materialize: {_n_full_after} full, "
                        f"{_n_inh_after} inherited, "
                        f"{_newly_mat} newly materialized this iter\n")
            _f_ss.write(f"  >> Selected for split: T{lb_simp_idx}\n\n")

        _phases["4_LB_selection"] = perf_counter() - _t_phase

        # ==================================================
        # === EF Upper Bounder solve on selected simplex ===
        # ==================================================
        _t_phase = perf_counter()
        # EXPERIMENTAL: dual-solver EF — run ipopt then gurobi, pick best
        ef_iter_info = {
            "ef_attempted": False,
            "ef_ok": False,
            "solver_status": None,
            "termination_condition": None,
            "ef_time_sec": 0.0,
            "K_ef": None,
            "ef_obj": None,
            "true_obj": None,
            "ub_updated_by_ef": False,
        }
        # Store per-solver results for logging
        _ef_dual_results = {}  # solver_name -> {ok, K, ef_obj, true_obj, status, term, time}

        if enable_ef_ub:
            ef_iter_info["ef_attempted"] = True
            verts4 = [tuple(map(float, v)) for v in lb_simp_rec["verts"]]
            c_pts = lb_simp_rec.get("c_point_per_scene", [])

            for _solver_tag, _ef_inst in [("ipopt", ef_ub_ipopt), ("gurobi", ef_ub_gurobi)]:
                if _ef_inst is None:
                    _ef_dual_results[_solver_tag] = {"ok": False, "status": "not_initialized"}
                    continue
                try:
                    _ef_inst.update_simplex_vertices(verts4)
                    if c_pts:
                        _ef_inst.set_warm_start(c_pts)
                    _ef_ok, _K_ef, _ef_obj, _ef_info = _ef_inst.solve()
                    _res = {
                        "ok": _ef_ok,
                        "K": _K_ef,
                        "ef_obj": _ef_obj,
                        "true_obj": None,
                        "status": _ef_info.get("solver_status"),
                        "term": _ef_info.get("termination_condition"),
                        "time": _ef_info.get("time_sec", 0.0),
                        "lower_bound": _ef_info.get("lower_bound"),  # Gurobi LB
                    }
                    if _ef_ok and _K_ef is not None:
                        _true = evaluate_true_expected_obj(base_bundles, first_vars_list, _K_ef)
                        # Fallback: if the per-scenario recalculation returns
                        # Q_max (infeasible), trust the EF solver's objective.
                        # This occurs when the EF optimal point slightly
                        # violates coupling constraints (e.g. sum(x) <= 500)
                        # due to solver numerical tolerance, making individual
                        # BaseBundle.eval_at() calls return infeasible.
                        from bundles import Q_max as _Q_max_import
                        if _true >= _Q_max_import and _ef_obj is not None and math.isfinite(_ef_obj):
                            if verbose:
                                print(f"[Iter {it}] EF-{_solver_tag}: true_obj={_true:.3e} >= Q_max, "
                                      f"falling back to ef_obj*S = {_ef_obj}*{S} = {_ef_obj*S:.6f}")
                            _true = _ef_obj * S
                        _res["true_obj"] = _true
                    _ef_dual_results[_solver_tag] = _res
                    if verbose:
                        _tstr = f"{_res['true_obj']/S:.6e}" if _res['true_obj'] is not None else "N/A"
                        print(f"[Iter {it}] EF-{_solver_tag}: ok={_ef_ok}, "
                              f"ef_obj={_ef_obj}, true_obj={_tstr}, time={_res['time']:.3f}s")
                except Exception as e:
                    _ef_dual_results[_solver_tag] = {"ok": False, "status": "exception", "term": str(e)}
                    if verbose:
                        print(f"[Iter {it}] EF-{_solver_tag} exception: {e}")

            # Pick the best (lowest true_obj) among successful solvers
            _best_solver = None
            _best_true = float('inf')
            for _stag, _sres in _ef_dual_results.items():
                if _sres.get("ok") and _sres.get("true_obj") is not None:
                    if _sres["true_obj"] < _best_true:
                        _best_true = _sres["true_obj"]
                        _best_solver = _stag

            if _best_solver is not None:
                _bres = _ef_dual_results[_best_solver]
                ef_iter_info["ef_ok"] = True
                ef_iter_info["solver_status"] = f"{_best_solver}:{_bres.get('status')}"
                ef_iter_info["termination_condition"] = f"{_best_solver}:{_bres.get('term')}"
                ef_iter_info["ef_time_sec"] = sum(
                    r.get("time", 0.0) for r in _ef_dual_results.values() if isinstance(r.get("time"), (int, float))
                )
                ef_iter_info["K_ef"] = _bres["K"]
                ef_iter_info["ef_obj"] = _bres["ef_obj"]
                ef_iter_info["true_obj"] = _bres["true_obj"]

                if _best_true < UB_incumbent:
                    UB_incumbent = _best_true
                    UB_global = UB_incumbent
                    UB_node = _bres["K"]
                    UB_node_incumbent = _bres["K"]  # FIX-A: sync persistent incumbent point
                    ub_updated_this_iter = True
                    ub_source_this_iter = f"EF_{_best_solver}"
                    ub_simplex_id_this_iter = tuple(sorted(lb_simp_rec["vert_idx"]))
                    ef_iter_info["ub_updated_by_ef"] = True
                    if verbose:
                        print(f"[Iter {it}] EF-UB improved by {_best_solver}: "
                              f"{_best_true/S:.6e} (per-scen), K={_bres['K']}")
            else:
                # Both solvers failed
                _any = next(iter(_ef_dual_results.values()), {})
                ef_iter_info["solver_status"] = _any.get("status", "all_failed")
                ef_iter_info["termination_condition"] = _any.get("term", "all_failed")
                if verbose:
                    print(f"[Iter {it}] EF-UB: both solvers failed")

            # Build EF/MS log row (deferred: will be written at end-of-iteration
            # so LB_global_end_sum and UB_global_end_sum reflect post-split values)
            try:
                ms_issues = collect_ms_cs_issues(lb_simp_rec)
                _ef_log_row = {
                    "iter": it,
                    "simplex_id": tuple(sorted(lb_simp_rec["vert_idx"])),
                    "ef_attempted": ef_iter_info["ef_attempted"],
                    "ef_ok": ef_iter_info["ef_ok"],
                    "solver_status": ef_iter_info["solver_status"],
                    "termination_condition": ef_iter_info["termination_condition"],
                    "ef_time_sec": f"{ef_iter_info['ef_time_sec']:.3f}",
                    "K_ef": ef_iter_info["K_ef"],
                    "ef_obj": ef_iter_info["ef_obj"],
                    "true_obj": ef_iter_info["true_obj"],
                    "ub_updated": ef_iter_info["ub_updated_by_ef"],
                    "UB_incumbent": f"{UB_incumbent/S:.9f}",
                    # LB construction terms (from pre-split selected simplex)
                    "lb_simp_LB_sur": f"{lb_simp_rec['LB']:.6f}",
                    "lb_simp_LB_linear": f"{lb_simp_rec.get('LB_terms',{}).get('LB_linear', 0.0):.6f}",
                    "lb_simp_c_total_finite": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_finite', float('nan')):.6f}",
                    "lb_simp_c_total_all": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_all', 0.0):.6f}",
                    "lb_simp_ms_total": f"{lb_simp_rec.get('LB_terms',{}).get('ms_total', 0.0):.6f}",
                    "lb_simp_min_fverts_sum": f"{lb_simp_rec.get('LB_terms',{}).get('min_fverts_sum', 0.0):.6f}",
                    "lb_simp_lb_case": lb_simp_rec.get('LB_terms',{}).get('lb_case', ''),
                    # Placeholders — filled at end-of-iteration
                    "LB_global_end_sum": "",
                    "UB_global_end_sum": "",
                }
                _ef_log_row.update(ms_issues)
            except Exception:
                _ef_log_row = None

        else:
            # EF disabled/unavailable — still build deferred log row
            try:
                ms_issues = collect_ms_cs_issues(lb_simp_rec)
                _ef_log_row = {
                    "iter": it,
                    "simplex_id": tuple(sorted(lb_simp_rec["vert_idx"])),
                    "ef_attempted": ef_iter_info["ef_attempted"],
                    "ef_ok": ef_iter_info["ef_ok"],
                    "solver_status": ef_iter_info["solver_status"],
                    "termination_condition": ef_iter_info["termination_condition"],
                    "ef_time_sec": "",
                    "K_ef": "",
                    "ef_obj": "",
                    "true_obj": "",
                    "ub_updated": "",
                    "UB_incumbent": f"{UB_incumbent/S:.9f}",
                    # LB construction terms (from pre-split selected simplex)
                    "lb_simp_LB_sur": f"{lb_simp_rec['LB']:.6f}",
                    "lb_simp_LB_linear": f"{lb_simp_rec.get('LB_terms',{}).get('LB_linear', 0.0):.6f}",
                    "lb_simp_c_total_finite": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_finite', float('nan')):.6f}",
                    "lb_simp_c_total_all": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_all', 0.0):.6f}",
                    "lb_simp_ms_total": f"{lb_simp_rec.get('LB_terms',{}).get('ms_total', 0.0):.6f}",
                    "lb_simp_min_fverts_sum": f"{lb_simp_rec.get('LB_terms',{}).get('min_fverts_sum', 0.0):.6f}",
                    "lb_simp_lb_case": lb_simp_rec.get('LB_terms',{}).get('lb_case', ''),
                    # Placeholders — filled at end-of-iteration
                    "LB_global_end_sum": "",
                    "UB_global_end_sum": "",
                }
                _ef_log_row.update(ms_issues)
            except Exception:
                _ef_log_row = None

        # === Per-scenario bound debug: log dual > primal violations ===
        try:
            ms_metas = lb_simp_rec.get("ms_meta_per_scene", [])
            cs_metas = lb_simp_rec.get("cs_meta_per_scene", [])
            _simplex_id_str = str(tuple(sorted(lb_simp_rec["vert_idx"])))
            for s in range(S):
                # MS dual > primal check
                mm = ms_metas[s] if s < len(ms_metas) and ms_metas[s] is not None else {}
                ms_lb = mm.get("dual_bound")
                ms_pri = mm.get("primal_obj")
                if (ms_lb is not None and ms_pri is not None
                        and math.isfinite(ms_lb) and math.isfinite(ms_pri)
                        and ms_lb > ms_pri + 1e-8):
                    _gap = ms_lb - ms_pri
                    _line = (f"[Iter {it}] [MS] simplex_idx={lb_simp_rec['simplex_index']}, "
                             f"verts={_simplex_id_str}, scenario={s}, "
                             f"dual_bound={ms_lb}, primal_obj={ms_pri}, gap={_gap}\n")
                    with open(dbg_dual_gt_obj_path, "a", encoding="utf-8") as _fdgo:
                        _fdgo.write(_line)
                # CS dual > primal check
                cm = cs_metas[s] if s < len(cs_metas) and cs_metas[s] is not None else {}
                cs_lb = cm.get("dual_bound")
                cs_pri = cm.get("primal_obj")
                if (cs_lb is not None and cs_pri is not None
                        and math.isfinite(cs_lb) and math.isfinite(cs_pri)
                        and cs_lb > cs_pri + 1e-8):
                    _gap = cs_lb - cs_pri
                    _line = (f"[Iter {it}] [CS] simplex_idx={lb_simp_rec['simplex_index']}, "
                             f"verts={_simplex_id_str}, scenario={s}, "
                             f"dual_bound={cs_lb}, primal_obj={cs_pri}, gap={_gap}\n")
                    with open(dbg_dual_gt_obj_path, "a", encoding="utf-8") as _fdgo:
                        _fdgo.write(_line)
        except Exception:
            pass  # bound debug must never crash the algorithm


        # === NEW: record c info for the simplex ===
        c_scene = [float(x) for x in lb_simp_rec.get("c_per_scene", [])]
        finite_c = [x for x in c_scene if math.isfinite(x)]
        if finite_c:
            c_agg = float(sum(finite_c))
        else:
            c_agg = float('-inf')

        lb_c_agg_hist.append(c_agg)
        lb_c_per_scene_hist.append(c_scene)

        '''
        ms_ub_active_this_iter = []
        for r in ub_active:
            ms_scene = r.get("ms_per_scene", None)
            if ms_scene is None:
                continue
            ms_ub_active_this_iter.append([float(x) for x in ms_scene])
        ms_ub_active_per_iter.append(ms_ub_active_this_iter)
        '''


        # 6) ms_a
        # ms_a: smallest ms among active simplices that have MS computed
        # (excludes ms_status="none" to avoid silently treating absent MS as real +inf)
        if any(active_mask.values()):
            _ms_vals_active = [
                float(r.get("ms", float('inf')))
                for r in per_tet
                if active_mask.get(r["simplex_index"], False)
                and r.get("ms_status", "full") != "none"
            ]
        else:
            _ms_vals_active = [
                float(r.get("ms", float('inf')))
                for r in per_tet
                if r.get("ms_status", "full") != "none"
            ]
        ms_a = float(min(_ms_vals_active)) if _ms_vals_active else float('inf')
        ms_iter = ms_a


        # === Print the current round's optimality gap (PRE-SPLIT, not final) ===
        # Note: This is before end-of-iter update; final gap printed later
        gap_abs = float(UB_global - LB_global)
        gap_pct = (gap_abs / (abs(UB_global) + 1e-16)) * 100.0
        if verbose:
            print(f"[Iter {it}] Pre-split gap: {gap_abs:.6e} ({gap_pct:.3f}%)")

        # 7) record
        LB_hist.append(LB_global)
        UB_hist.append(UB_global)
        ms_hist.append(ms_iter)
        node_count.append(len(nodes))
        UB_node_hist.append(UB_node)
        ms_a_hist.append(ms_a)
        ms_b_hist.append(ms_b)
        active_ratio_hist.append(active_ratio)

        # NEW: time & simplex stats for this iteration
        now = perf_counter()
        iter_time_hist.append(now - t_start)

        num_simplices = len(per_tet)
        num_active_simplices = sum(
            1 for r in per_tet if active_mask.get(r["simplex_index"], False)
        )
        simplex_hist.append(num_simplices)
        active_simplex_hist.append(num_active_simplices)
        # NOTE: split_kind_hist is recorded AFTER the actual split (see post-subdivide block)

        # LB/UB per user definition: CSV writing moved to AFTER end-of-iter block



        # LB/UB per user definition: Stop conditions moved to AFTER end-of-iter block

        # 8) print
        # Geometric containment: find which simplex(es) contain UB_node
        _ub_pt = np.asarray(UB_node, dtype=float)
        _geom_containing = []
        _geom_tol = 1e-8
        for _r in per_tet:
            _V = np.array(_r["verts"], dtype=float)  # (4, 3)
            # Barycentric: p = V[0] + T @ lam_123, where T = [V1-V0, V2-V0, V3-V0]^T
            _T = (_V[1:] - _V[0]).T  # (3, 3)
            try:
                _lam123 = _safe_linalg.solve(_T, _ub_pt - _V[0])
                _lam0 = 1.0 - _lam123.sum()
                _lam_all = np.array([_lam0, *_lam123])
                if np.all(_lam_all >= -_geom_tol):
                    _geom_containing.append((_r["simplex_index"], _lam_all))
            except np.linalg.LinAlgError:
                pass  # degenerate simplex, skip

        if verbose:
            print(f"[Iter {it}] Active simplex ratio = {active_ratio:.6f}")
            if _geom_containing:
                _simp_ids = [s for s, _ in _geom_containing]
                print(f"[Iter {it}] UB point {UB_node} is geometrically in simplices {_simp_ids}")
                for _si, _lam in _geom_containing:
                    _r_match = [r for r in per_tet if r["simplex_index"] == _si][0]
                    print(f"  T{_si} verts={_r_match['vert_idx']}: "
                          f"lam=[{', '.join(f'{l:.6f}' for l in _lam)}], "
                          f"LB={_r_match['LB']:.9f}, c_s={_r_match['c_per_scene']}")

                # === CS dual vs primal diagnostic for simplices containing UB point ===
                print(f"\n== [Iter {it}] CS DUAL vs PRIMAL diagnostic (simplices containing UB point) ==")
                for _si, _lam in _geom_containing:
                    _r_match = [r for r in per_tet if r["simplex_index"] == _si][0]
                    _cs_metas = _r_match.get("cs_meta_per_scene", [])
                    _cs_pts   = _r_match.get("c_point_per_scene", [])
                    print(f"  --- T{_si} ---")
                    for _s_idx in range(S):
                        _meta = _cs_metas[_s_idx] if _s_idx < len(_cs_metas) else {}
                        if _meta is None:
                            _meta = {}
                        _dual = _meta.get("dual_bound", None)
                        _prim = _meta.get("primal_obj", None)
                        _status = _meta.get("status", "?")
                        _cs_pt = _cs_pts[_s_idx] if _s_idx < len(_cs_pts) else None

                        _dual_str = f"{_dual:.9f}" if _dual is not None else "N/A"
                        _prim_str = f"{_prim:.9f}" if _prim is not None else "N/A"
                        _check = ""
                        if _dual is not None and _prim is not None:
                            if _dual > _prim + 1e-8:
                                _check = "  [!] DUAL > PRIMAL (Gurobi inconsistency!)"
                            else:
                                _check = f"  [OK] dual <= primal (gap={_prim - _dual:.6e})"

                        print(f"    scen {_s_idx}: status={_status}, dual={_dual_str}, primal={_prim_str}{_check}")
                        print(f"             cs_primal_K={_cs_pt}")

                        # Independent Q evaluation at the CS primal point
                        if _cs_pt is not None:
                            try:
                                _indep_Q = evaluate_Q_at(base_bundles[_s_idx], first_vars_list[_s_idx], _cs_pt)
                                _indep_str = f"{_indep_Q:.9f}"
                                _verdict = ""
                                if _dual is not None and math.isfinite(_indep_Q):
                                    if _indep_Q < _dual - 1e-8:
                                        _verdict = "  -> DUAL BOUND IS UNRELIABLE (indep_Q < dual)"
                                    else:
                                        _verdict = f"  -> dual <= indep_Q (consistent, diff={_indep_Q - _dual:.6e})"
                                print(f"             indep_Q(cs_K)={_indep_str}{_verdict}")
                            except Exception as _e:
                                print(f"             indep_Q(cs_K): FAILED ({_e})")

                    # Per-scenario UB comparison
                    _ub_per_scen = UB_global * S  # total UB
                    print(f"  sum(c_s_dual)={sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.9f}, "
                          f"UB_total={_ub_per_scen:.9f}, UB_per_scen={UB_global:.9f}")

                    # === Per-scenario Q_s(K_EF) evaluation ===
                    print(f"\n  == Per-scenario Q_s at K_EF = {UB_node} ==")
                    _sum_q_ef = 0.0
                    for _s_idx in range(S):
                        try:
                            _q_ef_s = evaluate_Q_at(base_bundles[_s_idx], first_vars_list[_s_idx], UB_node)
                            _sum_q_ef += _q_ef_s
                        except Exception as _e:
                            _q_ef_s = float('nan')
                            print(f"    scen {_s_idx}: Q_s(K_EF) evaluation FAILED: {_e}")
                            continue

                        _cs_m = _cs_metas[_s_idx] if _s_idx < len(_cs_metas) else None
                        _cs_dual_s = (_cs_m or {}).get("dual_bound", None)
                        _cs_prim_s = (_cs_m or {}).get("primal_obj", None)

                        _flag = ""
                        if _cs_dual_s is not None:
                            if _q_ef_s < _cs_dual_s - 1e-8:
                                _flag = "  [!] Q_s(K_EF) < c_s_dual -> GUROBI CS DUAL IS WRONG"
                            elif _q_ef_s < _cs_dual_s + 1e-6:
                                _flag = "  ~ Q_s(K_EF) ~= c_s_dual (within tol)"
                            else:
                                _flag = f"  [OK] Q_s(K_EF) > c_s_dual (diff={_q_ef_s - _cs_dual_s:.6e})"

                        _cs_d_str = f"{_cs_dual_s:.9f}" if _cs_dual_s is not None else "N/A"
                        print(f"    scen {_s_idx}: Q_s(K_EF)={_q_ef_s:.9f}, c_s_dual={_cs_d_str}{_flag}")

                    print(f"    SUM Q_s(K_EF)={_sum_q_ef:.9f}, sum(c_s)={sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.9f}, "
                          f"diff={_sum_q_ef - sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.6e}")
                print()
            else:
                print(f"[Iter {it}] UB point {UB_node} is NOT inside any simplex (outside mesh)")
            msb_src = f"T{ms_b_simp}" if ms_b_simp is not None else "N/A"
            print(f"[Iter {it}] LB = {LB_global:.6f} = UB({UB_global:.6f}) + ms_b({ms_b:.3e}) from {msb_src}")

        _phases["5_EF_UB_solve"] = perf_counter() - _t_phase

        # 9) next node candidate ranking & selection
        # ------------------------------------------------
        _t_phase = perf_counter()
        # For LB-simplex, check if "all ms are +inf" -> use c_s-based fallback point selection
        use_c_fallback = False
        lb_ms_list = lb_simp_rec.get("ms_per_scene", [])
        if lb_ms_list:
            if all(math.isinf(float(ms_val)) for ms_val in lb_ms_list):
                c_list_lb = lb_simp_rec.get("c_per_scene", [])
                if c_list_lb and any(math.isfinite(float(c)) for c in c_list_lb):
                    use_c_fallback = True
                    if verbose:
                        print(f"[Iter {it}] Using c_s-based candidate in LB simplex T{lb_simp_idx} "
                              f"because all ms_s are +inf.")

        # New strategy: select next points only from the "currently tightest simplex block".
        active = [lb_simp_rec]              
        pool_records = [lb_simp_rec]
    
        # Collect candidate points for each scene from this simplex.
        # - Normal case: use ms_per_scene + xms_per_scene
        # - Fallback case (use_c_fallback=True): use c_per_scene + c_point_per_scene
        cand_items = []

        # Tolerance parameters for point selection


        for rec in pool_records:
            sid = rec["simplex_index"]
            verts = np.array(rec["verts"])

            if use_c_fallback:
                val_list = rec.get("c_per_scene", [])
                pts_list = rec.get("c_point_per_scene", [None] * len(val_list))
                # In fallback, we only have c_s points
                for s in range(len(val_list)):
                    cand_items.append({
                        "simplex_index": sid,
                        "scene": s,
                        "cand_ms": val_list[s],
                        "cand_pt": pts_list[s],
                        "pt_source": "c_s_fallback",
                        "_rec": rec
                    })
            else:
                ms_vals = rec.get("ms_per_scene", [])
                ms_pts = rec.get("xms_per_scene", [None] * len(ms_vals))
                c_vals = rec.get("c_per_scene", [])
                c_pts = rec.get("c_point_per_scene", [None] * len(c_vals))

                # Always add BOTH ms and c_s as separate candidates
                for s in range(len(ms_vals)):
                    ms_pt = ms_pts[s]
                    # Add ms candidate
                    cand_items.append({
                        "simplex_index": sid,
                        "scene": s,
                        "cand_ms": ms_vals[s],
                        "cand_pt": ms_pt,
                        "pt_source": "ms",
                        "_rec": rec
                    })
                    # Add c_s candidate (if available and different from ms)
                    c_pt = c_pts[s] if s < len(c_pts) else None
                    if c_pt is not None:
                        c_val = c_vals[s] if s < len(c_vals) else ms_vals[s]
                        cand_items.append({
                            "simplex_index": sid,
                            "scene": s,
                            "cand_ms": c_val,
                            "cand_pt": c_pt,
                            "pt_source": "c_s",
                            "_rec": rec
                        })

        # --- Aggregate candidate points (avg_cs, avg_ms) ---
        # Always add the average of all valid per-scenario c_s and ms points
        # for the selected simplex, when they are well-defined.
        for rec in pool_records:
            sid = rec["simplex_index"]

            # avg_cs: mean of all valid c_point_per_scene
            _cs_pts_raw = rec.get("c_point_per_scene", [])
            _valid_cs = [pt for pt in _cs_pts_raw
                         if pt is not None and all(math.isfinite(v) for v in pt)]
            if _valid_cs:
                _avg_cs_pt = tuple(float(x) for x in np.mean(_valid_cs, axis=0))
                # Use the average of valid c_per_scene values as the surrogate score
                _cs_vals_raw = rec.get("c_per_scene", [])
                _valid_cs_vals = [float(v) for v in _cs_vals_raw
                                  if v is not None and math.isfinite(float(v))]
                _avg_cs_val = float(np.mean(_valid_cs_vals)) if _valid_cs_vals else 0.0
                cand_items.append({
                    "simplex_index": sid,
                    "scene": -1,
                    "cand_ms": _avg_cs_val,
                    "cand_pt": _avg_cs_pt,
                    "pt_source": "avg_cs",
                    "_rec": rec
                })

            # avg_ms: mean of all valid xms_per_scene (only in non-fallback branch)
            if not use_c_fallback:
                _ms_pts_raw = rec.get("xms_per_scene", [])
                _valid_ms = [pt for pt in _ms_pts_raw
                             if pt is not None and all(math.isfinite(v) for v in pt)]
                if _valid_ms:
                    _avg_ms_pt = tuple(float(x) for x in np.mean(_valid_ms, axis=0))
                    _ms_vals_raw = rec.get("ms_per_scene", [])
                    _valid_ms_vals = [float(v) for v in _ms_vals_raw
                                      if v is not None and math.isfinite(float(v))]
                    _avg_ms_val = float(np.mean(_valid_ms_vals)) if _valid_ms_vals else 0.0
                    cand_items.append({
                        "simplex_index": sid,
                        "scene": -1,
                        "cand_ms": _avg_ms_val,
                        "cand_pt": _avg_ms_pt,
                        "pt_source": "avg_ms",
                        "_rec": rec
                    })

        # (Mode 2 note: when split_mode==2, the user supplies custom initial_nodes
        #  which are already Delaunay-tessellated at init. No special in-loop logic
        #  is needed — Mode 1 candidate selection runs normally from here.)

        # === Debug: write candidate details to debug_candidates.txt ===
        with open(dbg_candidates_path, "a", encoding="utf-8") as _dbf:
            _dbf.write(f"{'='*80}\n")
            _dbf.write(f"[Iter {it}] LB simplex = T{lb_simp_idx}\n")
            _dbf.write(f"  LB simplex vertices:\n")
            for vi, v in enumerate(lb_simp_rec['verts']):
                vidx = lb_simp_rec['vert_idx'][vi]
                _dbf.write(f"    v{vi} (node #{vidx}): {tuple(map(float, v))}\n")
            _dbf.write(f"  Existing nodes ({len(nodes)} total):\n")
            for ni, nd in enumerate(nodes):
                _dbf.write(f"    node #{ni}: {tuple(map(float, nd))}\n")
            _dbf.write(f"\n  Candidates ({len(cand_items)} total):\n")
            for ci_idx, ci in enumerate(cand_items):
                pt = ci['cand_pt']
                src = ci.get('pt_source', '?')
                if pt is not None:
                    dist = min_dist_to_nodes(pt, nodes)
                    collision = "COLLISION" if dist < min_dist else "ok"
                else:
                    dist = float('inf')
                    collision = "None_pt"
                _dbf.write(
                    f"    [{ci_idx}] simp=T{ci['simplex_index']}, scene={ci['scene']}, "
                    f"source={src}, val={float(ci['cand_ms']):.6e}, "
                    f"pt={tuple(map(float, pt)) if pt is not None else None}, "
                    f"dist_to_nodes={dist:.6e}, {collision}\n"
                )
            _dbf.write("\n")
        # --- POINT_SELECT_MODE: 1=ms-first, 2=weighted composite, 3=weighted avg ms,
        #     5=pseudo-Bayesian (mu - lambda*sigma) ---
        POINT_SELECT_MODE = 5

        # --- Mode 5 tuning parameters ---
        MODE5_LAMBDA = 1.0          # exploration weight: higher = more variance reward
        MODE5_VTOL_FRAC = 0.05      # vertex-nearness threshold as fraction of simplex diameter

        new_node = None
        chosen_ms = None
        chosen_cand = None
        stop_due_to_collision = False
        collision_node_idx = None    # index of the node the candidate collided with

        def handle_collision(cand_pt, ci, stage_note="active"):
            nonlocal stop_due_to_collision, collision_node_idx
            X = np.asarray(nodes, float)
            P = np.asarray(cand_pt, float)
            dists = np.linalg.norm(X - P, axis=1)
            j_star = int(np.argmin(dists))
            d_star = float(dists[j_star])
            orange_ids = [r["simplex_index"] for r in per_tet if j_star in r["vert_idx"]]
            debug_pack = {
                "reason": "candidate_too_close",
                "iter": it,
                "stage": stage_note,
                "min_dist": float(min_dist),
                "closest_node_index": j_star,
                "closest_node_point": tuple(map(float, nodes[j_star])),
                "closest_distance": d_star,
                "cand_simplex": int(ci["simplex_index"]),
                "cand_scene": int(ci["scene"]),
                "cand_point": tuple(map(float, cand_pt)),
                "cand_ms": float(ci["cand_ms"]),
                "UB_global": float(UB_global),
                "LB_global": float(LB_global),
                "active_ratio": float(active_ratio),
                "UB_node": tuple(map(float, UB_node)),
                "active_mask": {int(k): bool(v) for k, v in active_mask.items()},
                "nodes_snapshot": [tuple(map(float, nd)) for nd in nodes],
                "per_tet_snapshot": [
                    {
                        "simplex_index": int(r["simplex_index"]),
                        "vert_idx": list(map(int, r["vert_idx"])),
                        "verts": [tuple(map(float, x)) for x in r['verts']],
                        "ms": float(r["ms"]),
                        "ms_per_scene": [float(x) for x in r.get("ms_per_scene", [])],
                        "LB": float(r["LB"]),
                        "UB": float(r["UB"]),
                        "best_scene": int(r["best_scene"]) if r.get("best_scene") is not None else -1,
                        "x_ms_best_scene": tuple(map(float, r["x_ms_best_scene"])) if r.get("x_ms_best_scene") is not None else None,
                        "volume": float(r["volume"]),
                    } for r in per_tet
                ],
                "highlight_simplices": list(map(int, orange_ids)),
            }
            global LAST_DEBUG
            LAST_DEBUG = debug_pack
            if verbose:
                print(
                    f"[STOP] Candidate {tuple(map(float, cand_pt))} "
                    f"(scene {ci['scene']}) is too close to existing node #{j_star} at distance {d_star:.3e} "
                    f"(< {min_dist:g}). Highlighted simplices: {sorted(orange_ids)}"
                )
            stop_due_to_collision = True
            collision_node_idx = j_star

        # ---------- Mode 2 (ms weighted composite from LB-best simplex) ----------
        if POINT_SELECT_MODE == 2:
            rec = lb_simp_rec
            ms_list  = rec.get("ms_per_scene", [])
            pts_list = rec.get("xms_per_scene", [])

            weights = []
            points  = []
            for ms_val, pt in zip(ms_list, pts_list):
                if pt is None:
                    continue
                w = max(0.0, -float(ms_val))
                weights.append(w)
                points.append(np.asarray(pt, float))

            if weights:
                w_arr = np.asarray(weights, float)
                if w_arr.sum() <= 0:
                    w_arr[:] = 1.0
                w_arr /= w_arr.sum()

                candidate_pt = sum(w * p for w, p in zip(w_arr, points))

                if min_dist_to_nodes(candidate_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(candidate_pt, rec)
                    new_node   = cand_pt_pert
                    chosen_ms  = float(np.dot(w_arr, np.array(ms_list, float)))
                    chosen_cand = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": chosen_ms,
                        "cand_pt": cand_pt_pert,
                        "_rec": rec,
                        "loc_type": loc_type,
                        "loc_info": loc_info,
                        "pt_source": "mode2_weighted",
                    }
                    if verbose:
                        print(
                            f"Chosen node (MODE2) {tuple(map(float, cand_pt_pert))} "
                            f"with weighted ms={chosen_ms:.3e} "
                            f"(simp T{rec['simplex_index']})"
                        )
                else:
                    dummy_ci = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": 0.0,
                        "cand_pt": tuple(candidate_pt),
                    }
                    handle_collision(candidate_pt, dummy_ci, stage_note="mode2")


        # ---------- Mode 3: weighted average of all ms candidate points ----------
        if POINT_SELECT_MODE == 3 and (not stop_due_to_collision):
            # Collect only true ms-sourced candidates with valid points
            ms_cands_m3 = [ci for ci in cand_items
                           if ci.get("pt_source", "") == "ms" and ci["cand_pt"] is not None]
            if verbose:
                print(f"[mode3] using {len(ms_cands_m3)} true ms points for weighted average")

            weights_m3 = []
            points_m3  = []
            for ci in ms_cands_m3:
                w = max(0.0, -float(ci["cand_ms"]))  # more negative ms → higher weight
                weights_m3.append(w)
                points_m3.append(np.asarray(ci["cand_pt"], float))

            if weights_m3:
                w_arr = np.asarray(weights_m3, float)
                if w_arr.sum() <= 0:
                    w_arr[:] = 1.0
                w_arr /= w_arr.sum()

                candidate_pt = sum(w * p for w, p in zip(w_arr, points_m3))

                # Use the LB-best simplex record for snapping/subdivision
                rec = lb_simp_rec
                if min_dist_to_nodes(candidate_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(candidate_pt, rec)
                    new_node   = cand_pt_pert
                    chosen_ms  = float(np.dot(w_arr, [ci["cand_ms"] for ci in ms_cands_m3]))
                    chosen_cand = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": chosen_ms,
                        "cand_pt": cand_pt_pert,
                        "_rec": rec,
                        "loc_type": loc_type,
                        "loc_info": loc_info,
                        "pt_source": "mode3_weighted_all_ms",
                    }
                    if verbose:
                        print(
                            f"Chosen node (MODE3) {tuple(map(float, cand_pt_pert))} "
                            f"with weighted ms={chosen_ms:.3e} "
                            f"({len(ms_cands_m3)} ms points, simp T{rec['simplex_index']})"
                        )
                else:
                    dummy_ci = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": 0.0,
                        "cand_pt": tuple(candidate_pt),
                    }
                    handle_collision(candidate_pt, dummy_ci, stage_note="mode3")


        # ---------- Mode 5: pseudo-Bayesian (mu - lambda * sigma) ----------
        if POINT_SELECT_MODE == 5 and (not stop_due_to_collision):
            rec = lb_simp_rec
            _m5_verts = np.array(rec["verts"], dtype=float)

            # Compute simplex diameter (max edge length) for vertex-nearness threshold
            _m5_diam = 0.0
            for _vi in range(len(_m5_verts)):
                for _vj in range(_vi + 1, len(_m5_verts)):
                    _m5_diam = max(_m5_diam, float(np.linalg.norm(_m5_verts[_vi] - _m5_verts[_vj])))
            _m5_vtol = MODE5_VTOL_FRAC * _m5_diam

            # Helper: is a point too close to any simplex vertex?
            def _m5_near_vertex(pt, verts, tol):
                p = np.asarray(pt, float)
                for v in verts:
                    if np.linalg.norm(p - np.asarray(v, float)) < tol:
                        return True
                return False

            # Filter candidates: must have valid point and not be None
            _m5_pool = [ci for ci in cand_items if ci["cand_pt"] is not None]

            # Step B: separate into non-vertex and vertex-near sets
            _m5_filtered = [ci for ci in _m5_pool
                            if not _m5_near_vertex(ci["cand_pt"], _m5_verts, _m5_vtol)]
            _m5_vtx_removed = len(_m5_pool) - len(_m5_filtered)

            # Fallback: if all candidates filtered, use unfiltered pool
            _m5_use_fallback = (len(_m5_filtered) == 0 and len(_m5_pool) > 0)
            _m5_working = _m5_filtered if _m5_filtered else _m5_pool

            if verbose:
                print(f"[mode5] pool={len(_m5_pool)}, vtx_removed={_m5_vtx_removed}, "
                      f"remaining={len(_m5_filtered)}, fallback={_m5_use_fallback}, "
                      f"vtol={_m5_vtol:.3e}, diam={_m5_diam:.3e}")

            # Step C: evaluate Q_s(x) for each candidate across all scenarios
            # Uniform weights (equal probability per scenario)
            _m5_w = np.ones(S, dtype=float) / S

            _m5_scored = []  # list of (ci, q_vals, mu, sigma, score, near_vtx)
            for ci in _m5_working:
                cand_pt = ci["cand_pt"]
                q_vals = []
                _q_ok = True
                for s in range(S):
                    try:
                        qv = evaluate_Q_at(base_bundles[s], first_vars_list[s], cand_pt)
                        if not math.isfinite(qv):
                            _q_ok = False
                            break
                        q_vals.append(float(qv))
                    except Exception:
                        _q_ok = False
                        break
                if not _q_ok or len(q_vals) != S:
                    continue  # skip candidates with failed Q evaluation

                q_arr = np.array(q_vals, dtype=float)
                mu = float(np.dot(_m5_w, q_arr))
                # Weighted std: sqrt(sum(w_i * (q_i - mu)^2))
                sigma = float(np.sqrt(np.dot(_m5_w, (q_arr - mu) ** 2)))
                score = mu - MODE5_LAMBDA * sigma

                _near_vtx = _m5_near_vertex(cand_pt, _m5_verts, _m5_vtol)
                _m5_scored.append((ci, q_vals, mu, sigma, score, _near_vtx))

            # --- Debug: write mode5 scoring to debug_candidates.txt ---
            try:
                with open(dbg_candidates_path, "a", encoding="utf-8") as _dbf:
                    _dbf.write(f"\n--- [Iter {it}] MODE5 scoring (lambda={MODE5_LAMBDA}, "
                               f"vtol={_m5_vtol:.3e}, fallback={_m5_use_fallback}) ---\n")
                    _dbf.write(f"  {len(_m5_scored)} candidates scored "
                               f"(of {len(_m5_working)} working, {len(_m5_pool)} total):\n")
                    for _rank, (_ci, _qv, _mu, _sig, _sc, _nv) in enumerate(
                            sorted(_m5_scored, key=lambda x: x[4]), start=1):
                        _pt_str = tuple(map(float, _ci["cand_pt"]))
                        _src = _ci.get("pt_source", "?")
                        _scn = _ci.get("scene", -1)
                        _dbf.write(
                            f"    [{_rank}] source={_src}, scene={_scn}, "
                            f"pt={_pt_str}\n"
                            f"         Q_s={[f'{v:.6e}' for v in _qv]}\n"
                            f"         mu={_mu:.6e}, sigma={_sig:.6e}, "
                            f"score={_sc:.6e}, near_vtx={_nv}\n")
                    _dbf.write("\n")
            except Exception:
                pass

            # Step E: select candidate with minimum score
            if _m5_scored:
                _m5_scored.sort(key=lambda x: x[4])  # sort by score ascending
                _best_ci, _best_qv, _best_mu, _best_sig, _best_score, _ = _m5_scored[0]
                _best_pt = _best_ci["cand_pt"]

                if min_dist_to_nodes(_best_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(_best_pt, rec)
                    new_node = cand_pt_pert
                    chosen_ms = _best_score
                    _best_ci["loc_type"] = loc_type
                    _best_ci["loc_info"] = loc_info
                    chosen_cand = _best_ci
                    if verbose:
                        print(f"Chosen node (MODE5) {tuple(map(float, cand_pt_pert))} "
                              f"score={_best_score:.6e} (mu={_best_mu:.6e}, "
                              f"sigma={_best_sig:.6e}, lam={MODE5_LAMBDA}), "
                              f"source={_best_ci.get('pt_source','?')}, "
                              f"simp T{rec['simplex_index']}")
                else:
                    # All mode5 candidates collide — try remaining scored candidates
                    _m5_selected = False
                    for _ci, _qv, _mu, _sig, _sc, _ in _m5_scored[1:]:
                        _pt = _ci["cand_pt"]
                        if min_dist_to_nodes(_pt, nodes) >= min_dist:
                            cand_pt_pert, loc_type, loc_info = _snap_feature(_pt, rec)
                            new_node = cand_pt_pert
                            chosen_ms = _sc
                            _ci["loc_type"] = loc_type
                            _ci["loc_info"] = loc_info
                            chosen_cand = _ci
                            _m5_selected = True
                            if verbose:
                                print(f"Chosen node (MODE5 fallback) "
                                      f"{tuple(map(float, cand_pt_pert))} "
                                      f"score={_sc:.6e}, "
                                      f"source={_ci.get('pt_source','?')}")
                            break
                    if not _m5_selected:
                        # All scored candidates collide
                        handle_collision(_best_pt, _best_ci, stage_note="mode5")
            else:
                if verbose:
                    print(f"[mode5] WARNING: no candidates could be scored")


        # ---------- Mode 1: try ms candidates, then c_s, then edge midpoint ----------
        if POINT_SELECT_MODE == 1 and (not stop_due_to_collision):
            # Separate ms-sourced and c_s-sourced candidates, filtering out None points
            # ms group: "ms", "avg_ms", "c_s_fallback", and any unrecognized source
            # cs group: "c_s", "avg_cs"
            _cs_sources = {"c_s", "avg_cs"}
            ms_cands_raw = [ci for ci in cand_items if ci.get("pt_source", "ms") not in _cs_sources]
            _ms_none_count = sum(1 for ci in ms_cands_raw if ci["cand_pt"] is None)
            ms_cands = [ci for ci in ms_cands_raw if ci["cand_pt"] is not None]
            cs_cands_raw = [ci for ci in cand_items if ci.get("pt_source", "") in _cs_sources]
            _cs_none_count = sum(1 for ci in cs_cands_raw if ci["cand_pt"] is None)
            cs_cands = [ci for ci in cs_cands_raw if ci["cand_pt"] is not None]

            # Sort each group by ms value (ascending = most negative first)
            def score_item(ci):
                ms = ci["cand_ms"]
                pt = ci["cand_pt"]
                d  = (float('inf') if pt is None else min_dist_to_nodes(pt, nodes))
                return (ms, -d)

            ms_cands_sorted = sorted(ms_cands, key=score_item)
            cs_cands_sorted = sorted(cs_cands, key=score_item)

            # Build combined ordered list: all ms first, then all c_s
            candidates_sorted = ms_cands_sorted + cs_cands_sorted

            # Track selection info
            _selection_source = None   # "ms" or "c_s"
            _selection_rank = None     # rank within its group (1-based)
            _ms_collisions = 0
            _cs_collisions = 0

            # Phase 1: try all ms candidates in order
            for rank, ci in enumerate(ms_cands_sorted, start=1):
                cand_pt = ci["cand_pt"]
                if cand_pt is None:
                    continue
                if min_dist_to_nodes(cand_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(cand_pt, ci.get("_rec", None))
                    new_node = cand_pt_pert
                    ci["loc_type"] = loc_type
                    ci["loc_info"] = loc_info
                    chosen_ms  = ci["cand_ms"]
                    chosen_cand = ci
                    _selection_source = "ms"
                    _selection_rank = rank
                    if verbose:
                        pt_source = ci.get("pt_source", "ms")
                        print(
                            f"Chosen node {tuple(map(float, cand_pt_pert))} "
                            f"with ms={chosen_ms:.3e} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}, "
                            f"ms_rank #{rank}/{len(ms_cands_sorted)}, source={pt_source})"
                        )
                    break
                else:
                    _ms_collisions += 1
                    if verbose:
                        print(
                            f"[ms skip #{rank}] {tuple(map(float, cand_pt))} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}) "
                            f"too close to existing nodes (< {min_dist:g})."
                        )

            # Phase 2: if all ms collided, try c_s candidates
            if new_node is None and cs_cands_sorted:
                if verbose and _ms_collisions > 0:
                    print(f"[Iter {it}] All {_ms_collisions} ms candidates collided, trying c_s candidates...")
                for rank, ci in enumerate(cs_cands_sorted, start=1):
                    cand_pt = ci["cand_pt"]
                    if cand_pt is None:
                        continue
                    if min_dist_to_nodes(cand_pt, nodes) >= min_dist:
                        cand_pt_pert, loc_type, loc_info = _snap_feature(cand_pt, ci.get("_rec", None))
                        new_node = cand_pt_pert
                        ci["loc_type"] = loc_type
                        ci["loc_info"] = loc_info
                        chosen_ms  = ci["cand_ms"]
                        chosen_cand = ci
                        _selection_source = "c_s"
                        _selection_rank = rank
                        if verbose:
                            print(
                                f"Chosen node (c_s) {tuple(map(float, cand_pt_pert))} "
                                f"with c_s={chosen_ms:.3e} "
                                f"(simp T{ci['simplex_index']}, scene {ci['scene']}, "
                                f"c_s_rank #{rank}/{len(cs_cands_sorted)})"
                            )
                        break
                    else:
                        _cs_collisions += 1
                        if verbose:
                            print(
                                f"[c_s skip #{rank}] {tuple(map(float, cand_pt))} "
                                f"(simp T{ci['simplex_index']}, scene {ci['scene']}) "
                                f"too close to existing nodes (< {min_dist:g})."
                            )

            # Phase 3: if ALL candidates collided, trigger edge midpoint fallback
            if new_node is None and (_ms_collisions + _cs_collisions) > 0:
                if verbose:
                    print(f"[Iter {it}] ALL candidates collided "
                          f"({_ms_collisions} ms + {_cs_collisions} c_s). "
                          f"Falling back to edge midpoint.")
                # Find the last collision candidate for handle_collision info
                last_ci = (cs_cands_sorted[-1] if cs_cands_sorted
                           else ms_cands_sorted[-1] if ms_cands_sorted else None)
                if last_ci and last_ci["cand_pt"] is not None:
                    handle_collision(last_ci["cand_pt"], last_ci, stage_note="all_collided")
                _selection_source = "collision_edge_midpoint"

            # Print selected simplex vertices
            if new_node is not None and chosen_cand is not None and verbose:
                simp_idx_sel = int(chosen_cand['simplex_index'])
                verts_sel = chosen_cand["_rec"]["verts"]
                print(f"[Selected Simp Info] Iter {it} | Simplex T{simp_idx_sel} Vertices:")
                for v_i, v in enumerate(verts_sel):
                    print(f"  v{v_i}: {tuple(map(float, v))}")
                print(f"  -> New Point: {tuple(map(float, new_node))}")
                print(f"[Iter {it}] LB simplex = T{lb_simp_idx}, "
                      f"next node simplex = T{int(chosen_cand['simplex_index'])}, "
                      f"scene {int(chosen_cand['scene'])}")

            # Selection summary
            if verbose:
                sel_info = f"source={_selection_source}, rank={_selection_rank}"
                if _ms_collisions > 0:
                    sel_info += f", ms_collisions={_ms_collisions}"
                if _cs_collisions > 0:
                    sel_info += f", cs_collisions={_cs_collisions}"
                print(f"[Iter {it}] Selection: {sel_info}")

                top_msg = "N/A"
                if len(candidates_sorted) > 0:
                    t0 = candidates_sorted[0]
                    metric_name = "ms" if not use_c_fallback else "c_s"
                    top_msg = (f"T{int(t0['simplex_index'])}, scene={t0['scene']}, "
                               f"{metric_name}={float(t0['cand_ms']):.3e}")
                print(f"[Iter {it}] candidate rank #1: {top_msg}")
                _print_candidates_table(candidates_sorted, nodes, topN=10)
                print()

        # === Iter 0/1 summary log (first split iteration) ===
        if _deferred_first_split or it == 0:
            try:
                _i0_lines = []
                _i0_lines.append("=" * 70)
                _i0_eval_mode = "deferred first split (iter 1)" if _deferred_first_split else "full MS+CS"
                _i0_lines.append(f"[Iter {it}] SIMPLEX SUMMARY  (eval_mode={_i0_eval_mode})")
                _i0_lines.append(f"  Number of simplices: {len(per_tet)}")
                _i0_lines.append(f"  Number of scenarios: {S}")
                _i0_lines.append("")

                for r in sorted(per_tet, key=lambda rr: rr["simplex_index"]):
                    _sid = r["simplex_index"]
                    _vidx = tuple(sorted(r["vert_idx"]))
                    _lb_ps = float(r["LB"]) / S
                    _cs_st_r = r.get("cs_status", "?")
                    _ms_st_r = r.get("ms_status", "?")
                    _sel_tag = "  [SELECTED]" if _sid == lb_simp_idx else ""

                    # Avg dual bound = average of constant cuts (c_s) across scenarios
                    _c_agg = r.get("c_agg", None)
                    if _c_agg is not None and math.isfinite(float(_c_agg)):
                        _avg_dual = float(_c_agg) / S
                        _avg_dual_str = f"{_avg_dual:.9f}"
                    else:
                        _avg_dual_str = "N/A"

                    # Q value of avg c_s point
                    _cs_pts = r.get("c_point_per_scene", [])
                    _valid_cs = [pt for pt in _cs_pts if pt is not None and all(math.isfinite(v) for v in pt)]
                    if _valid_cs:
                        _avg_cs_pt = tuple(np.mean(_valid_cs, axis=0))
                        _cs_q_val = 0.0
                        for _s in range(S):
                            _cs_q_val += evaluate_Q_at(base_bundles[_s], first_vars_list[_s], _avg_cs_pt)
                        _cs_q_str = f"{_cs_q_val/S:.9f} (per-scen), point={_avg_cs_pt}"
                    else:
                        _cs_q_str = "N/A (no valid c_s points)"

                    _i0_lines.append(f"  --- T{_sid} {_vidx}{_sel_tag} ---")
                    _i0_lines.append(f"    cs_status: {_cs_st_r}    ms_status: {_ms_st_r}")
                    _i0_lines.append(f"    LB: {_lb_ps:.9f} (per-scen)")
                    _i0_lines.append(f"    Avg dual bound (box c_s): {_avg_dual_str}")
                    _i0_lines.append(f"    Avg c_s point Q-value: {_cs_q_str}")
                    _i0_lines.append(f"    Volume: {r.get('volume', 0.0):.6e}")

                # IPOPT EF result (only for the selected simplex)
                _i0_lines.append("")
                _i0_lines.append("  --- IPOPT EF Result (selected LB simplex only) ---")
                _sel_vid = tuple(sorted(lb_simp_rec["vert_idx"])) if lb_simp_rec else "N/A"
                _i0_lines.append(f"    Selected simplex: T{lb_simp_idx} {_sel_vid}")
                _ipopt_r = _ef_dual_results.get("ipopt", {})
                if _ipopt_r and _ipopt_r.get("ok"):
                    _ip_ef = _ipopt_r.get("ef_obj")
                    _ip_true = _ipopt_r.get("true_obj")
                    _ip_ef_str = f"{_ip_ef/S:.9f}" if _ip_ef is not None else "N/A"
                    _ip_true_str = f"{_ip_true/S:.9f}" if _ip_true is not None else "N/A"
                    _i0_lines.append(f"    EF-IPOPT: ok=True, ef_obj(per-scen)={_ip_ef_str}, "
                                     f"true_obj(per-scen)={_ip_true_str}, "
                                     f"status={_ipopt_r.get('status')}, term={_ipopt_r.get('term')}, "
                                     f"time={_ipopt_r.get('time', 0.0):.3f}s")
                    if _ipopt_r.get("K") is not None:
                        _i0_lines.append(f"    EF-IPOPT K point: {_ipopt_r['K']}")
                        # Barycentric coordinates of K within the selected simplex
                        try:
                            _K_pt = np.asarray(_ipopt_r["K"], float)
                            _sv = np.array(lb_simp_rec["verts"], float)  # (4, 3)
                            _T = (_sv[1:] - _sv[0]).T  # (3, 3)
                            _bary_rest = _safe_linalg.solve(_T, _K_pt - _sv[0])  # (3,)
                            _bary = np.concatenate([[1.0 - _bary_rest.sum()], _bary_rest])
                            _bary_str = ", ".join(f"{v:.6f}" for v in _bary)
                            _i0_lines.append(f"    Barycentric coords: [{_bary_str}]")
                        except Exception:
                            _i0_lines.append(f"    Barycentric coords: N/A (computation failed)")
                else:
                    _i0_lines.append(f"    EF-IPOPT: not available or failed")

                # Current UB
                _i0_lines.append("")
                _i0_lines.append("  --- Current UB ---")
                try:
                    _ub_ps = f"{UB_incumbent/S:.9f}" if math.isfinite(UB_incumbent) else str(UB_incumbent)
                except Exception:
                    _ub_ps = "N/A"
                _i0_lines.append(f"    UB value: {_ub_ps} (per-scen)")
                _i0_lines.append(f"    UB point: {UB_node}")
                _ub_src = ub_source_this_iter if ub_updated_this_iter else ub_source_current
                _i0_lines.append(f"    UB source: {_ub_src}")

                # Next chosen simplex
                _i0_lines.append("")
                _i0_lines.append("  --- Next Chosen Simplex ---")
                if chosen_cand is not None:
                    _ch_sid = chosen_cand["simplex_index"]
                    _ch_rec = chosen_cand.get("_rec", {})
                    _ch_vid = tuple(sorted(_ch_rec.get("vert_idx", [])))
                    _ch_scene = chosen_cand.get("scene", "N/A")
                    _ch_src = chosen_cand.get("pt_source", "unknown")
                    _i0_lines.append(f"    Simplex: T{_ch_sid} {_ch_vid}")
                    _i0_lines.append(f"    Scene: {_ch_scene}, source: {_ch_src}")
                    if new_node is not None:
                        _i0_lines.append(f"    New point: {tuple(map(float, new_node))}")
                else:
                    _i0_lines.append(f"    No candidate chosen (collision or stop)")

                _i0_lines.append("=" * 70)

                with open(dbg_iter0_summary_path, "a", encoding="utf-8") as _fi0:
                    _fi0.write("\n".join(_i0_lines) + "\n")
            except Exception as _e:
                # Debug log must never crash the algorithm
                print(f"[debug_lIter0_summary] WARNING: failed to write: {_e}")

        # ------------------------------------------------
        _fallback_used = False
        _fallback_edge_verts = None
        _fallback_simplex_idx = None

        if stop_due_to_collision:
            # --- Edge-midpoint fallback for LP-type problems ---
            # When the ms/cs candidate lands on an existing vertex (common for
            # LP problems where optimizing a linear function over a simplex
            # always yields a vertex), split an edge of the LB simplex.
            # Priority: edges connected to the collision vertex first,
            # sorted by length (longest first); random tie-break.
            _verts_arr = np.array(lb_simp_rec["verts"], float)
            _vert_idxs = list(lb_simp_rec["vert_idx"])
            _lb_simp_idx = int(lb_simp_rec["simplex_index"])

            # Collect edge midpoints, separated by collision-connected vs other
            _collision_edges = []
            _other_edges = []
            for _ei in range(len(_vert_idxs)):
                for _ej in range(_ei + 1, len(_vert_idxs)):
                    _midpt = tuple(float(c) for c in 0.5 * (_verts_arr[_ei] + _verts_arr[_ej]))
                    _elen = float(np.linalg.norm(_verts_arr[_ei] - _verts_arr[_ej]))
                    _edge_info = (_elen, _ei, _ej, _midpt)
                    if collision_node_idx is not None and \
                       collision_node_idx in (_vert_idxs[_ei], _vert_idxs[_ej]):
                        _collision_edges.append(_edge_info)
                    else:
                        _other_edges.append(_edge_info)

            # Sort: shuffle first (random tie-break), then stable-sort longest first
            _rng = np.random.default_rng()
            for _lst in [_collision_edges, _other_edges]:
                _rng.shuffle(_lst)
                _lst.sort(key=lambda x: -x[0])

            # Try collision-connected edges first, then others
            _fallback_used = False
            for _fb_group, _fb_list in [("collision_edge", _collision_edges),
                                         ("other_edge", _other_edges)]:
                for _elen, _ei, _ej, _midpt in _fb_list:
                    _fb_dist = min_dist_to_nodes(_midpt, nodes)
                    if _fb_dist >= min_dist:
                        new_node = _midpt
                        stop_due_to_collision = False
                        _fallback_used = True
                        if verbose:
                            print(f"[Iter {it}] Collision fallback ({_fb_group} midpoint "
                                  f"v{_vert_idxs[_ei]}-v{_vert_idxs[_ej]}, "
                                  f"len={_elen:.3e}): {_midpt}, dist={_fb_dist:.3e}")
                        # Store subdivision info for downstream mesh update
                        _fallback_edge_verts = (_vert_idxs[_ei], _vert_idxs[_ej])
                        _fallback_simplex_idx = _lb_simp_idx
                        break
                    else:
                        if verbose:
                            print(f"[Iter {it}] Collision fallback ({_fb_group} midpoint "
                                  f"v{_vert_idxs[_ei]}-v{_vert_idxs[_ej]}): "
                                  f"too close (dist={_fb_dist:.3e})")
                if _fallback_used:
                    break

            if not _fallback_used:
                termination_reason = "collision"
                if verbose:
                    print(f"[Iter {it}] Stop: all edge-midpoint fallbacks exhausted.")
                break

        if new_node is None:
            termination_reason = "no_valid_candidate"
            if verbose:
                print("New node too close for all candidates (or infeasible ms); stop.")
            # Flush debug log before early exit
            if _iter_debug_lines:
                try:
                    with open(debug_path, "a", encoding="utf-8") as _fdbg:
                        _fdbg.write(f"=== Iter {it} ===\n")
                        _fdbg.writelines(_iter_debug_lines)
                        _fdbg.flush()
                except Exception:
                    pass
            break

        #==debug, can delete
        n_tets = len(per_tet)
        n_active = sum(1 for r in per_tet if active_mask.get(r["simplex_index"], False))
        print(f"[Iter {it}] per_tet={n_tets}, active after q_cut={n_active}, tri_is_None={tri is None}")


        # Print how many simplices failed quality check before Visualization
        if verbose:
            print(f"[Iter {it}] bad-quality active simplices (q < {q_cut:g}): {bad_quality_count}")


        # === debug: Check which simplex the light green true solution points fall into, and active status ===
        if enable_3d_plot and true_opt_points is not None:
            # tri might still be None, or some _Dummy placeholder, do not use find_simplex then
            if (tri is None) or (not hasattr(tri, "find_simplex")):
                if verbose:
                    print(f"[Iter {it}] tri has no find_simplex (type={type(tri)}), skip true_opt debug.")
            else:
                for i, p in enumerate(true_opt_points):
                    p = np.asarray(p, float)
                    simp_idx = int(tri.find_simplex(p))
                    is_act = bool(active_mask.get(simp_idx, False))
                    print(
                        f"[Iter {it}] true_opt[{i}] in simplex {simp_idx}, active={is_act}"
                    )

        # Visualization (plot 3D figures)
        if enable_3d_plot and (plot_every is not None) and (it % plot_every == 0):
            # Use the actual chosen simplex for highlighting if available
            if chosen_cand is not None:
                hl_simplices = [chosen_cand["simplex_index"]]
            elif ms_b_simp is not None:
                hl_simplices = [ms_b_simp]
            else:
                hl_simplices = None
            
            # Use the actual chosen point (new_node) if available, otherwise fallback to cand_pt
            pt_to_plot = new_node if new_node is not None else cand_pt

            plot_iteration_plotly(
                it,
                nodes,
                tri,
                active_mask,
                UB_node,
                pt_to_plot,
                per_tet,
                highlight_simplices=hl_simplices,
                true_opt_points=true_opt_points,
                UB_global=UB_global,
                LB_global=LB_global,
                output_dir=plot_output_dir,
                axis_labels=axis_labels,
            )

        _phases["6_candidate_selection"] = perf_counter() - _t_phase

        # add node and evaluate
        _t_phase = perf_counter()
        t_q0 = perf_counter()
        new_vals = []
        scene_times = [[] for _ in range(S)]
        q_call_cnt = 0  
        for s in range(S):
            t0_q = perf_counter()
            print(f"[Iter {it}] evaluating Q for scenario {s}")
            val, _q_meta = evaluate_Q_at(base_bundles[s], first_vars_list[s], new_node, return_meta=True)
            dt_q = perf_counter() - t0_q
            new_vals.append(val)
            scene_times[s].append(dt_q)
            q_call_cnt += 1
            # --- Q-eval failure/NaN/Inf detection for simplex_debug.txt ---
            _q_term = _q_meta.get('termination_condition', '') if _q_meta else ''
            _q_bad = (_q_term not in ('optimal', 'locallyOptimal'))
            _q_nan = (not math.isfinite(val)) if isinstance(val, float) else False
            if _q_bad or _q_nan:
                _issue = "Q_NaN_Inf" if _q_nan else "Q_solve_fail"
                _iter_debug_lines.append(
                    f"  scen={s} type=Q issue={_issue} "
                    f"ok={_q_meta.get('ok','?') if _q_meta else '?'} "
                    f"status={_q_meta.get('status','?') if _q_meta else '?'} "
                    f"term={_q_term} "
                    f"val={val} "
                    f"time={dt_q:.4f}s\n"
                )

        # ---------------------------------------------------------------
        # Q-eval retry: if any scenario Q is infeasible, retry with a
        # random point inside the same simplex (barycentric sampling).
        # This does NOT change the candidate-selection logic at all —
        # only the post-selection evaluation recovery.
        # ---------------------------------------------------------------
        from bundles import Q_max as _Q_max_val
        MAX_Q_RETRIES = 20

        def _q_is_usable(vals_list):
            """True if ALL scenario Q-values are finite and below Q_max."""
            for v in vals_list:
                if not isinstance(v, (int, float)):
                    return False
                if not math.isfinite(v) or v >= _Q_max_val:
                    return False
            return True

        _q_retry_triggered = False
        _q_retry_count = 0

        if not _q_is_usable(new_vals):
            _q_retry_triggered = True
            # Determine the simplex vertices for random barycentric sampling
            _retry_rec = chosen_cand.get("_rec", lb_simp_rec) if chosen_cand else lb_simp_rec
            _retry_verts = np.array(_retry_rec["verts"], dtype=float)  # (d+1, d)
            _retry_d = _retry_verts.shape[1]  # dimension
            _retry_rng = np.random.default_rng()
            _original_new_node = new_node         # save for fallback
            _original_new_vals = list(new_vals)   # save for fallback

            for _retry_i in range(MAX_Q_RETRIES):
                # Generate uniform random point inside the simplex using
                # barycentric coordinates: sample d Exp(1) values, normalize.
                _bary = _retry_rng.exponential(1.0, size=_retry_d + 1)
                _bary /= _bary.sum()
                _rand_pt = tuple(map(float, _bary @ _retry_verts))

                # Evaluate Q at the random point for all scenarios
                _retry_vals = []
                _retry_ok = True
                for s in range(S):
                    t0_q = perf_counter()
                    val_r, _qm_r = evaluate_Q_at(
                        base_bundles[s], first_vars_list[s], _rand_pt,
                        return_meta=True)
                    dt_q = perf_counter() - t0_q
                    scene_times[s].append(dt_q)
                    q_call_cnt += 1
                    _retry_vals.append(val_r)
                    # Log failures
                    _qt_r = _qm_r.get('termination_condition', '') if _qm_r else ''
                    _qb_r = (_qt_r not in ('optimal', 'locallyOptimal'))
                    _qn_r = (not math.isfinite(val_r)) if isinstance(val_r, float) else False
                    if _qb_r or _qn_r:
                        _issue_r = "Q_NaN_Inf" if _qn_r else "Q_solve_fail"
                        _iter_debug_lines.append(
                            f"  scen={s} type=Q_retry#{_retry_i+1} issue={_issue_r} "
                            f"ok={_qm_r.get('ok','?') if _qm_r else '?'} "
                            f"term={_qt_r} val={val_r} time={dt_q:.4f}s\n"
                        )
                        _retry_ok = False

                _q_retry_count = _retry_i + 1

                if _q_is_usable(_retry_vals):
                    # Success — use the random point instead
                    new_node = _rand_pt
                    new_vals = _retry_vals
                    # Recompute geometric classification for the new point
                    # so subdivision uses the correct loc_type/loc_info.
                    _retry_snapped, _retry_loc_type, _retry_loc_info = \
                        _snap_feature(_rand_pt, _retry_rec)
                    new_node = _retry_snapped
                    if chosen_cand is not None:
                        chosen_cand["loc_type"] = _retry_loc_type
                        chosen_cand["loc_info"] = _retry_loc_info
                    if verbose:
                        print(f"[Iter {it}] Q-eval retry #{_retry_i+1}: SUCCESS. "
                              f"Using random simplex point {_fmt_point(new_node)} "
                              f"(loc_type={_retry_loc_type}).")
                    break
                else:
                    if verbose:
                        print(f"[Iter {it}] Q-eval retry #{_retry_i+1}: "
                              f"still infeasible at {_fmt_point(_rand_pt)}.")
            else:
                # All retries failed — fall back to original values
                # (existing failure behavior: Q_max values propagate as-is)
                new_node = _original_new_node
                new_vals = _original_new_vals
                if verbose:
                    print(f"[Iter {it}] Q-eval retry: all {MAX_Q_RETRIES} retries exhausted. "
                          f"Keeping original point with infeasible Q-values.")

        t_q = perf_counter() - t_q0
        timing["iter_Q_new_time"][timing_idx] = t_q

        iter_q_times_detail.append(scene_times)
        per_iter_q_counts.append(q_call_cnt)

        # === NEW: Print next point details ===
        if verbose and chosen_cand is not None:
            _is_mode2or3 = (chosen_cand.get("pt_source", "").startswith("mode"))
            _has_scene = (chosen_cand.get("scene", -1) >= 0)
            if _is_mode2or3 or _has_scene:
                print(f"\n[Iter {it}] Next Point Details:")
                sid = chosen_cand["simplex_index"]
                rec = chosen_cand["_rec"]
                pt_type = chosen_cand.get("pt_source", "c_s" if use_c_fallback else "ms")
                coords_str = _fmt_point(new_node)

                if _is_mode2or3:
                    # Mode 2: weighted composite — print summary per scenario
                    print(f"  Simplex T{sid}, Type={pt_type}, Point={coords_str}")
                    lambdas = chosen_cand.get("loc_info", {}).get("lambdas", None)
                    if lambdas is not None:
                        vert_idx = rec["vert_idx"]
                        coord_header = "(" + ", ".join(f"x{i+1}" for i in range(len(new_node))) + ")"
                        header = ["Scene", "As", "ms", "As+ms", "Q"]
                        colw = [8, 15, 15, 15, 15]

                        def fmt_row(cols):
                            return "".join(str(c).ljust(w) for c, w in zip(cols, colw))

                        print(fmt_row(header))
                        print("-" * sum(colw))
                        for _s in range(S):
                            q_verts = [scen_values[_s][v] for v in vert_idx]
                            as_val = float(np.dot(lambdas, q_verts))
                            _ms_list_m2 = rec.get("ms_per_scene", [])
                            ms_val = float(_ms_list_m2[_s]) if _s < len(_ms_list_m2) else float('inf')
                            as_plus_ms = as_val + ms_val
                            q_val = float(new_vals[_s])
                            print(fmt_row([_s, f"{as_val:.4e}", f"{ms_val:.4e}",
                                           f"{as_plus_ms:.4e}", f"{q_val:.4e}"]))
                        print()
                else:
                    # Mode 1: single-scene detail
                    scene = chosen_cand["scene"]
                    lambdas = chosen_cand.get("loc_info", {}).get("lambdas", None)
                    if lambdas is not None:
                        vert_idx = rec["vert_idx"]
                        q_verts = [scen_values[scene][v] for v in vert_idx]
                        as_val = float(np.dot(lambdas, q_verts))
                        _ms_list_m1 = rec.get("ms_per_scene", [])
                        ms_val = float(_ms_list_m1[scene]) if scene < len(_ms_list_m1) else float('inf')
                        as_plus_ms = as_val + ms_val
                        q_val = float(new_vals[scene])

                        coord_header = "(" + ", ".join(f"x{i+1}" for i in range(len(new_node))) + ")"
                        header = ["Simplex", "Scene", "Type", "As", "ms", "As+ms", "Q", coord_header]
                        colw = [10, 8, 8, 15, 15, 15, 15, 30]

                        def fmt_row(cols):
                            return "".join(str(c).ljust(w) for c, w in zip(cols, colw))

                        print(fmt_row(header))
                        print("-" * sum(colw))
                        row = [
                            f"T{sid}", scene, pt_type,
                            f"{as_val:.4e}", f"{ms_val:.4e}", f"{as_plus_ms:.4e}", f"{q_val:.4e}",
                            coords_str
                        ]
                        print(fmt_row(row))
                        print()

        # === append node ===
        nodes.append(tuple(map(float, new_node)))
        new_node_index = len(nodes) - 1
        for ω in range(S):
            scen_values[ω].append(new_vals[ω])

        _phases["7_Q_eval_new_node"] = perf_counter() - _t_phase

        # === NEW: update mesh by star-subdividing the simplex that generated new_node ===
        _t_phase = perf_counter()
        if chosen_cand is not None:
            selection_reason_hist.append(chosen_cand.get("pt_source", "unknown")) # Record reason
            sid = int(chosen_cand["simplex_index"])
            loc_type = chosen_cand.get("loc_type", "interior")
            loc_info = chosen_cand.get("loc_info", None)
            
            # LB/UB per user definition: save selected simplex record BEFORE split
            selected_rec_before_split = chosen_cand.get("_rec", None)
            if selected_rec_before_split is None:
                # Try to find it from per_tet
                for r in per_tet:
                    if r["simplex_index"] == sid:
                        selected_rec_before_split = r
                        break
            selected_simplex_id = tuple(sorted(selected_rec_before_split["vert_idx"])) if selected_rec_before_split else None

            # Assign code to split type: 1=interior, 2=edge, 3=face
            split_code = {"interior": 1, "edge": 2, "face": 3}.get(loc_type, 0)
            if verbose:
                print(f"[Iter {it}] subdivision type = {loc_type} "
                      f"(code={split_code}) on simplex T{sid}")

            # LB/UB per user definition: capture n_children from subdivide
            if loc_type == "edge" and loc_info is not None:
                edge_verts = loc_info["edge_verts"]   # Here follows the meaning defined in _snap_feature
                if verbose:
                    print(f"           edge local verts = {edge_verts}")
                n_children = tet_mesh.subdivide_edge(sid, new_node_index, edge_verts)
            elif loc_type == "face" and loc_info is not None:
                face_verts = loc_info["face_verts"]
                if verbose:
                    print(f"           face local verts = {face_verts}")
                n_children = tet_mesh.subdivide_face(sid, new_node_index, face_verts)
            else:
                # Default treated as interior point, star subdivision
                n_children = tet_mesh.subdivide(sid, new_node_index)
        else:
            # Collision fallback: chosen_cand is None but we have edge info
            if _fallback_used:
                selection_reason_hist.append("collision_edge_midpoint")
                sid = _fallback_simplex_idx
                if verbose:
                    print(f"[Iter {it}] subdivision type = edge (collision fallback) "
                          f"on simplex T{sid}, edge={_fallback_edge_verts}")
                n_children = tet_mesh.subdivide_edge(sid, new_node_index, _fallback_edge_verts)
                # Also set selected_rec_before_split for LB split diagnostic
                selected_rec_before_split = lb_simp_rec
                selected_simplex_id = tuple(sorted(lb_simp_rec["vert_idx"]))
            else:
                n_children = 0  # No split happened
                selected_rec_before_split = None
                selected_simplex_id = None

        # Record actual split kind AFTER the subdivide call
        kind = getattr(tet_mesh, "last_split_kind", None)
        if kind is None:
            kind = "none"
        split_kind_hist.append(kind)

        add_node_hist.append(new_node)
        if verbose:
            print(f"[Iter {it}] Elapsed: {perf_counter() - t_iter0:.3f}s")

        # ====================================================================
        # LB/UB per user definition: end-of-iter min-LB and incumbent UB from NEW simplices only
        # ====================================================================
        
        # Track which simplex indices are "new" this iteration
        # LB/UB per user definition: ALWAYS use n_children-based calculation (no it==0 special case)
        # subdivide does pop()+extend(), so new children are at the END of tets list
        n_simplices_after = len(tet_mesh.tets)
        # new_ids = last n_children indices after subdivide
        new_simplex_indices = list(range(n_simplices_after - n_children, n_simplices_after))
        
        _phases["8_mesh_subdivide"] = perf_counter() - _t_phase

        # Re-evaluate per_tet after split
        _t_phase = perf_counter()

        # ---- EAGER POST-SPLIT: evaluate new children, keep old records ----
        pts = np.asarray(nodes, dtype=float)

        # Determine inherited LB for new children
        if _deferred_first_split:
            _inherited_lb = float(_saved_box_LB) if _saved_box_LB is not None else float('inf')
            _inherited_cs_status = "inherited_box"
        else:
            _inherited_lb = float(selected_rec_before_split["LB"]) \
                            if selected_rec_before_split is not None \
                            else float('inf')
            _inherited_cs_status = "inherited_parent"

        # Rebuild per_tet_end for current mesh after split
        per_tet_end = []
        _new_child_indices = []   # simplex indices of newly created children
        _pre_split_by_key = {
            tuple(sorted(r["vert_idx"])): r for r in per_tet
        }
        for k, simp in enumerate(tet_mesh.tets):
            idxs = list(map(int, simp))
            tet_key = tuple(sorted(idxs))
            if tet_key in _pre_split_by_key:
                # Unchanged simplex: deep-copy old record, update simplex_index
                rec = deepcopy(_pre_split_by_key[tet_key])
                rec["simplex_index"] = k
                per_tet_end.append(rec)
            else:
                # New child from this split
                _new_child_indices.append(k)
                verts = [tuple(pts[i]) for i in idxs]
                V_arr = np.array(verts, dtype=float)
                try:
                    vol = simplex_volume(V_arr)
                except Exception:
                    vol = 0.0
                n_v = len(idxs)
                fverts_per_scene_child = [
                    [scen_values[s][i] for i in idxs] for s in range(S)
                ]
                fverts_sum_child = [
                    sum(fverts_per_scene_child[s][j] for s in range(S))
                    for j in range(n_v)
                ]
                per_tet_end.append({
                    "simplex_index": k,
                    "vert_idx": idxs,
                    "verts": verts,
                    "fverts_sum": fverts_sum_child,
                    "ms_per_scene": [float('inf')] * S,
                    "xms_per_scene": [None] * S,
                    "c_per_scene":  [None] * S,
                    "c_point_per_scene": [None] * S,
                    "ms": float('inf'),
                    "c_agg": float('-inf'),
                    "LB": _inherited_lb,
                    "UB": float('inf'),
                    "x_ms_best_scene": None,
                    "best_scene": None,
                    "volume": vol,
                    "n_infeas_verts": 0,
                    "ms_meta_per_scene": [None] * S,
                    "cs_meta_per_scene": [None] * S,
                    "LB_terms": {"lb_case": _inherited_cs_status},
                    "cs_status": _inherited_cs_status,
                    "ms_status": "none",
                    "is_inherited_placeholder": True,
                    "c_fallback_per_scene": list(
                        selected_rec_before_split.get("c_fallback_per_scene", [None] * S)
                    ) if selected_rec_before_split is not None else [None] * S,
                    "c_fallback_source_per_scene": list(
                        selected_rec_before_split.get("c_fallback_source_per_scene", ["parent"] * S)
                    ) if selected_rec_before_split is not None else ["none"] * S,
                    # ---- INACTIVE fields (new children are always active) ----
                    "inactive": False,
                    "inactive_reason": None,
                })

        # EAGER POST-SPLIT: evaluate ONLY the newly created children
        # from this split (identified by _new_child_indices).
        # Old unaffected simplices keep their existing records unchanged.
        _new_child_set = set(_new_child_indices)
        _n_eager_evaluated = 0
        for _child_rec in per_tet_end:
            if _child_rec["simplex_index"] in _new_child_set:
                if verbose:
                    print(f"[Iter {it}] Eager post-split: evaluating child "
                          f"T{_child_rec['simplex_index']} "
                          f"{tuple(sorted(_child_rec['vert_idx']))}")
                ensure_ms_for_simplex(
                    _child_rec, ms_bundles, nodes, scen_values,
                    ms_cache, lb_sur_cache, S,
                    verbose=verbose, iter_num=it,
                    use_fbbt=use_fbbt, use_obbt=use_obbt,
                    obbt_solver_name=obbt_solver_name,
                    obbt_solver_opts=obbt_solver_opts,
                    obbt_tol=obbt_tol,
                    max_obbt_rounds=max_obbt_rounds,
                    _debug_buf=_iter_debug_lines,
                )
                _child_rec.pop("is_inherited_placeholder", None)
                _n_eager_evaluated += 1
        if verbose:
            print(f"[Iter {it}] Eager post-split: "
                  f"evaluated {_n_eager_evaluated} / "
                  f"{len(_new_child_indices)} new children")

        per_tet_dict = {r["simplex_index"]: r for r in per_tet_end}
        # INVARIANT: the LB simplex must be selected from the current active set.
        # Rebuild active mask for the post-split mesh (per_tet_end) using the
        # shared helper so the same gap/quality/shape logic is applied.
        _active_mask_end = _build_active_mask(per_tet_end, UB_global, active_tol, q_cut)
        _active_per_tet_end = [r for r in per_tet_end if _active_mask_end.get(r["simplex_index"], False)]
        LB_global_end = float(min(r["LB"] for r in _active_per_tet_end)) if _active_per_tet_end else float('inf')
        lb_simp_rec_end = min(_active_per_tet_end, key=lambda r: r["LB"]) if _active_per_tet_end else per_tet_end[0]


        # ====== POST-SPLIT DIAGNOSTIC: LB > UB check + dual > primal (ALL simplices) ======
        _tol_gap_debug = 1e-6
        _tol_dual = 1e-6
        _n_lb_gt_ub = 0
        _n_dual_gt_primal = 0
        _lb_gt_ub_lines = []
        _dual_gt_lines = []

        for _rec in per_tet_end:
            # Skip inherited placeholders — LB is valid but MS/CS data is meaningless
            if _rec.get("is_inherited_placeholder"):
                continue
            _rec_lb = float(_rec["LB"])
            _rec_idx = _rec["simplex_index"]
            _rec_vidx = _rec["vert_idx"]

            # --- (G) Per-simplex LB > UB decomposition ---
            if math.isfinite(_rec_lb) and _rec_lb > UB_incumbent + _tol_gap_debug:
                _n_lb_gt_ub += 1
                _lb_gt_ub_lines.append("=" * 70)
                _lb_gt_ub_lines.append(f"[Iter {it}] SIMPLEX LB > UB  —  T{_rec_idx} {tuple(sorted(_rec_vidx))}")
                _lb_gt_ub_lines.append(f"  LB = {_rec_lb:.9e}")
                _lb_gt_ub_lines.append(f"  UB = {UB_incumbent:.9e}")
                _lb_gt_ub_lines.append(f"  LB - UB = {_rec_lb - UB_incumbent:.9e}")
                _lb_gt_ub_lines.append(f"  LB / S = {_rec_lb / S:.9f}")
                _lb_gt_ub_lines.append(f"  UB / S = {UB_incumbent / S:.9f}")
                _lb_gt_ub_lines.append(f"  ms_status = {_rec.get('ms_status', '?')}")
                _lb_gt_ub_lines.append(f"  cs_status = {_rec.get('cs_status', '?')}")
                _lb_gt_ub_lines.append(f"  verts = {_rec.get('verts', '?')}")
                _lb_gt_ub_lines.append(f"  fverts_sum = {_rec.get('fverts_sum', '?')}")
                _lb_gt_ub_lines.append(f"  best_scene = {_rec.get('best_scene', '?')}")
                _lb_gt_ub_lines.append(f"  x_ms_best_scene = {_rec.get('x_ms_best_scene', '?')}")

                # Per-scene ms and c values
                _ms_ps = _rec.get("ms_per_scene", [])
                _c_ps = _rec.get("c_per_scene", [])
                _lb_gt_ub_lines.append(f"  ms_per_scene = {_ms_ps}")
                _lb_gt_ub_lines.append(f"  c_per_scene  = {_c_ps}")

                # LB_terms decomposition
                _lbt = _rec.get("LB_terms", {})
                _lb_gt_ub_lines.append(f"  LB_terms:")
                for _k, _v in _lbt.items():
                    _lb_gt_ub_lines.append(f"    {_k} = {_v}")

                # Per-scenario metadata
                _ms_meta = _rec.get("ms_meta_per_scene", [])
                _cs_meta = _rec.get("cs_meta_per_scene", [])
                for _w in range(len(_ms_ps)):
                    _lb_gt_ub_lines.append(f"  --- scenario {_w} ---")
                    if _w < len(_ms_meta) and _ms_meta[_w] is not None:
                        _lb_gt_ub_lines.append(f"    ms_meta: {_ms_meta[_w]}")
                    else:
                        _lb_gt_ub_lines.append(f"    ms_meta: cached/None")
                    if _w < len(_cs_meta) and _cs_meta[_w] is not None:
                        _lb_gt_ub_lines.append(f"    cs_meta: {_cs_meta[_w]}")
                    else:
                        _lb_gt_ub_lines.append(f"    cs_meta: cached/None")
                _lb_gt_ub_lines.append("")

            # --- (H) Dual > primal check for ALL simplices and ALL scenarios ---
            _ms_meta = _rec.get("ms_meta_per_scene", [])
            _cs_meta = _rec.get("cs_meta_per_scene", [])
            for _w in range(S):
                # MS check
                if _w < len(_ms_meta) and _ms_meta[_w] is not None:
                    _md = _ms_meta[_w].get("dual_bound")
                    _mp = _ms_meta[_w].get("primal_obj")
                    if (_md is not None and _mp is not None
                            and math.isfinite(_md) and math.isfinite(_mp)
                            and _md > _mp + _tol_dual):
                        _n_dual_gt_primal += 1
                        _dual_gt_lines.append(
                            f"[Iter {it}] MS  T{_rec_idx} {tuple(sorted(_rec_vidx))} scen={_w}  "
                            f"dual={_md:.9e} primal={_mp:.9e} diff={_md - _mp:.9e}  "
                            f"status={_ms_meta[_w].get('status','?')} term={_ms_meta[_w].get('termination_condition','?')}"
                        )
                # CS check
                if _w < len(_cs_meta) and _cs_meta[_w] is not None:
                    _cd = _cs_meta[_w].get("dual_bound")
                    _cp = _cs_meta[_w].get("primal_obj")
                    if (_cd is not None and _cp is not None
                            and math.isfinite(_cd) and math.isfinite(_cp)
                            and _cd > _cp + _tol_dual):
                        _n_dual_gt_primal += 1
                        _dual_gt_lines.append(
                            f"[Iter {it}] CS  T{_rec_idx} {tuple(sorted(_rec_vidx))} scen={_w}  "
                            f"dual={_cd:.9e} primal={_cp:.9e} diff={_cd - _cp:.9e}  "
                            f"status={_cs_meta[_w].get('status','?')} term={_cs_meta[_w].get('termination_condition','?')}"
                        )

        # Write diagnostic files
        if _lb_gt_ub_lines:
            with open(dbg_lb_gt_ub_path, "a", encoding="utf-8") as _f:
                _f.write("\n".join(_lb_gt_ub_lines) + "\n\n")
        if _dual_gt_lines:
            with open(dbg_dual_gt_obj_all_path, "a", encoding="utf-8") as _f:
                _f.write("\n".join(_dual_gt_lines) + "\n\n")

        # --- (I) Console warning summary ---
        if _n_lb_gt_ub > 0:
            print(f"[WARN] Iter {it}: {_n_lb_gt_ub} simplex(es) have LB > current UB by more than {_tol_gap_debug}. "
                  f"See {dbg_lb_gt_ub_path}", flush=True)
        if _n_dual_gt_primal > 0:
            print(f"[WARN] Iter {it}: {_n_dual_gt_primal} dual > primal violation(s) across all simplices. "
                  f"See {dbg_dual_gt_obj_all_path}", flush=True)
        # ====== END POST-SPLIT DIAGNOSTIC ======

        # === LB Split Diagnostic: compare parent vs children LBs ===
        if selected_rec_before_split is not None:
            parent_LB = float(selected_rec_before_split["LB"])
            parent_id = tuple(sorted(selected_rec_before_split["vert_idx"]))
            # Previous global min LB (from pre-split evaluation)
            prev_global_min_LB = float(min(r["LB"] for r in per_tet))
            
            # Determine if new global LB is INSIDE (child) or OUTSIDE (old simplex)
            new_lb_simplex_idx = lb_simp_rec_end["simplex_index"]
            if n_children > 0:
                new_lb_is_inside = new_lb_simplex_idx in new_simplex_indices
                location_str = "INSIDE" if new_lb_is_inside else "OUTSIDE"
            else:
                new_lb_is_inside = None
                location_str = "N/A"
            
            _diag_lines = []
            _diag_lines.append("=" * 70)
            _diag_lines.append(f"[Iter {it}] LB SPLIT DIAGNOSTIC")
            _diag_lines.append(f"  Selected (parent) simplex: {parent_id}")
            _diag_lines.append(f"  Parent LB: {parent_LB/S:.9f} (per-scen)")

            # EF solution on this simplex (if available)
            _ef_raw = ef_iter_info.get("ef_obj")
            _ef_recalc = ef_iter_info.get("true_obj")
            if _ef_raw is not None and _ef_recalc is not None:
                _diag_lines.append(
                    f"  EF ef_obj(IPOPT): {_ef_raw/S:.9f}, "
                    f"true_obj(recalc): {_ef_recalc/S:.9f}, "
                    f"used=min={min(_ef_raw, _ef_recalc)/S:.9f} (per-scen)"
                )
            elif _ef_recalc is not None:
                _diag_lines.append(f"  EF true_obj: {_ef_recalc/S:.9f} (per-scen)")
            elif _ef_raw is not None:
                _diag_lines.append(f"  EF ef_obj(IPOPT): {_ef_raw/S:.9f} (per-scen), true_obj: N/A")
            else:
                _diag_lines.append(f"  EF: N/A")

            # Simplex volume: |det([v1-v0, v2-v0, v3-v0])| / 6
            try:
                _sv = np.array(selected_rec_before_split["verts"])
                _vol = abs(_safe_linalg.det((_sv[1:] - _sv[0]))) / 6.0
                _diag_lines.append(f"  Simplex volume: {_vol:.6e}")
            except Exception:
                _diag_lines.append(f"  Simplex volume: N/A")

            _diag_lines.append(f"  Children ({n_children}):")
            
            child_lbs = []
            for new_idx in new_simplex_indices:
                if new_idx in per_tet_dict:
                    child_rec = per_tet_dict[new_idx]
                    child_lb = float(child_rec["LB"])
                    child_id = tuple(sorted(child_rec["vert_idx"]))
                    child_lbs.append(child_lb)
                    _inherited_tag = " [inherited]" if child_rec.get("is_inherited_placeholder") else ""
                    _diag_lines.append(f"    T{new_idx} {child_id}: LB = {child_lb/S:.9f}{_inherited_tag}")
            
            new_global_min_id = tuple(sorted(lb_simp_rec_end["vert_idx"]))
            _diag_lines.append(f"  New global LB: {LB_global_end/S:.9f} (from T{lb_simp_rec_end['simplex_index']} {new_global_min_id})")
            _diag_lines.append(f"  Location: {location_str}")

            # --- Avg CS point Q-value (true objective at mean of c_s solution points) ---
            _cs_pts = lb_simp_rec.get("c_point_per_scene", [])
            _valid_cs_pts = [pt for pt in _cs_pts if pt is not None and all(math.isfinite(v) for v in pt)]
            if _valid_cs_pts:
                _avg_cs_pt = tuple(np.mean(_valid_cs_pts, axis=0))
                _cs_true_val = 0.0
                for s in range(S):
                    _cs_true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], _avg_cs_pt)
                _diag_lines.append(f"  Avg CS point Q-value: {_cs_true_val/S:.9f} (per-scen), point={_avg_cs_pt}")
            else:
                _diag_lines.append(f"  Avg CS point Q-value: N/A (no valid c_s points)")

            # --- Current UB info + simplex provenance ---
            try:
                _ub_ps = f"{UB_incumbent/S:.9f}" if math.isfinite(UB_incumbent) else str(UB_incumbent)
            except Exception:
                _ub_ps = "N/A"
            _diag_lines.append(f"  Current UB: {_ub_ps} (per-scen)")
            _diag_lines.append(f"  UB point: {UB_node}")
            _diag_lines.append(f"  UB source: {ub_source_this_iter if ub_updated_this_iter else ub_source_current}")
            _diag_lines.append(f"  UB simplex: {ub_simplex_id_this_iter if ub_updated_this_iter else ub_simplex_id_current}")

            # --- IPOPT EF solver status ---
            _ipopt_res = _ef_dual_results.get("ipopt", {})
            if _ipopt_res:
                _ip_ok = _ipopt_res.get("ok", False)
                _ip_status = _ipopt_res.get("status", "N/A")
                _ip_term = _ipopt_res.get("term", "N/A")
                _ip_time = _ipopt_res.get("time", 0.0)
                _ip_ef_obj = _ipopt_res.get("ef_obj")
                _ip_true = _ipopt_res.get("true_obj")
                _ip_true_ps = f"{_ip_true/S:.9f}" if _ip_true is not None else "N/A"
                _diag_lines.append(f"  EF-IPOPT: ok={_ip_ok}, status={_ip_status}, term={_ip_term}, "
                                   f"time={_ip_time:.3f}s, ef_obj={_ip_ef_obj}, true_obj(per-scen)={_ip_true_ps}")
            else:
                _diag_lines.append(f"  EF-IPOPT: not available")

            _diag_lines.append("=" * 70)
            
            # Print to console
            if verbose:
                for line in _diag_lines:
                    print(line)
            
            # Write to debug_lb_after_split.txt (append within this run)
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _fdiag:
                _fdiag.write("\n".join(_diag_lines) + "\n\n")
        
        # Save UB_node BEFORE UB candidate processing (for UB_in_split diagnostic)
        _ub_node_at_split = UB_node

        # UB_global_end: generate candidates ONLY from NEW simplices
        # LB/UB per user definition: use incumbent logic (min with previous UB)
        UB_global_end = UB_incumbent  # Start from incumbent (not reset per iteration)
        ub_source_end = ub_source_this_iter if ub_updated_this_iter else ub_source_current
        ub_updated_end = False
        print(f"[DBG] starting UB candidate processing, new_simplex_indices={new_simplex_indices}", flush=True)
        _ub_cand_log = []  # collect (child_idx, type, point, q_value, improved_ub) for ms_cs_detail log
        
        for new_idx in new_simplex_indices:
            if new_idx not in per_tet_dict:
                continue
            rec = per_tet_dict[new_idx]
            # Skip inherited placeholders — no real MS/CS data
            if rec.get("is_inherited_placeholder"):
                continue
            
            # (a) avgMS point: average of per-scenario xms points for this simplex
            xms_list = rec.get("xms_per_scene", [])
            valid_xms = [pt for pt in xms_list if pt is not None and all(math.isfinite(v) for v in pt)]
            if valid_xms:
                avg_ms_pt = tuple(np.mean(valid_xms, axis=0))
                # Evaluate TRUE objective at avg_ms_pt (expected value)
                true_val = 0.0
                for s in range(S):
                    true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], avg_ms_pt)
                
                # LB/UB per user definition: allow incumbent update when true_val < UB
                # If true_val < LB_global_end, emit warning (potential invalid LB) but still update
                if true_val < LB_global_end - 1e-8:
                    print(f"[Iter {it}] WARNING: UB candidate {true_val/S:.6e} < LB {LB_global_end/S:.6e} (potential invalid LB)")
                if true_val < UB_global_end:
                    # DEBUG: mesh membership check for avgMS candidate
                    _ms_mesh = _find_any_simplex_containing_point(avg_ms_pt, per_tet_end)
                    with open(dbg_ub_mesh_path, "a", encoding="utf-8") as _f:
                        _f.write(f"[Iter {it}] UB CANDIDATE: avgMS_new_simplex\n")
                        _f.write(f"  point       = {_fmt_point(avg_ms_pt, 6)}\n")
                        _f.write(f"  obj_value   = {true_val:.9e} (per-scen: {true_val/S:.9f})\n")
                        _f.write(f"  inside_mesh = {_ms_mesh['found']}\n")
                        _f.write(f"  simplex_idx = {_ms_mesh['simplex_index']}\n")
                        _f.write(f"  tol_used    = {_ms_mesh['tol_used']}\n")
                        _f.write(f"  n_checked   = {_ms_mesh['n_checked']}\n\n")
                    UB_global_end = true_val
                    UB_node = avg_ms_pt
                    ub_updated_end = True
                    ub_source_end = "avgMS_new_simplex"
                    ub_simplex_id_this_iter = tuple(sorted(rec["vert_idx"]))
                    # Add to library for monotonicity
                    ub_candidate_library.append((avg_ms_pt, true_val))
                    if verbose:
                        print(f"[Iter {it}] UB candidate from new simplex {new_idx} (avgMS): {true_val/S:.6e} (per scenario)")
                _ub_cand_log.append((new_idx, "avgMS", avg_ms_pt, true_val, true_val < UB_global_end + 1e-12))
            
            # (a2) avgCS point: average of per-scenario c_s solution points for this simplex
            cpts_list = rec.get("c_point_per_scene", [])
            valid_cpts = [pt for pt in cpts_list if pt is not None and all(math.isfinite(v) for v in pt)]
            if valid_cpts:
                avg_cs_pt = tuple(np.mean(valid_cpts, axis=0))
                # Evaluate TRUE objective at avg_cs_pt (expected value)
                cs_true_val = 0.0
                for s in range(S):
                    cs_true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], avg_cs_pt)
                
                if cs_true_val < LB_global_end - 1e-8:
                    print(f"[Iter {it}] WARNING: UB candidate (avgCS) {cs_true_val/S:.6e} < LB {LB_global_end/S:.6e} (potential invalid LB)")
                if cs_true_val < UB_global_end:
                    # DEBUG: mesh membership check for avgCS candidate
                    _cs_mesh = _find_any_simplex_containing_point(avg_cs_pt, per_tet_end)
                    with open(dbg_ub_mesh_path, "a", encoding="utf-8") as _f:
                        _f.write(f"[Iter {it}] UB CANDIDATE: avgCS_new_simplex\n")
                        _f.write(f"  point       = {_fmt_point(avg_cs_pt, 6)}\n")
                        _f.write(f"  obj_value   = {cs_true_val:.9e} (per-scen: {cs_true_val/S:.9f})\n")
                        _f.write(f"  inside_mesh = {_cs_mesh['found']}\n")
                        _f.write(f"  simplex_idx = {_cs_mesh['simplex_index']}\n")
                        _f.write(f"  tol_used    = {_cs_mesh['tol_used']}\n")
                        _f.write(f"  n_checked   = {_cs_mesh['n_checked']}\n\n")
                    UB_global_end = cs_true_val
                    UB_node = avg_cs_pt
                    ub_updated_end = True
                    ub_source_end = "avgCS_new_simplex"
                    ub_simplex_id_this_iter = tuple(sorted(rec["vert_idx"]))
                    ub_candidate_library.append((avg_cs_pt, cs_true_val))
                    if verbose:
                        print(f"[Iter {it}] UB candidate from new simplex {new_idx} (avgCS): {cs_true_val/S:.6e} (per scenario)")
                _ub_cand_log.append((new_idx, "avgCS", avg_cs_pt, cs_true_val, cs_true_val < UB_global_end + 1e-12))
            
            # (b) EF candidate: use true_obj (recalculated Q-value) for UB.
            #     ef_obj is the raw IPOPT objective which can be a relaxed/local
            #     value lower than the actual Q-value — NOT valid as a UB candidate.
        if ef_iter_info.get("ef_ok"):
            _ef_recalc = ef_iter_info.get("true_obj")
            if _ef_recalc is not None and math.isfinite(_ef_recalc):
                if _ef_recalc < UB_global_end:
                    # DEBUG: mesh membership check for EF candidate
                    _ef_pt = ef_iter_info["K_ef"]
                    if _ef_pt is not None:
                        _ef_mesh = _find_any_simplex_containing_point(_ef_pt, per_tet_end)
                        with open(dbg_ub_mesh_path, "a", encoding="utf-8") as _f:
                            _f.write(f"[Iter {it}] UB CANDIDATE: EF\n")
                            _f.write(f"  point       = {_fmt_point(_ef_pt, 6)}\n")
                            _f.write(f"  obj_value   = {_ef_recalc:.9e} (per-scen: {_ef_recalc/S:.9f})\n")
                            _f.write(f"  inside_mesh = {_ef_mesh['found']}\n")
                            _f.write(f"  simplex_idx = {_ef_mesh['simplex_index']}\n")
                            _f.write(f"  tol_used    = {_ef_mesh['tol_used']}\n")
                            _f.write(f"  n_checked   = {_ef_mesh['n_checked']}\n\n")
                    UB_global_end = _ef_recalc
                    UB_node = ef_iter_info["K_ef"]
                    ub_updated_end = True
                    ub_source_end = "EF"
                    ub_simplex_id_this_iter = selected_simplex_id
                    if verbose:
                        print(f"[Iter {it}] UB_global_end improved by EF: {_ef_recalc/S:.6e} (per-scen), "
                              f"ef_obj={ef_iter_info.get('ef_obj')}, true_obj={_ef_recalc}")
        
        # Provenance logging uses end-of-iter incumbent UB only
        # Update provenance if UB changed this iteration
        if ub_updated_end:
            ub_updated_this_iter = True
            ub_source_current = ub_source_end
            ub_origin_iter_current = it
            ub_simplex_id_current = ub_simplex_id_this_iter
            iter_logger.update_ub_provenance(
                updated=True,
                source=ub_source_end,
                simplex_id=ub_simplex_id_this_iter,
                origin_iter=it
            )
        else:
            iter_logger.update_ub_provenance(updated=False)
        
        print(f"[DBG] LB/UB override section, LB_global_end={LB_global_end}, UB_global_end={UB_global_end}", flush=True)
        # === OVERRIDE LB_hist/UB_hist with end-of-iteration values ===
        # Capture pre-split values BEFORE override (used by split log below)
        _lb_pre_split = float(LB_hist[-1]) if LB_hist else float('nan')
        _ub_pre_split = float(UB_hist[-1]) if UB_hist else float('nan')
        # These will be used for both CSV and summary table
        if LB_hist:
            LB_hist[-1] = LB_global_end
        if UB_hist:
            UB_hist[-1] = UB_global_end
        
        # Also update the current-iteration LB_global/UB_global for consistency
        LB_global = LB_global_end
        UB_global = UB_global_end
        lb_simp_rec = lb_simp_rec_end
        
        # FIX-A: update incumbent UB value AND point together
        UB_incumbent = UB_global_end
        UB_node_incumbent = UB_node  # UB_node was already updated by whichever source improved it
        
        # ====== DEBUG: end-of-iteration UB mesh membership check ======
        try:
            if UB_node is not None and per_tet_end:
                _ub_mesh_result = _find_any_simplex_containing_point(UB_node, per_tet_end)
                _ub_mesh_lines = []
                _ub_mesh_lines.append(f"{'='*60}")
                _ub_mesh_lines.append(f"[Iter {it}] END-OF-ITERATION UB MESH MEMBERSHIP CHECK")
                _ub_mesh_lines.append(f"  UB_global_end  = {UB_global_end:.9e} (per-scen: {UB_global_end/S:.9f})")
                _ub_mesh_lines.append(f"  UB_node        = {_fmt_point(UB_node, 6)}")
                _ub_mesh_lines.append(f"  ub_source      = {ub_source_end if ub_updated_end else ub_source_current}")
                _ub_mesh_lines.append(f"  inside_mesh    = {_ub_mesh_result['found']}")
                _ub_mesh_lines.append(f"  simplex_index  = {_ub_mesh_result['simplex_index']}")
                _ub_mesh_lines.append(f"  tol_used       = {_ub_mesh_result['tol_used']}")
                _ub_mesh_lines.append(f"  n_simplices    = {_ub_mesh_result['n_checked']}")
                if _ub_mesh_result['lambdas'] is not None:
                    _ub_mesh_lines.append(f"  lambdas        = {_ub_mesh_result['lambdas']}")
                _ub_mesh_lines.append("")
                with open(dbg_ub_mesh_path, "a", encoding="utf-8") as _f:
                    _f.write("\n".join(_ub_mesh_lines) + "\n")
                
                if not _ub_mesh_result['found']:
                    print(f"[WARN] Iter {it}: UB_node is OUTSIDE the current simplex mesh/partition. "
                          f"See {dbg_ub_mesh_path}", flush=True)
        except Exception as _e:
            print(f"[WARN] Iter {it}: UB mesh membership check failed: {_e}", flush=True)
        # ====== END DEBUG ======

        # ====================================================================
        # End of LB/UB per user definition
        # ====================================================================

        _phases["9_end_of_iter_reeval"] = perf_counter() - _t_phase

        # === Reset deferred-first-split flag after iter 1 completes ===
        if _deferred_first_split:
            _deferred_first_split = False

        # ================================================================
        # (A) simplex_record_split.txt — append per-iteration split record
        # ================================================================
        try:
            _sel_lb_pre = float(selected_rec_before_split["LB"]) if selected_rec_before_split else float('nan')
            _sel_sid = int(selected_rec_before_split["simplex_index"]) if selected_rec_before_split else "N/A"
            _split_lines = []
            _split_lines.append(f"=== Iter {it} ===\n")
            _split_lines.append(f"selected_simplex: T{_sel_sid}  LB_pre={_sel_lb_pre/S:.9f}\n")
            # Log vertex coordinates of the selected simplex
            if selected_rec_before_split is not None:
                _sel_verts = selected_rec_before_split.get("verts", [])
                _sel_vidxs = selected_rec_before_split.get("vert_idx", [])
                for _vi, (_vidx, _vc) in enumerate(zip(_sel_vidxs, _sel_verts)):
                    _vc_str = _fmt_point(_vc)
                    _split_lines.append(f"  v{_vidx}: {_vc_str}\n")
            _split_lines.append(f"UB_pre={_ub_pre_split/S:.9f}  LB_global_pre={_lb_pre_split/S:.9f}\n")
            _split_lines.append(f"children (created {n_children}):\n")
            for _ci in new_simplex_indices:
                if _ci in per_tet_dict:
                    _cr = per_tet_dict[_ci]
                    _cid = tuple(sorted(_cr["vert_idx"]))
                    _c_sum = sum(float(x) for x in _cr.get("c_per_scene", []) if x is not None and np.isfinite(float(x)))
                    _c_vol = float(_cr.get("volume", 0.0))
                    _split_lines.append(f"  child T{_ci} {_cid} LB={float(_cr['LB'])/S:.9f} sum_cs={_c_sum/S:.9f} vol={_c_vol:.6e}\n")
            # Log next-node coordinates and barycentric coordinates
            _nn_str = _fmt_point(new_node) if new_node is not None else "None"
            _split_lines.append(f"next_node: {_nn_str}\n")
            # Selection strategy info
            try:
                _sel_src = _selection_source if _selection_source is not None else "N/A"
                _sel_rank = _selection_rank if _selection_rank is not None else "N/A"
                _sel_loc = chosen_cand.get("loc_type", "N/A") if chosen_cand is not None else "N/A"
                _split_lines.append(f"selection: source={_sel_src}  rank={_sel_rank}  snap={_sel_loc}\n")
            except (NameError, Exception):
                _split_lines.append("selection: N/A\n")
            # Log None-point counts
            try:
                _split_lines.append(f"candidates: ms_valid={len(ms_cands)}  ms_none={_ms_none_count}  cs_valid={len(cs_cands)}  cs_none={_cs_none_count}\n")
            except (NameError, Exception):
                pass
            # Barycentric coordinates
            _bary_str = "N/A"
            if new_node is not None and chosen_cand is not None:
                _bary_lambdas = chosen_cand.get("loc_info", {}).get("lambdas", None) if chosen_cand.get("loc_info") else None
                if _bary_lambdas is not None:
                    _bary_str = "(" + ", ".join(f"{float(l):.6f}" for l in _bary_lambdas) + ")"
                else:
                    # Compute barycentric coords from simplex vertices
                    try:
                        _nn_rec = chosen_cand.get("_rec", selected_rec_before_split)
                        if _nn_rec is not None:
                            _nn_verts = np.array(_nn_rec["verts"], dtype=float)
                            _nn_pt = np.array(new_node, dtype=float)
                            # Solve V^T * lambda = pt, with sum(lambda)=1
                            _d = _nn_verts.shape[1]
                            _A = np.vstack([_nn_verts.T, np.ones((1, _d + 1))])
                            _b = np.append(_nn_pt, 1.0)
                            _lam = _safe_linalg.solve(_A, _b)
                            _bary_str = "(" + ", ".join(f"{float(l):.6f}" for l in _lam) + ")"
                    except Exception:
                        _bary_str = "error"
            _split_lines.append(f"barycentric: {_bary_str}\n")
            _split_lines.append(f"sample_retry_triggered={'yes' if _q_retry_triggered else 'no'}  retry_count={_q_retry_count}\n")
            # ---- FBBT/OBBT per-child tightening report ----
            _n_children = len(new_simplex_indices)
            _n_child_inactive = 0
            _total_checks = 0
            _total_infeasible = 0
            _child_detail_lines = []
            for _ci in new_simplex_indices:
                if _ci in per_tet_dict:
                    _cr = per_tet_dict[_ci]
                    _cr_inactive = _cr.get("inactive", False)
                    _cr_reason = _cr.get("inactive_reason", "")
                    if _cr_inactive:
                        _n_child_inactive += 1
                        # Parse which scenario triggered: "tightening_infeasible_scene_X"
                        _infeas_scene = -1
                        if "scene_" in _cr_reason:
                            try:
                                _infeas_scene = int(_cr_reason.split("scene_")[-1])
                            except (ValueError, IndexError):
                                _infeas_scene = -1
                        # Checks performed = scenes 0.._infeas_scene (inclusive)
                        _checks_this = _infeas_scene + 1 if _infeas_scene >= 0 else S
                        _total_checks += _checks_this
                        _total_infeasible += 1
                        _child_detail_lines.append(
                            f"  child T{_ci}: INACTIVE  "
                            f"infeasible_at_scene={_infeas_scene}  "
                            f"checks={_checks_this}/{S}\n")
                    else:
                        _total_checks += S
                        _child_detail_lines.append(
                            f"  child T{_ci}: active  checks={S}/{S}\n")
            _split_lines.append(
                f"fbbt_obbt_tightening: {_n_children} children, "
                f"{_total_checks} checks, "
                f"{_n_child_inactive} inactive, "
                f"{_n_children - _n_child_inactive} active\n")
            _split_lines.extend(_child_detail_lines)
            # Also show cumulative inactive across entire mesh
            _cum_inactive = sum(1 for r in per_tet_end if r.get("inactive", False))
            _split_lines.append(
                f"cumulative_inactive: {_cum_inactive}/{len(per_tet_end)} simplices\n")
            _split_lines.append(f"end: UB={UB_global_end/S:.9f}  LB={LB_global_end/S:.9f}\n\n")
            with open(split_log_path, "a", encoding="utf-8") as _fsplit:
                _fsplit.writelines(_split_lines)
        except Exception as _e_split_log:
            # FIX-B: Print full traceback to console AND write to persistent crash file
            import traceback as _tb_split
            _tb_str = _tb_split.format_exc()
            print(f"[WARNING] split log write failed at iter {it}: {_e_split_log}")
            print(_tb_str)
            try:
                _crash_file = os.path.join(_csv_output_dir, "split_log_crash.txt")
                with open(_crash_file, "a", encoding="utf-8") as _fc:
                    _fc.write(f"=== Iter {it} ===\n")
                    _sel_id = "N/A"
                    try:
                        _sel_id = int(selected_rec_before_split["simplex_index"]) if selected_rec_before_split else "N/A"
                    except Exception:
                        pass
                    _fc.write(f"simplex_id: {_sel_id}\n")
                    _fc.write(f"error: {_e_split_log}\n")
                    _fc.write(_tb_str)
                    _fc.write("\n")
            except Exception:
                pass  # crash file write itself must never crash the algorithm

        # ================================================================
        # (A2) simplex_record_ms_cs_detail.txt — per-scenario ms/cs values and points
        # ================================================================
        try:
            _mscs_lines = []
            _mscs_lines.append(f"=== Iter {it} ===")
            _sel_rec = selected_rec_before_split if selected_rec_before_split else lb_simp_rec
            _sel_vidxs = _sel_rec.get("vert_idx", [])
            _sel_verts = np.array(_sel_rec.get("verts", []), dtype=float)
            _sel_sid = int(_sel_rec.get("simplex_index", -1))
            _cs_st = _sel_rec.get("cs_status", "?")
            _ms_st = _sel_rec.get("ms_status", "?")
            _mscs_lines.append(f"simplex: T{_sel_sid}  verts={tuple(sorted(_sel_vidxs))}  cs_status={_cs_st}  ms_status={_ms_st}")

            # Helper: compute barycentric coordinates of point p in simplex V
            def _compute_bary(p, V):
                """p: (d,) array, V: (d+1, d) array. Returns (d+1,) barycentric coords."""
                try:
                    d = V.shape[1]
                    T = (V[1:] - V[0]).T  # (d, d)
                    lam_rest = _safe_linalg.solve(T, p - V[0])
                    lam0 = 1.0 - lam_rest.sum()
                    return np.concatenate([[lam0], lam_rest])
                except Exception:
                    return None

            def _fmt_pt(pt):
                if pt is None:
                    return "None"
                return "(" + ", ".join(f"{float(c):.6f}" for c in pt) + ")"

            def _fmt_bary(bary):
                if bary is None:
                    return "N/A"
                return "(" + ", ".join(f"{float(l):.6f}" for l in bary) + ")"

            # --- MS per scenario ---
            _ms_vals = _sel_rec.get("ms_per_scene", [])
            _ms_pts = _sel_rec.get("xms_per_scene", [])
            _mscs_lines.append(f"")
            _mscs_lines.append(f"  MS per scenario ({len(_ms_vals)} scenarios):")
            for _s_idx in range(len(_ms_vals)):
                _mv = float(_ms_vals[_s_idx]) if _s_idx < len(_ms_vals) else float('nan')
                _mp = _ms_pts[_s_idx] if _s_idx < len(_ms_pts) else None
                if _mp is not None:
                    _mp_arr = np.asarray(_mp, dtype=float)
                    _mb = _compute_bary(_mp_arr, _sel_verts)
                else:
                    _mb = None
                _mscs_lines.append(
                    f"    s={_s_idx}  ms={_mv:.6e}  pt={_fmt_pt(_mp)}  bary={_fmt_bary(_mb)}"
                )

            # --- CS per scenario ---
            _cs_vals = _sel_rec.get("c_per_scene", [])
            _cs_pts = _sel_rec.get("c_point_per_scene", [])
            _mscs_lines.append(f"")
            _mscs_lines.append(f"  CS per scenario ({len(_cs_vals)} scenarios):")
            for _s_idx in range(len(_cs_vals)):
                _cv = float(_cs_vals[_s_idx]) if _s_idx < len(_cs_vals) else float('nan')
                _cp = _cs_pts[_s_idx] if _s_idx < len(_cs_pts) else None
                if _cp is not None:
                    _cp_arr = np.asarray(_cp, dtype=float)
                    _cb = _compute_bary(_cp_arr, _sel_verts)
                else:
                    _cb = None
                _mscs_lines.append(
                    f"    s={_s_idx}  cs={_cv:.6e}  pt={_fmt_pt(_cp)}  bary={_fmt_bary(_cb)}"
                )

            # --- UB candidates: avgMS and avgCS points with Q-values ---
            _mscs_lines.append(f"  UB candidates (avg MS/CS points from new children):")
            if _ub_cand_log:
                for _uc_idx, _uc_type, _uc_pt, _uc_qval, _uc_improved in _ub_cand_log:
                    _uc_pt_str = _fmt_pt(_uc_pt)
                    _uc_q_per_scen = _uc_qval / S if S > 0 else float('nan')
                    _uc_tag = "  <-- BEST" if _uc_improved else ""
                    _mscs_lines.append(
                        f"    child T{_uc_idx}  type={_uc_type}  pt={_uc_pt_str}  "
                        f"Q_sum={_uc_qval:.6e}  Q_per_scen={_uc_q_per_scen:.9f}{_uc_tag}"
                    )
            else:
                _mscs_lines.append(f"    (none)")
            _mscs_lines.append(f"  current UB (per-scen): {UB_global_end/S:.9f}")

            _mscs_lines.append(f"")
            with open(ms_cs_detail_path, "a", encoding="utf-8") as _fmscs:
                _fmscs.write("\n".join(_mscs_lines) + "\n")
        except Exception as _e_mscs_log:
            print(f"[WARNING] ms/cs detail log write failed: {_e_mscs_log}")

        # ================================================================
        # (B) simplex_record_subproblem_runtime.txt — append per-iteration
        # ================================================================
        try:
            # --- MS timing: extract from per_tet_end metadata (post-split) ---
            _ms_total_time = 0.0
            _ms_total_calls = 0
            for _r in per_tet_end:
                for _ms_m in (_r.get("ms_meta_per_scene") or []):
                    if _ms_m is not None and _ms_m.get("time_sec") is not None:
                        _ms_total_time += float(_ms_m["time_sec"])
                        _ms_total_calls += 1
            _ms_avg = _ms_total_time / _ms_total_calls if _ms_total_calls > 0 else 0.0

            # --- CS timing: extract from per_tet_end metadata (post-split) ---
            _cs_total_time = 0.0
            _cs_total_calls = 0
            for _r in per_tet_end:
                for _cs_m in (_r.get("cs_meta_per_scene") or []):
                    if _cs_m is not None and _cs_m.get("time_sec") is not None:
                        _cs_total_time += float(_cs_m["time_sec"])
                        _cs_total_calls += 1
            _cs_avg = _cs_total_time / _cs_total_calls if _cs_total_calls > 0 else 0.0

            # --- Q-eval timing ---
            _q_total_time = sum(sum(st) for st in scene_times) if scene_times else 0.0
            _q_count = q_call_cnt
            _q_avg = _q_total_time / _q_count if _q_count > 0 else 0.0

            # --- Build compact per-iteration runtime block ---
            _rt_lines = []
            _rt_lines.append(f"=== Iter {it} ===\n")
            _rt_lines.append(f"ms: total_time={_ms_total_time:.4f}, calls={_ms_total_calls}, avg={_ms_avg:.4f}\n")
            _rt_lines.append(f"cs: total_time={_cs_total_time:.4f}, calls={_cs_total_calls}, avg={_cs_avg:.4f}\n")
            _rt_lines.append(f"Q:  total_time={_q_total_time:.4f}, calls={_q_count}, avg={_q_avg:.4f}\n")
            # Filter to timeout-only events
            _timeout_lines = [
                ln for ln in _iter_nonopt_buf
                if any(kw in ln.lower() for kw in ("time_limit", "maxtimelimit", "timelimit"))
            ] if _iter_nonopt_buf else []
            if _timeout_lines:
                _rt_lines.append(f"timeouts ({len(_timeout_lines)}):\n")
                _rt_lines.extend(_timeout_lines)
            _rt_lines.append(f"\n")
            with open(runtime_log_path, "a", encoding="utf-8") as _frt:
                _frt.writelines(_rt_lines)
        except Exception as _e_rt_log:
            print(f"[WARNING] runtime log write failed: {_e_rt_log}")

        # ================================================================
        # (C) simplex_debug.txt — append only if issues detected
        # ================================================================
        if _iter_debug_lines:
            try:
                with open(debug_path, "a", encoding="utf-8") as _fdbg:
                    _fdbg.write(f"=== Iter {it} ===\n")
                    _fdbg.writelines(_iter_debug_lines)
                    _fdbg.flush()
            except Exception as _e_dbg:
                print(f"[WARNING] debug log write failed: {_e_dbg}")

        t_iter = perf_counter() - t_iter0
        _phases["TOTAL"] = t_iter
        _iter_phase_times.append(_phases)

        # === Print detailed timing summary for this iteration ===
        if verbose:
            print(f"\n{'='*70}")
            print(f"[TIMING] === Iter {it} phase breakdown (total {t_iter:.2f}s) ===")
            for _pname, _ptime in _phases.items():
                if _pname == "TOTAL":
                    continue
                pct = _ptime / t_iter * 100 if t_iter > 0 else 0
                bar = '#' * int(pct / 2)
                print(f"  {_pname:30s} : {_ptime:10.2f}s  ({pct:5.1f}%)  {bar}")
            print(f"  {'TOTAL':30s} : {t_iter:10.2f}s")
            # Count cache hits vs fresh solves
            n_tets_iter = len(per_tet)
            n_fresh_ms = sum(per_scene_calls)
            n_total_possible = n_tets_iter * S
            n_cache_hits = n_total_possible - n_fresh_ms
            print(f"  #simplices={n_tets_iter}, #scenarios={S}, total_ms_pairs={n_total_possible}")
            print(f"  fresh ms solves={n_fresh_ms}, cache hits={n_cache_hits}")
            print(f"{'='*70}\n")
        timing["iter_total_time"][timing_idx] = t_iter 
        
        # === Update best_lb_ever / best_ub_ever AFTER end-of-iter ===
        # LB/UB per user definition: update monotonic envelopes using end-of-iter values
        best_lb_ever = max(best_lb_ever, LB_global_end)
        best_ub_ever = min(best_ub_ever, UB_global_end)
        
        print(f"[DBG] about to write CSV, LB_global_end={LB_global_end}, UB_global_end={UB_global_end}", flush=True)
        # === Append current iteration to CSV (AFTER end-of-iter update) ===
        # LB/UB per user definition: use end-of-iter LB/UB (same as summary table)
        iter_time = iter_time_hist[-1]
        n_nodes = node_count[-1]
        lb_val = LB_global_end / S  # end-of-iteration LB
        ub_val = UB_global_end / S  # end-of-iteration UB
        abs_gap = (UB_global_end - LB_global_end) / S
        rel_gap = abs_gap / (abs(ub_val) + 1e-16)
        # Additional monotonic envelope columns
        lb_ever = best_lb_ever / S
        ub_ever = best_ub_ever / S
        # === Determine UB_in_split: is UB_node inside the split simplex? ===
        # Use _ub_node_at_split (saved BEFORE UB candidate processing) so we
        # test the UB point that existed at split time, not one that may have
        # been replaced by a post-split UB candidate.
        _ub_in_split_str = "N/A"
        try:
            if selected_rec_before_split is not None and _ub_node_at_split is not None:
                _split_V = np.array(selected_rec_before_split["verts"], dtype=float)  # (4,3)
                _ub_pt = np.asarray(_ub_node_at_split, dtype=float)
                _T = (_split_V[1:] - _split_V[0]).T  # (3,3)
                _lam123 = _safe_linalg.solve(_T, _ub_pt - _split_V[0])
                _lam0 = 1.0 - _lam123.sum()
                _lam_all = np.array([_lam0, *_lam123])
                _ub_in_split_str = "INSIDE" if np.all(_lam_all >= -1e-8) else "OUTSIDE"
        except (NameError, Exception):
            _ub_in_split_str = "N/A"

        # LB_in_split: reuse location_str from split diagnostic (INSIDE/OUTSIDE)
        # location_str is only defined when selected_rec_before_split is not None AND
        # the LB split diagnostic block runs; use a safe lookup.
        _lb_in_split_str = "N/A"
        try:
            _lb_in_split_str = location_str  # type: ignore[possibly-undefined]
        except NameError:
            pass

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # --- cs_status_choosen: CS failure count of selected simplex before split ---
            _cs_chosen_str = "none"
            if selected_rec_before_split is not None:
                _cs_chosen_str = _format_cs_failure_status(
                    selected_rec_before_split.get("c_per_scene"))
            # --- cs_status_all: CS failure count of each new child simplex ---
            _cs_all_parts = []
            try:
                for _child_rec in per_tet_end:
                    if _child_rec["simplex_index"] in _new_child_set:
                        _cs_all_parts.append(
                            _format_cs_failure_status(
                                _child_rec.get("c_per_scene")))
            except (NameError, Exception):
                pass
            _cs_all_str = "; ".join(_cs_all_parts) if _cs_all_parts else "none"

            # --- cs_fallback: how many failed CS scenarios were recovered by fallback ---
            _cs_fb_total_failed = 0
            _cs_fb_total_recovered = 0
            try:
                for _child_rec in per_tet_end:
                    if _child_rec["simplex_index"] in _new_child_set:
                        _native = _child_rec.get("c_per_scene", [])
                        _fb = _child_rec.get("c_fallback_per_scene")
                        for _si in range(len(_native)):
                            _nv = _native[_si]
                            if _nv is None or not math.isfinite(_nv):
                                _cs_fb_total_failed += 1
                                if (_fb is not None and _si < len(_fb)
                                        and _fb[_si] is not None
                                        and math.isfinite(_fb[_si])):
                                    _cs_fb_total_recovered += 1
            except (NameError, Exception):
                pass
            _cs_fb_str = f"{_cs_fb_total_recovered}/{_cs_fb_total_failed}" if _cs_fb_total_failed > 0 else "none"

            # --- UB_status: did EF solve succeed and produce a new UB candidate? ---
            if not ef_iter_info.get("ef_attempted", False):
                _ub_status_str = "not_attempted"
            elif ef_iter_info.get("ef_ok", False):
                # EF solved successfully; check if it actually improved UB
                _ub_status_str = "ok_improved" if ub_source_end == "EF" else "ok_not_best"
            else:
                _ub_status_str = "failed"

            writer.writerow([f"{iter_time:.3f}", n_nodes, f"{lb_val:.9f}", f"{ub_val:.9f}", f"{rel_gap*100:.7f}%", f"{abs_gap:.5f}", f"{lb_ever:.9f}", f"{ub_ever:.9f}", _lb_in_split_str, _ub_in_split_str, _ub_status_str, _cs_chosen_str, _cs_all_str, _cs_fb_str])

        # === Write EF solver details to debug_ef_solver.txt ===
        try:
            with open(dbg_ef_solver_path, "a", encoding="utf-8") as _fef:
                _fef.write(f"--- [Iter {it}] ---\n")
                for _stag in ["ipopt", "gurobi"]:
                    _sres = _ef_dual_results.get(_stag, {})
                    _s_ok = _sres.get("ok", False)
                    _s_true = _sres.get("true_obj")
                    _s_true_ps = f"{_s_true/S:.9f}" if _s_true is not None else "N/A"
                    _lb_str = ""
                    if _sres.get("lower_bound") is not None:
                        _lb_str = f", gurobi_LB={_sres['lower_bound']:.9f}"
                    _fef.write(
                        f"  [{_stag:6s}] ok={_s_ok}, "
                        f"status={_sres.get('status')}, term={_sres.get('term')}, "
                        f"time={_sres.get('time', 0.0):.3f}s, "
                        f"ef_obj={_sres.get('ef_obj')}, "
                        f"true_obj(per-scen)={_s_true_ps}{_lb_str}\n"
                    )
                _winner = ef_iter_info.get('solver_status', 'none').split(':')[0] if ef_iter_info.get('ef_ok') else 'none'
                _fef.write(
                    f"  [winner] {_winner}, ub_updated={ef_iter_info.get('ub_updated_by_ef', False)}, "
                    f"UB_incumbent={UB_incumbent/S:.9f}, "
                    f"LB_global_end={LB_global_end/S:.9f}, "
                    f"UB_global_end={UB_global_end/S:.9f}\n\n"
                )
        except Exception:
            pass  # logging must never crash the algorithm

        # === Convergence stopping condition (AFTER end-of-iter) ===
        if gap_stop_tol is not None and float(gap_stop_tol) > 0.0:
            gap_rel_end = float(UB_global_end - LB_global_end) / (abs(UB_global_end) + 1e-16)
            if gap_rel_end <= float(gap_stop_tol):
                termination_reason = "gap_converged"
                if verbose:
                    print(f"[Iter {it}] Stop: UB-LB gap {gap_rel_end:.6e} <= tol {float(gap_stop_tol):.6e}.")
                break

        # === Time limit stopping condition ===
        if time_limit is not None and time_limit > 0:
            elapsed = perf_counter() - t_start
            if elapsed >= time_limit:
                termination_reason = "time_limit"
                if verbose:
                    print(f"[Iter {it}] Stop: Time limit reached ({elapsed:.2f}s >= {time_limit:.2f}s).")
                break

        # === Per-iteration diagnostic logging ===
        try:
            # EF info: no EF solve in this code path
            ef_info = {
                "EF_attempted": ef_iter_info.get("ef_attempted", False),
                "EF_enabled": enable_ef_ub,
                "time_sec": ef_iter_info.get("ef_time_sec"),
                "status": ef_iter_info.get("solver_status"),
                "termination_condition": ef_iter_info.get("termination_condition"),
                "solver_status": ef_iter_info.get("solver_status"),
                "used_for_UB": ef_iter_info.get("ub_updated_by_ef", False),
            }

            # UB info
            ub_info = {"updated_this_iter": ub_updated_this_iter}

            # LB simplex tracking
            # LB/UB per user definition: use pre-saved selected_simplex_id (saved BEFORE split)
            # selected_simplex_id: the simplex chosen for branching/splitting this iteration (before split)
            # Note: selected_simplex_id was saved when chosen_cand was processed, before the split
            lb_selected_simplex_id = selected_simplex_id if 'selected_simplex_id' in dir() else None
            
            # best_simplex_id_before_split: the branching simplex (same as selected)
            lb_best_before_split = lb_selected_simplex_id
            
            # best_simplex_id_after_split: computed from lb_simp_rec_end (argmin LB after split/rebuild)
            lb_best_after_split = tuple(sorted(lb_simp_rec_end["vert_idx"])) if lb_simp_rec_end else None
            
            # Check if after-split best LB simplex is a child of the selected (parent) simplex
            if n_children > 0 and lb_simp_rec_end is not None:
                stays_in_selected = (lb_simp_rec_end["simplex_index"] in new_simplex_indices)
            else:
                stays_in_selected = None
            
            lb_info = {
                "selected_simplex_id": lb_selected_simplex_id,
                "best_simplex_id_before_split": lb_best_before_split,
                "best_simplex_id_after_split": lb_best_after_split,
                "stays_in_selected": stays_in_selected,
            }

            # MS/CS fallback aggregation for the END-OF-ITER global-LB simplex
            # Use metadata stored in lb_simp_rec_end (post-split argmin LB)
            ms_status_summary = {}
            ms_fallback_scenarios = []
            ms_fallback_reason_counts = {}
            cs_status_summary = {}
            cs_fallback_scenarios = []
            cs_fallback_reason_counts = {}

            ms_meta_list = lb_simp_rec_end.get("ms_meta_per_scene", []) if lb_simp_rec_end else []
            cs_meta_list = lb_simp_rec_end.get("cs_meta_per_scene", []) if lb_simp_rec_end else []

            for s in range(len(ms_meta_list)):
                # MS metadata
                ms_meta = ms_meta_list[s]
                if ms_meta:
                    st = ms_meta.get("status", "unknown")
                    ms_status_summary[st] = ms_status_summary.get(st, 0) + 1
                    if ms_meta.get("used_fallback", False):
                        ms_fallback_scenarios.append(s)
                        reason = ms_meta.get("fallback_reason", "other")
                        ms_fallback_reason_counts[reason] = ms_fallback_reason_counts.get(reason, 0) + 1
                else:
                    ms_status_summary["cached"] = ms_status_summary.get("cached", 0) + 1

            for s in range(len(cs_meta_list)):
                # CS metadata
                cs_meta = cs_meta_list[s]
                if cs_meta:
                    st = cs_meta.get("status", "unknown")
                    cs_status_summary[st] = cs_status_summary.get(st, 0) + 1
                    if cs_meta.get("used_fallback", False):
                        cs_fallback_scenarios.append(s)
                        reason = cs_meta.get("fallback_reason", "other")
                        cs_fallback_reason_counts[reason] = cs_fallback_reason_counts.get(reason, 0) + 1
                else:
                    cs_status_summary["cached"] = cs_status_summary.get("cached", 0) + 1

            ms_agg = {
                "status_summary": ms_status_summary,
                "fallback_any": len(ms_fallback_scenarios) > 0,
                "fallback_count": len(ms_fallback_scenarios),
                "fallback_scenarios": ms_fallback_scenarios,
                "fallback_reason_counts": ms_fallback_reason_counts,
            }

            cs_agg = {
                "status_summary": cs_status_summary,
                "fallback_any": len(cs_fallback_scenarios) > 0,
                "fallback_count": len(cs_fallback_scenarios),
                "fallback_scenarios": cs_fallback_scenarios,
                "fallback_reason_counts": cs_fallback_reason_counts,
            }

            # Log this iteration
            iter_logger.log_iteration(it, ef_info, ub_info, lb_info, ms_agg, cs_agg)

        except Exception as e:
            # Logging must never crash the algorithm
            if verbose:
                print(f"[Iter {it}] Warning: diagnostic logging failed: {e}")

        it += 1

    # === Final summary table (like Gurobi log) ===
    if verbose and LB_hist:
        print("\n===== Simplex search summary (per-scenario) =====")
        header = (
            f"{'Time (s)':>10} "
            f"{'# Nodes':>8} "
            f"{'LB':>14} "
            f"{'UB':>14} "
            f"{'Rel. Gap':>10} "
            f"{'Abs. Gap':>10} "
            f"{'#simplex':>10} "
            f"{'#active':>10}"
            f"{'Split':>8} "
            f"{'Selection':>15}"
        )
        print(header)
        print("-" * len(header))

        for k in range(len(LB_hist)):
            t_k   = iter_time_hist[k]
            n_k   = node_count[k]
            # Divide LB, UB, and Abs. Gap by number of scenarios
            lb_k  = LB_hist[k] / S
            ub_k  = UB_hist[k] / S
            gap_abs = (UB_hist[k] - LB_hist[k]) / S  # Per-scenario gap
            gap_rel = gap_abs / (abs(ub_k) + 1e-16)
            nsimp = simplex_hist[k]
            nact  = active_simplex_hist[k]
            split_kind = split_kind_hist[k] if k < len(split_kind_hist) else "none"
            reason = selection_reason_hist[k] if k < len(selection_reason_hist) else "N/A"

            print(
                f"{t_k:>10.3f} "
                f"{n_k:>8d} "
                f"{lb_k:>14.9f} "
                f"{ub_k:>14.9f} "
                f"{gap_rel*100:>9.4f}% "
                f"{gap_abs:>10.5f} "
                f"{nsimp:>10d} "
                f"{nact:>10d}"
                f"{split_kind:>8} "
                f"{reason:>15}"
            )



    # === Close iteration logger ===
    iter_logger.close()

    return {
        "nodes": np.array(nodes, float),
        "LB_hist": LB_hist,
        "UB_hist": UB_hist,
        "ms_hist": ms_hist,
        "ms_b_hist": ms_b_hist,
        "node_count": node_count,
        "UB_node_hist": UB_node_hist,
        "added_nodes": add_node_hist,
        "active_ratio_hist": active_ratio_hist,
        "timing": timing,
        "ms_ub_active_per_iter": ms_ub_active_per_iter,
        "iter_q_times_detail": iter_q_times_detail,
        "per_iter_q_counts": per_iter_q_counts,
        "c_hist": c_hist_per_iter,
        "lb_c_agg_hist": lb_c_agg_hist,           
        "lb_c_per_scene_hist": lb_c_per_scene_hist, 
        "iter_ms_times_detail": iter_ms_times_detail,
        "per_iter_ms_counts": per_iter_ms_counts,
        "termination_reason": termination_reason,

    }

# Backward-compatible alias: general callers can use run_simplex
run_simplex = run_pid_simplex_3d
