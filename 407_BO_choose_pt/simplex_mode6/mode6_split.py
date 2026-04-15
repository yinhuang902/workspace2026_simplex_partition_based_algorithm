"""
mode6_split.py — Cluster-Aware Split-Point Selection for Simplex Method.

This module implements the 'mode6' split-point rule, which:
  1. Clusters the scenario-wise c_s points in the current simplex.
  2. Detects whether cluster centers lie near a common edge, face, or interior.
  3. Samples candidate split points continuously in the chosen domain.
  4. Scores candidates based on cluster separation, centrality, child quality,
     and collision avoidance.
  5. Returns the best candidate or None (with a failure reason).

Designed for correctness, transparency, and debuggability over speed.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from simplex_geometry import (
    barycentric_coordinates,
    simplex_quality,
    simplex_volume,
)

# ---------------------------------------------------------------------------
#  Default parameters (all configurable via function arguments)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    cluster_dist_frac=0.15,     # clustering threshold = frac * diameter
    bary_tol_edge=0.05,         # barycentric "small" threshold for edge
    bary_tol_face=0.05,         # barycentric "small" threshold for face
    majority_frac=0.60,         # min weight fraction for majority domain
    n_samples_interior=2000,
    n_samples_face=500,
    n_samples_edge=100,         # structured grid
    eps_interior=0.05,          # min barycentric coord for interior samples
    eps_face=0.08,              # min barycentric coord for face samples
    eps_edge=0.05,              # edge parameter stay-away
    w_separation=0.40,
    w_centrality=0.25,
    w_quality=0.25,
    w_collision=0.10,
    centroid_bias_frac=0.30,    # fallback: push avg_cs toward centroid
)


# ===================================================================
#  A.  Clustering of c_s points
# ===================================================================

def cluster_cs_points(
    cs_points: np.ndarray,
    simplex_verts: np.ndarray,
    dist_threshold_frac: float = _DEFAULTS["cluster_dist_frac"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hierarchical agglomerative clustering of c_s points.

    Parameters
    ----------
    cs_points : (N, d) array of valid c_s scenario points.
    simplex_verts : (d+1, d) array of simplex vertex coordinates.
    dist_threshold_frac : fraction of simplex diameter used as
        the distance threshold for clustering.

    Returns
    -------
    labels : (N,) int array — cluster label per c_s point (1-indexed).
    sizes  : (K,) int array — number of points in each cluster.
    centers : (K, d) array  — centroid of each cluster.
    """
    N = len(cs_points)
    if N <= 1:
        labels = np.ones(N, dtype=int)
        sizes = np.array([N], dtype=int)
        centers = cs_points.copy().reshape(-1, cs_points.shape[1]) if N == 1 else np.empty((0, simplex_verts.shape[1]))
        return labels, sizes, centers

    # compute simplex diameter (max pairwise distance among vertices)
    n_v = len(simplex_verts)
    diam = 0.0
    for i in range(n_v):
        for j in range(i + 1, n_v):
            d = float(np.linalg.norm(simplex_verts[i] - simplex_verts[j]))
            diam = max(diam, d)
    if diam < 1e-15:
        diam = 1e-15  # degenerate simplex guard

    threshold = dist_threshold_frac * diam

    # scipy fclusterdata: single-linkage, distance criterion
    labels = fclusterdata(
        cs_points,
        t=max(threshold, 1e-15),
        criterion="distance",
        method="single",
        metric="euclidean",
    )
    labels = labels.astype(int)

    unique_labels = np.unique(labels)
    K = len(unique_labels)
    d = cs_points.shape[1]
    sizes = np.zeros(K, dtype=int)
    centers = np.zeros((K, d), dtype=float)

    for k_idx, lab in enumerate(unique_labels):
        mask = labels == lab
        sizes[k_idx] = int(mask.sum())
        centers[k_idx] = cs_points[mask].mean(axis=0)

    # re-label to 0-indexed for internal use
    label_map = {lab: k_idx for k_idx, lab in enumerate(unique_labels)}
    labels_0 = np.array([label_map[l] for l in labels], dtype=int)

    return labels_0, sizes, centers


# ===================================================================
#  B.  Edge / Face / Interior detection (majority-weighted)
# ===================================================================

def _classify_point_bary(lambdas, tol_edge, tol_face, n_verts):
    """Classify a point by its barycentric coordinates.

    Returns
    -------
    kind : str — "edge", "face", "interior", or "vertex"
    support : frozenset of int — indices of the 'big' barycentric coords
    """
    small = lambdas < tol_edge  # use the tighter tolerance for edge check
    n_small = int(small.sum())
    n_big = n_verts - n_small

    if n_big <= 1:
        return "vertex", frozenset(np.where(~small)[0])
    if n_big == 2:
        return "edge", frozenset(np.where(~small)[0])

    # Check face: exactly 1 small coord (for face tolerance)
    small_face = lambdas < tol_face
    n_small_face = int(small_face.sum())
    if n_small_face == 1:
        return "face", frozenset(np.where(~small_face)[0])

    return "interior", frozenset(range(n_verts))


def detect_search_domain(
    cluster_centers: np.ndarray,
    cluster_sizes: np.ndarray,
    simplex_verts: np.ndarray,
    vert_idx: List[int],
    bary_tol_edge: float = _DEFAULTS["bary_tol_edge"],
    bary_tol_face: float = _DEFAULTS["bary_tol_face"],
    majority_frac: float = _DEFAULTS["majority_frac"],
) -> Tuple[str, Optional[Tuple[int, ...]], dict]:
    """Detect whether clusters concentrate on an edge, face, or interior.

    Uses a **majority-weighted** rule: if cluster centers representing
    >= majority_frac of total cluster weight are near the same edge/face,
    that mode is selected.

    Parameters
    ----------
    cluster_centers : (K, d) array.
    cluster_sizes   : (K,) int array.
    simplex_verts   : (d+1, d) array.
    vert_idx        : list of int, global node indices of simplex vertices.
    bary_tol_edge   : threshold for "near edge" classification.
    bary_tol_face   : threshold for "near face" classification.
    majority_frac   : minimum weight fraction for majority rule.

    Returns
    -------
    mode         : "edge", "face", or "interior".
    feature_verts : tuple of global vertex indices (for edge/face), or None.
    diagnostics  : dict with per-center classification details.
    """
    K, d = cluster_centers.shape
    n_verts = len(simplex_verts)
    total_weight = float(cluster_sizes.sum())
    if total_weight <= 0:
        total_weight = 1.0

    # Classify each cluster center
    per_center = []
    for k in range(K):
        lam = barycentric_coordinates(cluster_centers[k], simplex_verts)
        kind, support = _classify_point_bary(lam, bary_tol_edge, bary_tol_face, n_verts)
        per_center.append({
            "cluster_idx": k,
            "lambdas": lam,
            "kind": kind,
            "support": support,
            "weight": int(cluster_sizes[k]),
        })

    # --- Try edge mode ---
    # Collect weight per edge (support frozen set of size 2)
    edge_weights = {}  # frozenset -> total weight
    for pc in per_center:
        if pc["kind"] == "edge":
            key = pc["support"]
            edge_weights[key] = edge_weights.get(key, 0) + pc["weight"]

    best_edge = None
    best_edge_weight = 0.0
    for edge_support, w in edge_weights.items():
        if w > best_edge_weight:
            best_edge_weight = w
            best_edge = edge_support

    if best_edge is not None and (best_edge_weight / total_weight) >= majority_frac:
        local_indices = sorted(best_edge)
        feature_verts_global = tuple(vert_idx[i] for i in local_indices)
        return "edge", feature_verts_global, {
            "per_center": per_center,
            "edge_local_indices": local_indices,
            "edge_weight_frac": best_edge_weight / total_weight,
        }

    # --- Try face mode ---
    face_weights = {}  # frozenset -> total weight
    for pc in per_center:
        if pc["kind"] == "face":
            key = pc["support"]
            face_weights[key] = face_weights.get(key, 0) + pc["weight"]
        elif pc["kind"] == "edge":
            # An edge point is also on multiple faces; check which faces
            # contain this edge. A face has n_verts-1 vertices. An edge
            # (2 vertices) is on any face that contains both.
            for face_combo in combinations(range(n_verts), n_verts - 1):
                if pc["support"].issubset(frozenset(face_combo)):
                    key = frozenset(face_combo)
                    face_weights[key] = face_weights.get(key, 0) + pc["weight"]

    best_face = None
    best_face_weight = 0.0
    for face_support, w in face_weights.items():
        if w > best_face_weight:
            best_face_weight = w
            best_face = face_support

    if best_face is not None and (best_face_weight / total_weight) >= majority_frac:
        local_indices = sorted(best_face)
        feature_verts_global = tuple(vert_idx[i] for i in local_indices)
        return "face", feature_verts_global, {
            "per_center": per_center,
            "face_local_indices": local_indices,
            "face_weight_frac": best_face_weight / total_weight,
        }

    # --- Interior mode ---
    return "interior", None, {
        "per_center": per_center,
        "reason": "no_majority_edge_or_face",
    }


# ===================================================================
#  C.  Continuous search-domain sampling
# ===================================================================

def sample_search_domain(
    mode: str,
    simplex_verts: np.ndarray,
    feature_verts_global: Optional[Tuple[int, ...]],
    vert_idx: List[int],
    rng: np.random.Generator | None = None,
    n_samples_interior: int = _DEFAULTS["n_samples_interior"],
    n_samples_face: int = _DEFAULTS["n_samples_face"],
    n_samples_edge: int = _DEFAULTS["n_samples_edge"],
    eps_interior: float = _DEFAULTS["eps_interior"],
    eps_face: float = _DEFAULTS["eps_face"],
    eps_edge: float = _DEFAULTS["eps_edge"],
) -> np.ndarray:
    """Sample candidate split points in the chosen domain.

    Parameters
    ----------
    mode : "edge", "face", or "interior".
    simplex_verts : (d+1, d) array.
    feature_verts_global : global vertex indices for the edge/face.
    vert_idx : global vertex indices of the simplex.
    rng : numpy random generator (or None for default).

    Returns
    -------
    candidates : (M, d) array of sampled points.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_verts, dim = simplex_verts.shape

    if mode == "edge":
        # Structured 1D grid on the edge
        if feature_verts_global is None or len(feature_verts_global) < 2:
            return np.empty((0, dim))

        # Map global indices to local simplex indices
        global_to_local = {g: i for i, g in enumerate(vert_idx)}
        try:
            li_a = global_to_local[feature_verts_global[0]]
            li_b = global_to_local[feature_verts_global[1]]
        except KeyError:
            return np.empty((0, dim))

        va = simplex_verts[li_a]
        vb = simplex_verts[li_b]

        ts = np.linspace(eps_edge, 1.0 - eps_edge, n_samples_edge)
        candidates = np.array([(1.0 - t) * va + t * vb for t in ts])
        return candidates

    elif mode == "face":
        # Sample barycentric coordinates on the sub-face
        if feature_verts_global is None:
            return np.empty((0, dim))

        global_to_local = {g: i for i, g in enumerate(vert_idx)}
        local_face = []
        for g in feature_verts_global:
            if g not in global_to_local:
                return np.empty((0, dim))
            local_face.append(global_to_local[g])

        face_verts = simplex_verts[local_face]  # (n_face, d)
        n_face = len(face_verts)

        # Over-sample then filter
        n_over = int(n_samples_face * 3)
        alphas = rng.dirichlet(np.ones(n_face), size=n_over)
        # reject if any coord < eps_face
        valid = np.all(alphas >= eps_face, axis=1)
        alphas = alphas[valid]
        if len(alphas) > n_samples_face:
            alphas = alphas[:n_samples_face]

        if len(alphas) == 0:
            return np.empty((0, dim))

        candidates = alphas @ face_verts
        return candidates

    else:  # interior
        # Sample barycentric coordinates for the full simplex
        n_over = int(n_samples_interior * 3)
        alphas = rng.dirichlet(np.ones(n_verts), size=n_over)
        valid = np.all(alphas >= eps_interior, axis=1)
        alphas = alphas[valid]
        if len(alphas) > n_samples_interior:
            alphas = alphas[:n_samples_interior]

        if len(alphas) == 0:
            return np.empty((0, dim))

        candidates = alphas @ simplex_verts
        return candidates


# ===================================================================
#  D.  Scoring function
# ===================================================================

def _assign_points_to_children(points, split_pt, simplex_verts, vert_idx):
    """Given a star-split at split_pt, assign each point to a child simplex.

    Star subdivision replaces each vertex in turn with split_pt,
    creating (d+1) children.  For each point, we find which child
    contains it (by barycentric-coordinate check).

    Returns
    -------
    assignments : (N,) int array — child index (0..d) or -1 if outside all.
    """
    n_verts = len(simplex_verts)
    N = len(points)
    assignments = np.full(N, -1, dtype=int)

    # Build child vertex arrays
    children = []
    for k in range(n_verts):
        child_verts = simplex_verts.copy()
        child_verts[k] = split_pt
        children.append(child_verts)

    for i, pt in enumerate(points):
        for k, child_v in enumerate(children):
            try:
                lam = barycentric_coordinates(pt, child_v)
                if np.all(lam >= -1e-8):
                    assignments[i] = k
                    break
            except Exception:
                continue

    return assignments


def _child_simplex_qualities(split_pt, simplex_verts):
    """Compute quality for each child of a star split.

    Returns list of quality values (one per child = d+1 children).
    """
    n_verts = len(simplex_verts)
    qualities = []
    for k in range(n_verts):
        child_verts = simplex_verts.copy()
        child_verts[k] = split_pt
        try:
            q = simplex_quality(child_verts)
        except Exception:
            q = 0.0
        qualities.append(q)
    return qualities


def score_candidate(
    x: np.ndarray,
    simplex_verts: np.ndarray,
    vert_idx: List[int],
    cluster_labels: np.ndarray,
    cluster_sizes: np.ndarray,
    cluster_centers: np.ndarray,
    all_nodes: np.ndarray,
    min_dist: float,
    mode: str,
    w_separation: float = _DEFAULTS["w_separation"],
    w_centrality: float = _DEFAULTS["w_centrality"],
    w_quality: float = _DEFAULTS["w_quality"],
    w_collision: float = _DEFAULTS["w_collision"],
) -> Tuple[float, dict]:
    """Score a candidate split point.

    Scoring is based on cluster_centers and cluster_sizes (not raw c_s
    points).  The raw c_s array is only needed for dispersion diagnostics
    and is handled separately in compute_dispersion_metrics().

    Returns
    -------
    total_score : float  (negative means hard-reject).
    breakdown   : dict with per-component scores and info.
    """
    breakdown = {}

    # --- Collision check (consistent with global min_dist) ---
    dists_to_nodes = np.linalg.norm(all_nodes - x, axis=1)
    nearest_idx = int(np.argmin(dists_to_nodes))
    nearest_dist = float(dists_to_nodes[nearest_idx])

    # Hard reject if below min_dist (same threshold as rest of algorithm)
    if nearest_dist < min_dist:
        breakdown["collision"] = {
            "nearest_idx": nearest_idx,
            "nearest_dist": nearest_dist,
            "hard_reject": True,
        }
        return -1.0, breakdown

    # Smooth collision sub-score: 0 at min_dist, 1 at 2*min_dist
    collision_score = min(1.0, nearest_dist / (2.0 * min_dist))
    breakdown["collision"] = {
        "nearest_idx": nearest_idx,
        "nearest_dist": nearest_dist,
        "hard_reject": False,
        "score": collision_score,
    }

    # --- Centrality: min barycentric coordinate ---
    try:
        lam = barycentric_coordinates(x, simplex_verts)
        centrality_score = float(np.clip(np.min(lam), 0.0, 1.0))
    except Exception:
        centrality_score = 0.0
        lam = None

    # For edge/face mode, recalculate centrality within the sub-domain
    if mode == "edge" and lam is not None:
        sorted_lam = np.sort(lam)[::-1]
        if len(sorted_lam) >= 2:
            centrality_score = float(min(sorted_lam[0], sorted_lam[1]))
            centrality_score = min(centrality_score, 0.5)  # max for edge midpoint
            centrality_score *= 2.0  # scale to [0, 1]
    elif mode == "face" and lam is not None:
        n_verts = len(simplex_verts)
        # For a face with n_verts-1 vertices, the "small" coord is the one
        # opposing the face. Centrality = min of the face-vertex coords.
        sorted_lam = np.sort(lam)[::-1]
        face_lams = sorted_lam[:n_verts - 1]
        centrality_score = float(np.min(face_lams)) if len(face_lams) > 0 else 0.0
        centrality_score = min(centrality_score * (n_verts - 1), 1.0)

    breakdown["centrality"] = {"score": centrality_score, "lambdas": lam}

    # --- Cluster separation (size-weighted) ---
    K = len(cluster_sizes)
    if K <= 1:
        # Only one cluster — separation is trivially 1.0
        separation_score = 1.0
    else:
        # Assign cluster centers to children
        child_assignments = _assign_points_to_children(
            cluster_centers, x, simplex_verts, vert_idx
        )

        # Size-weighted pairwise separation
        total_pair_weight = 0.0
        separated_weight = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                w = float(cluster_sizes[i]) * float(cluster_sizes[j])
                total_pair_weight += w
                if child_assignments[i] != child_assignments[j]:
                    separated_weight += w
                elif child_assignments[i] == -1 or child_assignments[j] == -1:
                    # Unassigned — don't count as separated or same
                    total_pair_weight -= w  # undo

        if total_pair_weight > 0:
            separation_score = separated_weight / total_pair_weight
        else:
            separation_score = 0.0

    breakdown["separation"] = {
        "score": separation_score,
        "n_clusters": K,
    }

    # --- Child quality (robust: 0.7*mean + 0.3*min) ---
    qualities = _child_simplex_qualities(x, simplex_verts)
    if qualities:
        q_mean = float(np.mean(qualities))
        q_min = float(np.min(qualities))
        quality_score = 0.7 * q_mean + 0.3 * q_min
    else:
        quality_score = 0.0

    breakdown["quality"] = {
        "score": quality_score,
        "child_qualities": qualities,
    }

    # --- Weighted total ---
    total = (
        w_separation * separation_score
        + w_centrality * centrality_score
        + w_quality * quality_score
        + w_collision * collision_score
    )
    breakdown["total"] = total

    return total, breakdown


# ===================================================================
#  E.  Dispersion diagnostics
# ===================================================================

def compute_dispersion_metrics(
    cs_points: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict:
    """Compute dispersion metrics for anti-false-convergence diagnostics.

    Uses all c_s points for raw metrics and cluster centers for cluster metrics.

    Returns
    -------
    dict with:
        max_pairwise_distance    : float (all c_s)
        variance_trace           : float (all c_s)
        sum_sq_dist_to_centers   : float (per-cluster sum)
        cluster_center_spread    : float (max pairwise among centers)
    """
    result = {}

    N = len(cs_points)

    # --- All-c_s metrics ---
    if N >= 2:
        max_pw = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.linalg.norm(cs_points[i] - cs_points[j]))
                max_pw = max(max_pw, d)
        result["max_pairwise_distance"] = max_pw

        mean_pt = cs_points.mean(axis=0)
        diff = cs_points - mean_pt
        result["variance_trace"] = float(np.sum(diff ** 2) / N)
    else:
        result["max_pairwise_distance"] = 0.0
        result["variance_trace"] = 0.0

    # --- Per-cluster metrics ---
    if len(cluster_centers) > 0 and N > 0:
        ssq = 0.0
        for i in range(N):
            c = cluster_centers[cluster_labels[i]]
            ssq += float(np.sum((cs_points[i] - c) ** 2))
        result["sum_sq_dist_to_centers"] = ssq
    else:
        result["sum_sq_dist_to_centers"] = 0.0

    # --- Cluster center spread ---
    K = len(cluster_centers)
    if K >= 2:
        max_cc = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                d = float(np.linalg.norm(cluster_centers[i] - cluster_centers[j]))
                max_cc = max(max_cc, d)
        result["cluster_center_spread"] = max_cc
    else:
        result["cluster_center_spread"] = 0.0

    return result


# ===================================================================
#  F.  Top-level entry point
# ===================================================================

def mode6_select_split_point(
    lb_simp_rec: dict,
    all_nodes: list,
    min_dist: float,
    S: int,
    verbose: bool = True,
    rng: np.random.Generator | None = None,
    # clustering params
    cluster_dist_frac: float = _DEFAULTS["cluster_dist_frac"],
    # domain detection params
    bary_tol_edge: float = _DEFAULTS["bary_tol_edge"],
    bary_tol_face: float = _DEFAULTS["bary_tol_face"],
    majority_frac: float = _DEFAULTS["majority_frac"],
    # sampling params
    n_samples_interior: int = _DEFAULTS["n_samples_interior"],
    n_samples_face: int = _DEFAULTS["n_samples_face"],
    n_samples_edge: int = _DEFAULTS["n_samples_edge"],
    eps_interior: float = _DEFAULTS["eps_interior"],
    eps_face: float = _DEFAULTS["eps_face"],
    eps_edge: float = _DEFAULTS["eps_edge"],
    # scoring params
    w_separation: float = _DEFAULTS["w_separation"],
    w_centrality: float = _DEFAULTS["w_centrality"],
    w_quality: float = _DEFAULTS["w_quality"],
    w_collision: float = _DEFAULTS["w_collision"],
) -> dict:
    """Select a split point using the cluster-aware mode6 rule.

    Parameters
    ----------
    lb_simp_rec : per_tet record for the LB-best simplex.
    all_nodes   : list of all existing node coordinates.
    min_dist    : minimum allowed distance between nodes.
    S           : number of scenarios.
    verbose     : if True, print progress messages.

    Returns
    -------
    dict with keys:
        split_point      : tuple or None
        loc_type         : str ("edge"/"face"/"interior")
        loc_info         : dict (edge_verts / face_verts / lambdas)
        score            : float
        score_breakdown  : dict
        search_mode      : str ("edge"/"face"/"interior")
        reason           : str or None (failure reason)
        diagnostics      : dict (full logging data)
    """
    if rng is None:
        rng = np.random.default_rng()

    diagnostics = {}
    simplex_idx = int(lb_simp_rec["simplex_index"])
    simplex_verts = np.asarray(lb_simp_rec["verts"], dtype=float)
    vert_idx = list(lb_simp_rec["vert_idx"])
    all_nodes_arr = np.asarray(all_nodes, dtype=float)

    if verbose:
        print(f"[mode6] === Simplex T{simplex_idx} ===")
        print(f"[mode6]   verts = {vert_idx}")
        for i, v in enumerate(simplex_verts):
            print(f"[mode6]   v{i} (node #{vert_idx[i]}): {tuple(map(float, v))}")

    # ---- Step 1: collect valid c_s points ----
    cs_pts_raw = lb_simp_rec.get("c_point_per_scene", [])
    valid_indices = []
    valid_pts = []
    for s, pt in enumerate(cs_pts_raw):
        if pt is not None:
            arr = np.asarray(pt, float)
            if np.all(np.isfinite(arr)):
                valid_indices.append(s)
                valid_pts.append(arr)

    diagnostics["n_valid_cs"] = len(valid_pts)
    diagnostics["n_total_cs"] = len(cs_pts_raw)
    diagnostics["valid_cs_scenarios"] = valid_indices

    if verbose:
        print(f"[mode6]   valid c_s points: {len(valid_pts)}/{len(cs_pts_raw)}")
        for i, (si, pt) in enumerate(zip(valid_indices, valid_pts)):
            print(f"[mode6]   c_s[scen {si}] = {tuple(map(float, pt))}")

    if len(valid_pts) < 1:
        if verbose:
            print("[mode6]   FAIL: no valid c_s points")
        return {
            "split_point": None,
            "loc_type": "interior",
            "loc_info": None,
            "score": -1.0,
            "score_breakdown": {},
            "search_mode": "interior",
            "reason": "no_valid_cs_points",
            "diagnostics": diagnostics,
        }

    cs_points = np.array(valid_pts, dtype=float)

    # ---- Step 2: cluster c_s points ----
    labels, sizes, centers = cluster_cs_points(
        cs_points, simplex_verts, dist_threshold_frac=cluster_dist_frac
    )

    diagnostics["cluster_labels"] = labels.tolist()
    diagnostics["cluster_sizes"] = sizes.tolist()
    diagnostics["cluster_centers"] = [tuple(map(float, c)) for c in centers]

    if verbose:
        K = len(sizes)
        print(f"[mode6]   clusters: K={K}, sizes={sizes.tolist()}")
        for k in range(K):
            print(f"[mode6]   cluster {k}: center={tuple(map(float, centers[k]))}, size={sizes[k]}")

    # ---- Step 3: compute dispersion metrics ----
    dispersion = compute_dispersion_metrics(cs_points, centers, labels)
    diagnostics["dispersion"] = dispersion
    if verbose:
        print(f"[mode6]   dispersion: max_pw_dist={dispersion['max_pairwise_distance']:.6e}, "
              f"var_trace={dispersion['variance_trace']:.6e}, "
              f"cluster_spread={dispersion['cluster_center_spread']:.6e}")

    # ---- Step 4: detect search domain ----
    search_mode, feature_verts, domain_diag = detect_search_domain(
        centers, sizes, simplex_verts, vert_idx,
        bary_tol_edge=bary_tol_edge,
        bary_tol_face=bary_tol_face,
        majority_frac=majority_frac,
    )
    diagnostics["search_mode"] = search_mode
    diagnostics["feature_verts"] = feature_verts
    # Store domain diagnostics without lambdas (not JSON-serializable as-is)
    _domain_diag_safe = {}
    for dk, dv in domain_diag.items():
        if dk == "per_center":
            _domain_diag_safe["per_center"] = [
                {k2: (v2.tolist() if hasattr(v2, 'tolist') else
                      (list(v2) if isinstance(v2, frozenset) else v2))
                 for k2, v2 in pc.items()}
                for pc in dv
            ]
        else:
            _domain_diag_safe[dk] = dv
    diagnostics["domain_detection"] = _domain_diag_safe

    if verbose:
        print(f"[mode6]   search_mode = {search_mode}")
        if feature_verts is not None:
            print(f"[mode6]   feature_verts (global) = {feature_verts}")
        if "edge_weight_frac" in domain_diag:
            print(f"[mode6]   edge weight frac = {domain_diag['edge_weight_frac']:.3f}")
        if "face_weight_frac" in domain_diag:
            print(f"[mode6]   face weight frac = {domain_diag['face_weight_frac']:.3f}")

    # ---- Step 5: sample candidate points ----
    candidates = sample_search_domain(
        mode=search_mode,
        simplex_verts=simplex_verts,
        feature_verts_global=feature_verts,
        vert_idx=vert_idx,
        rng=rng,
        n_samples_interior=n_samples_interior,
        n_samples_face=n_samples_face,
        n_samples_edge=n_samples_edge,
        eps_interior=eps_interior,
        eps_face=eps_face,
        eps_edge=eps_edge,
    )

    diagnostics["n_candidates_sampled"] = len(candidates)
    if verbose:
        print(f"[mode6]   sampled {len(candidates)} candidate points (mode={search_mode})")

    if len(candidates) == 0:
        if verbose:
            print("[mode6]   FAIL: no candidate points generated")
        return {
            "split_point": None,
            "loc_type": search_mode,
            "loc_info": None,
            "score": -1.0,
            "score_breakdown": {},
            "search_mode": search_mode,
            "reason": "no_candidates_sampled",
            "diagnostics": diagnostics,
        }

    # ---- Step 6: score all candidates, pick the best ----
    best_score = -2.0
    best_idx = -1
    best_breakdown = {}
    n_rejected = 0

    for ci, x in enumerate(candidates):
        sc, bd = score_candidate(
            x=x,
            simplex_verts=simplex_verts,
            vert_idx=vert_idx,
            cluster_labels=labels,
            cluster_sizes=sizes,
            cluster_centers=centers,
            all_nodes=all_nodes_arr,
            min_dist=min_dist,
            mode=search_mode,
            w_separation=w_separation,
            w_centrality=w_centrality,
            w_quality=w_quality,
            w_collision=w_collision,
        )
        if sc < 0:
            n_rejected += 1
            continue
        if sc > best_score:
            best_score = sc
            best_idx = ci
            best_breakdown = bd

    diagnostics["n_rejected"] = n_rejected
    diagnostics["n_valid_scored"] = len(candidates) - n_rejected

    if verbose:
        print(f"[mode6]   scored: {len(candidates) - n_rejected} valid, {n_rejected} rejected (collision)")

    if best_idx < 0:
        if verbose:
            print("[mode6]   FAIL: all candidates rejected (collision)")
        return {
            "split_point": None,
            "loc_type": search_mode,
            "loc_info": None,
            "score": -1.0,
            "score_breakdown": {},
            "search_mode": search_mode,
            "reason": "all_candidates_rejected",
            "diagnostics": diagnostics,
        }

    best_point = candidates[best_idx]
    best_pt_tuple = tuple(map(float, best_point))

    # Compute loc_type and loc_info using barycentric coordinates
    try:
        lam = barycentric_coordinates(best_point, simplex_verts)
    except Exception:
        lam = None

    if search_mode == "edge" and feature_verts is not None:
        loc_type = "edge"
        loc_info = {"edge_verts": feature_verts, "lambdas": lam}
    elif search_mode == "face" and feature_verts is not None:
        loc_type = "face"
        loc_info = {"face_verts": feature_verts, "lambdas": lam}
    else:
        loc_type = "interior"
        loc_info = {"lambdas": lam}

    # --- nearest-node details for logging ---
    dists_all = np.linalg.norm(all_nodes_arr - best_point, axis=1)
    nn_idx = int(np.argmin(dists_all))
    nn_dist = float(dists_all[nn_idx])
    nn_is_vert = nn_idx in vert_idx

    diagnostics["winning_point"] = best_pt_tuple
    diagnostics["winning_score"] = best_score
    diagnostics["winning_breakdown"] = {
        k: (v if not isinstance(v, np.ndarray) else v.tolist())
        for k, v in best_breakdown.items()
        if k != "quality"  # qualities can be verbose
    }
    diagnostics["winning_child_qualities"] = best_breakdown.get("quality", {}).get("child_qualities", [])
    diagnostics["nearest_node_idx"] = nn_idx
    diagnostics["nearest_node_dist"] = nn_dist
    diagnostics["nearest_node_is_simplex_vertex"] = nn_is_vert
    diagnostics["nearest_node_type"] = "simplex_vertex" if nn_is_vert else "external_old_node"
    diagnostics["nearest_node_point"] = tuple(map(float, all_nodes_arr[nn_idx]))

    # --- cluster-to-child assignment summary ---
    child_assignments = _assign_points_to_children(
        centers, best_point, simplex_verts, vert_idx
    )
    diagnostics["cluster_to_child"] = {
        f"cluster_{k}": {
            "center": tuple(map(float, centers[k])),
            "size": int(sizes[k]),
            "child_idx": int(child_assignments[k]),
        }
        for k in range(len(sizes))
    }

    if verbose:
        print(f"[mode6]   WINNER: point={best_pt_tuple}, score={best_score:.6f}")
        print(f"[mode6]   loc_type={loc_type}")
        sep = best_breakdown.get("separation", {}).get("score", "?")
        cen = best_breakdown.get("centrality", {}).get("score", "?")
        qua = best_breakdown.get("quality", {}).get("score", "?")
        col = best_breakdown.get("collision", {}).get("score", "?")
        print(f"[mode6]   scores: sep={sep}, cen={cen}, qual={qua}, coll={col}")
        print(f"[mode6]   nearest node: #{nn_idx} at dist={nn_dist:.6e} "
              f"({'simplex vertex' if nn_is_vert else 'external node'})")
        print(f"[mode6]   cluster→child: {child_assignments.tolist()}")
        # child quality summary
        c_quals = best_breakdown.get("quality", {}).get("child_qualities", [])
        if c_quals:
            print(f"[mode6]   child qualities: {[f'{q:.6e}' for q in c_quals]}")

    return {
        "split_point": best_pt_tuple,
        "loc_type": loc_type,
        "loc_info": loc_info,
        "score": best_score,
        "score_breakdown": best_breakdown,
        "search_mode": search_mode,
        "reason": None,
        "diagnostics": diagnostics,
    }
