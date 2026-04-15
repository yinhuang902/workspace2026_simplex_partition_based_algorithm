# Mode 7 Revised Implementation Plan (v2)

## 1. Desired Two-Stage Mode 7 Behavior

### Overview

Mode 7 selects a split point on a **face** of the LB-best tetrahedron to separate scenario `c_s` clusters.

The revised design is strictly two-stage:

**Stage A — Face Selection:**
Score each of the 4 tetrahedral faces using a combined metric that rewards both (a) cluster proximity to the face and (b) separation of the two dominant clusters when projected onto that face. Pick the highest-scoring face.

**Stage B — Split-Point Selection on the Chosen Face:**
On the chosen face, generate a candidate split point that tries to separate the two dominant clusters. Apply mandatory geometric safety checks including a child-volume prediction. If the candidate fails, try progressively safer fallback points on the same face before moving to the next-ranked face.

### What stays unchanged from v1
- Union-find clustering with `r_cluster = 0.12 * box_diag`
- Cluster pair selection by `w_i * w_j * ||mu_i - mu_j||`
- Fallback to avg_cs when all faces fail
- Termination logic
- All code outside the mode7 block

---

## 2. Revised Face-Selection Strategy (Stage A)

### The problem with proximity-only scoring

A face may be close to both cluster centroids, but the two centroids may project onto nearly the same point on that face. Splitting on such a face would not actually separate the clusters. Therefore face selection must account for both proximity **and** projected separation.

### Two-factor face score

For each face `f`, compute two sub-scores and combine them.

#### Factor 1: Proximity — `prox(f)`

For each cluster `k` with centroid `mu_k` and size `w_k`, compute the unsigned distance `d_k(f)` from `mu_k` to the face plane.

```
prox(f) = sum_k  w_k / (d_k(f) + epsilon)
```

where `epsilon = 1e-8 * box_diag`.

**Meaning:** Higher when large clusters are close to the face. This measures whether the face is geometrically relevant to the cluster structure.

#### Factor 2: Projected separation — `sep(f)`

Project the two dominant cluster centroids `mu1`, `mu2` onto face `f` (using the existing Gram-matrix + barycentric-clamp method). Let `proj1(f)`, `proj2(f)` be the clamped projections.

```
sep(f) = ||proj1(f) - proj2(f)||
```

**Meaning:** Higher when the two dominant clusters, after projection, are well-separated on that face. A face where both clusters project to nearly the same point gets a low separation score, even if both are close to the face.

#### Combined score

```
face_score(f) = prox(f) * sep(f)
```

This is multiplicative: a face must score well on **both** factors to rank high. A face with excellent proximity but near-zero separation gets a near-zero combined score. Similarly, a face with good separation but poor proximity (clusters are far away) also scores low.

#### Face ranking

Rank all 4 faces by `face_score(f)` descending. Process faces in this order during Stage B.

### Computing the face normal (for proximity)

For face with local vertex indices `(a, b, c)`:
- `e1 = verts[b] - verts[a]`
- `e2 = verts[c] - verts[a]`
- `n = e1 × e2` (3D cross product)
- `n_hat = n / ||n||`
- `d_k(f) = |n_hat · (mu_k - verts[a])|`

> [!NOTE]
> **Upgrade path:** The initial implementation uses cluster centroids for both proximity and separation. If this proves unstable in practice (e.g., a single outlier centroid skews the score), the next upgrade is to replace `d_k(f)` with the mean distance of all raw `c_s` points in cluster `k` to the face, rather than just the centroid distance. This upgrade is NOT part of the initial implementation but is the natural next step if centroid-only scoring proves insufficient.

---

## 3. Revised Split-Point Strategy on the Chosen Face (Stage B)

### Candidate generation order

For each face (processed in rank order from Stage A), try candidates in this order:

#### Candidate 1: Cluster-separation midpoint

1. Project `mu1` and `mu2` onto the face (Gram-matrix projection + barycentric clamp)
2. Compute `midpt = 0.5 * (proj1 + proj2)`
3. Apply all mandatory safety checks (§4)
4. If passes: accept

This is the primary candidate. It directly targets cluster separation on the face.

#### Candidate 2: Inward-shifted midpoint

If the raw midpoint fails (e.g., too close to an edge or vertex):

1. Compute face centroid: `fc = (v_a + v_b + v_c) / 3`
2. Shift the midpoint toward the centroid by a blend factor `alpha = 0.3`:
   `shifted = midpt + alpha * (fc - midpt)`
3. Apply all mandatory safety checks
4. If passes: accept

**Rationale:** This preserves partial cluster-separation intent while moving the point away from the boundary. It is a gentler fallback than jumping straight to the centroid.

#### Candidate 3: Face centroid

If both midpoint and shifted midpoint fail:

1. Use `fc = (v_a + v_b + v_c) / 3` directly
2. Apply all mandatory safety checks
3. If passes: accept

The centroid always has barycentric coordinates `(1/3, 1/3, 1/3)` and produces children with equal volumes. It is the safest possible face split point, though it carries no cluster-separation intent.

#### Exhaustion

If all 3 candidates fail for this face, move to the next-ranked face and repeat the same 3-candidate sequence.

If all 4 faces are exhausted, fall back to avg_cs (existing fallback path, unchanged).

---

## 4. Mandatory Geometric Safety Checks

Every candidate point (midpoint, shifted, centroid) must pass **all** of the following checks before acceptance. These are hard requirements, not optional.

### Check 1: Finite coordinates

Reject if any coordinate is NaN or Inf.

### Check 2: Distance to existing nodes

```
min_q ||candidate - q|| >= cand_node_tol    (= 1e-3 * box_diag)
```

Unchanged from v1.

### Check 3: Distance to face vertices

```
min_k ||candidate - face_vertex_k|| >= cand_vertex_tol    (= 5e-3 * box_diag)
```

Unchanged from v1.

### Check 4: Barycentric interior check

Compute face-barycentric coordinates `(lam0, lam1, lam2)` of the candidate on the face triangle, where `lam0 + lam1 + lam2 = 1`.

```
min(lam0, lam1, lam2) >= bary_min
```

> [!IMPORTANT]
> `bary_min` is a **tuning parameter**, not a fixed constant. Initial value: `0.05`. This may be adjusted based on empirical results. The primary hard safeguard is the child-volume check below, not this threshold.

### Check 5: Predicted child-volume check (MANDATORY)

This is the most important new check.

When a point is placed on a face of a tetrahedron, `subdivide_face` creates 3 children. Each child replaces one face vertex with the new point. The volume of each child relative to the parent is determined by the barycentric coordinates of the split point on the face:

```
vol_child_k / vol_parent = lam_k
```

where `lam_k` is the barycentric coordinate corresponding to face vertex `k`.

**Rejection rule:**

```
Reject if  min(lam0, lam1, lam2) * vol_parent < vol_floor
```

where:
- `vol_parent` is the volume of the current tetrahedron (available in `lb_simp_rec["volume"]`)
- `vol_floor` is the minimum acceptable child volume

**How to set `vol_floor`:**

Compute the initial mesh volume (volume of the root tetrahedron at iteration 0). This is available as `vol_parent` of the initial simplex, or can be computed from `box_diag`:

```
vol_floor = 1e-4 * (box_diag)^3 / 6
```

This is approximately `1e-4` times the volume of a regular tetrahedron inscribed in the bounding box. It prevents the algorithm from producing children that are vanishingly small.

**Interaction with bary_min:** The barycentric check (Check 4) provides a *relative* bound (`min_lam >= 0.05` means no child < 5% of parent). The volume check provides an *absolute* bound (no child below `vol_floor` regardless of parent size). Both are needed:
- `bary_min` prevents lopsided splits on large simplices
- `vol_floor` prevents any split on simplices that are already very small

### Check summary table

| Check | Quantity | Threshold | Source |
|-------|----------|-----------|--------|
| Finite | all coords finite | — | basic sanity |
| Node dist | min dist to nodes | `1e-3 * box_diag` | v1, unchanged |
| Vertex dist | min dist to face verts | `5e-3 * box_diag` | v1, unchanged |
| Bary min | min face barycentric | `bary_min` (tunable, init 0.05) | new, soft |
| Child volume | min predicted child vol | `vol_floor` (absolute) | new, **mandatory** |

---

## 5. Minimal-Change Integration Plan

### Files modified

Only `simplex_specialstart.py`, only the mode7 block (lines 4960–5203).

### Code structure

#### [KEEP] Lines 4960–5055
Constants, clustering, pair selection — unchanged.

#### [ADD] Two small local helper functions (defined inside the `if POINT_SELECT_MODE == 7` block)

**Helper 1: `_m7_score_face(face_local_indices, verts, centroids, sizes, mu1, mu2, epsilon)`**

Returns `(face_score, prox, sep, proj1, proj2)` for one face.

- Computes face normal via cross product
- Computes `prox = sum_k w_k / (d_k + eps)`
- Projects `mu1`, `mu2` onto face (Gram-matrix + bary clamp)
- Computes `sep = ||proj1 - proj2||`
- Returns combined `face_score = prox * sep`, plus the projections for reuse in Stage B

**Helper 2: `_m7_validate_face_candidate(candidate, face_verts_coords, verts, nodes, parent_vol, box_diag, bary_min, vol_floor, cand_node_tol, cand_vert_tol)`**

Returns `(ok, rejection_reason, bary_coords)`.

- Checks finite
- Checks node distance
- Checks vertex distance
- Computes face barycentric coordinates
- Checks `min(bary) >= bary_min`
- Checks `min(bary) * parent_vol >= vol_floor`
- Returns `ok=True` or `ok=False` with string reason

Both helpers are small (≤ 30 lines each), defined locally inside the mode7 block, and do not affect any other code.

#### [REPLACE] Lines 5057–5150 (current face-candidate loop)

Replace with:

1. **Face scoring loop:** Call `_m7_score_face` for each of 4 faces. Rank by score. Log scores.

2. **Candidate generation loop:** For each face in rank order:
   - **Try midpoint** of `proj1, proj2` (returned by the scoring helper)
   - If fails: **try shifted midpoint** (blend 0.3 toward face centroid)
   - If fails: **try face centroid**
   - If any candidate passes `_m7_validate_face_candidate`: accept and break
   - If all 3 fail: move to next face

3. **Accept block** (same structure as current): set `new_node`, `chosen_cand`, `_m7_did_cluster_split`

#### [KEEP] Lines 5152–5203
Fallback to avg_cs, termination — unchanged.

### What should NOT be touched
- Clustering logic (union-find, pair selection)
- Fallback chain (avg_cs + snap)
- Termination logic
- All other modes (1, 4, 6)
- Subdivision dispatch (already handles `loc_type="face"`)
- Any code outside mode7 block

---

## 6. Revised Diagnostics / Logging Plan

All output goes to existing `if verbose:` console stream. No new log files.

### After face scoring (Stage A):

```
[mode7]   face scores (prox * sep):
[mode7]     face 0 (v0,v1,v2): prox=12.3 sep=5.67 score=69.7
[mode7]     face 1 (v0,v1,v3): prox=45.1 sep=8.23 score=371.2  <-- best
[mode7]     face 2 (v0,v2,v3): prox= 8.9 sep=2.10 score=18.7
[mode7]     face 3 (v1,v2,v3): prox= 3.2 sep=0.45 score= 1.4
```

### During candidate generation (Stage B):

For each candidate tried:
```
[mode7]   face 1 candidate=midpoint: bary=(0.32,0.41,0.27) min_bary=0.27
          min_child_vol=0.27*1.23e-01=3.32e-02 (floor=2.1e-04) OK
          dist_to_nodes=0.456 dist_to_verts=0.123 -> ACCEPTED
```

Or:
```
[mode7]   face 1 candidate=midpoint: bary=(0.02,0.51,0.47) min_bary=0.02
          -> REJECTED: min_bary < 0.05
[mode7]   face 1 candidate=shifted_midpoint: bary=(0.12,0.45,0.43) min_bary=0.12
          min_child_vol=0.12*1.23e-01=1.48e-02 (floor=2.1e-04) OK
          dist_to_nodes=0.389 dist_to_verts=0.098 -> ACCEPTED
```

### Final summary line:
```
Chosen node (MODE7) (1.23, 4.56, 7.89) face=(v0,v1,v3) candidate_type=shifted_midpoint
  face_rank=1/4, face_score=371.2, dist_to_nodes=0.389
```

Or on fallback:
```
[mode7] All 4 faces exhausted, falling back to avg_cs
```

---

## 7. Differences from Previous Plan (v1)

| Aspect | v1 (Previous) | v2 (This revision) |
|--------|---------------|---------------------|
| **Face scoring** | Proximity only (`sum w_k / d_k`) | Proximity × Separation (`prox * sep`) |
| **Separation term** | Not in main score (backup tiebreaker) | Mandatory factor in combined score |
| **Child-volume check** | Optional recommendation | **Mandatory** acceptance check |
| **Volume floor** | Not defined | `1e-4 * box_diag^3 / 6` (absolute) |
| **Bary_min = 0.05** | Fixed constant | Tuning parameter, initial value 0.05 |
| **Candidate fallback** | Midpoint → centroid (2 steps) | Midpoint → shifted midpoint → centroid (3 steps) |
| **Shifted midpoint** | Not present | Blend factor `alpha=0.3` toward centroid |
| **Helper functions** | None proposed | 2 small local helpers (`_m7_score_face`, `_m7_validate_face_candidate`) |
| **Centroid-only upgrade note** | Not mentioned | Explicitly noted as next upgrade path |
| **Logging** | Minimal | Per-face scores, per-candidate check details, candidate type in final log |
