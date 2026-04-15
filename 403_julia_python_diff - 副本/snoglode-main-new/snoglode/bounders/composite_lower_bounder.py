"""
CompositeLowerBounder  —  orchestrator for two lower-bounding pathways:
  (1) Existing per-scenario LB (e.g., DropNonants)  → existing_lb
  (2) LP relaxation of the full EF                  → lp_relax_lb

Final node LB = max(existing_lb, lp_relax_lb).

MPI CONTRACT
------------
This class has TWO phases that must be called in order:

  Phase A  —  solve()                     (called BEFORE MPI aggregation)
               Runs only the existing per-scenario lower bounder.
               Sets lb_problem.objective  =  this rank's partial sum.
               Sets lb_problem.subproblem_solutions = per-scenario payload.

  [MPI allreduce on objective/feasible happens in solver.dispatch_lb_solve]

  Phase B  —  apply_lp_relaxation()       (called AFTER  MPI aggregation)
               Runs the LP relaxation on the FULLY-AGGREGATED existing LB.
               Updates lb_problem.objective = max(aggregated_existing, lp_relax).
               Stores diagnostics in lb_problem.lp_relax_diagnostics.

This two-phase split ensures that the MPI SUM on per-rank partial objectives
is not corrupted by a pre-aggregation max with the full-EF LP relaxation LB.

CRITICAL INVARIANT
------------------
Neither phase EVER writes to  lb_problem.subproblem_solutions.
The existing per-scenario solutions (set by DropNonants in Phase A) are
left FULLY INTACT for use by:
    - branching logic  (tree.py)
    - candidate generation  (compute.py)
    - solution inheritance  (lower_bounders.py)
"""
import logging
import pyomo.environ as pyo

from snoglode.bounders.lower_bounders import AbstractLowerBounder
from snoglode.relaxation.ef_relaxation_context import EFRelaxationContext

logger = logging.getLogger(__name__)


class CompositeLowerBounder:
    """
    Wraps an existing AbstractLowerBounder and supplements it with
    a Julia-style LP relaxation lower bound.
    """

    def __init__(self,
                 existing_bounder: AbstractLowerBounder,
                 subproblems,
                 max_oa_iters: int = 20):
        """
        Parameters
        ----------
        existing_bounder : AbstractLowerBounder
            The primary lower bounder (e.g. DropNonants).
        subproblems : Subproblems
            Used to build a *private* EF clone for LP relaxation.
        max_oa_iters : int
            Maximum OA/αBB strengthening iterations per node.
        """
        self.existing_bounder = existing_bounder
        self._lp_context = EFRelaxationContext(subproblems)
        self._max_oa_iters = max_oa_iters
        # Store subproblems reference for P0 (bounds extraction)
        self._subproblems = subproblems

    # ── Expose existing_bounder attributes for compatibility ──

    @property
    def perform_fbbt(self):
        return self.existing_bounder.perform_fbbt

    @perform_fbbt.setter
    def perform_fbbt(self, value):
        self.existing_bounder.perform_fbbt = value

    @property
    def inhert_solutions(self):
        return self.existing_bounder.inhert_solutions

    @inhert_solutions.setter
    def inhert_solutions(self, value):
        self.existing_bounder.inhert_solutions = value

    # ── Phase A: existing per-scenario LB (BEFORE MPI aggregation) ──

    def solve(self, subproblems, node, **kwargs) -> None:
        """
        Run ONLY the existing lower bounder (per-rank partial objectives).

        After this call:
          - lb_problem.objective  = this rank's partial sum of scenario objectives.
          - lb_problem.feasible   = this rank's feasibility (AND-reduced later).
          - lb_problem.subproblem_solutions = per-scenario payload (PRESERVED).

        LP relaxation is NOT run here.  It must be deferred to
        apply_lp_relaxation(), which runs AFTER MPI aggregation.
        """
        self.existing_bounder.solve(subproblems=subproblems, node=node)

    # ── P0: Extract tightened bounds from live subproblems ──

    def _extract_tightened_bounds(self):
        """
        Read current variable bounds from the live (FBBT/OBBT-tightened)
        subproblem models and map them to LP relaxation var indices.

        Variable name mapping
        ---------------------
        The private EF adds each scenario model as a Pyomo Block, so EF
        variable names are *qualified*:  ``"scen_0.x1"``, ``"scen_0.x2"``, etc.
        Live subproblem models are standalone ConcreteModels, so their
        variable names are *local*:  ``"x1"``, ``"x2"``, etc.

        We reconstruct the qualified name as  ``f"{scen_name}.{local_name}"``.

        Returns
        -------
        tightened : dict
            Mapping  EF var index → (lb, ub).
        n_matched : int
            Number of live variables successfully matched to an EF index.
        n_missed : int
            Number of live variables with no matching EF index.
        """
        pp = self._lp_context._pp
        var_name_to_idx = pp.var_name_to_idx
        tightened = {}
        n_matched = 0
        n_missed = 0

        for scen_name in self._subproblems.names:
            model = self._subproblems.model[scen_name]
            for var in model.component_data_objects(pyo.Var, active=True):
                # Construct the qualified name as it appears in the EF
                local_name = var.name
                qualified_name = f"{scen_name}.{local_name}"

                if qualified_name in var_name_to_idx:
                    idx = var_name_to_idx[qualified_name]
                elif local_name in var_name_to_idx:
                    # Fallback: try local name directly
                    idx = var_name_to_idx[local_name]
                else:
                    n_missed += 1
                    continue

                n_matched += 1
                lb = var.lb if var.has_lb() else None
                ub = var.ub if var.has_ub() else None

                # Take the tightest bounds across scenarios for shared vars
                if idx in tightened:
                    old_lb, old_ub = tightened[idx]
                    new_lb = max(lb, old_lb) if (lb is not None and old_lb is not None) else (lb or old_lb)
                    new_ub = min(ub, old_ub) if (ub is not None and old_ub is not None) else (ub or old_ub)
                    tightened[idx] = (new_lb, new_ub)
                else:
                    tightened[idx] = (lb, ub)

        return tightened, n_matched, n_missed

    # ── Phase B: LP relaxation + combine  (AFTER MPI aggregation) ──
    #    P0 only:  single-pass LP solve with FBBT-tightened bounds.
    #    P1 (iterative LP↔FBBT coupling) is deferred until P2
    #    (reduced-cost bound tightening) provides a real feedback channel
    #    from the LP solution back to the bound tightener.

    def apply_lp_relaxation(self, node, current_UB=float("inf")) -> None:
        """
        Run LP relaxation using tightened bounds from live subproblems and
        combine with the existing (fully-aggregated) LB.

        Flow (P0):
          1. Extract tightened bounds from live subproblems
          2. Solve LP relaxation using tightened bounds (single pass)
          3. Combine: final_lb = max(existing_lb, lp_relax_lb)

        PRECONDITION: must be called AFTER dispatch_lb_solve() has run
        MPI.allreduce on lb_problem.objective (SUM) and
        lb_problem.feasible (PROD).

        NEVER modifies lb_problem.subproblem_solutions.
        """
        existing_lb = node.lb_problem.objective
        existing_feasible = node.lb_problem.feasible

        # ── Guard 1: existing LB already infeasible → skip ──
        if not existing_feasible:
            node.lb_problem.lp_relax_diagnostics = {
                "existing_lb": existing_lb,
                "lp_relax_lb": float("-inf"),
                "final_lb": existing_lb,
                "lb_source": "existing",
                "lp_relax_feasible": False,
                "n_oa_rounds": 0,
                "n_oa_cuts": 0,
                "n_abb_cuts": 0,
                "lp_status": "skipped (existing infeasible)",
            }
            return

        # ── Guard 2: objective has unsupported NL terms → LP LB is invalid ──
        if self._lp_context.obj_unsupported:
            node.lb_problem.lp_relax_diagnostics = {
                "existing_lb": existing_lb,
                "lp_relax_lb": float("-inf"),
                "final_lb": existing_lb,
                "lb_source": "existing",
                "lp_relax_feasible": False,
                "n_oa_rounds": 0,
                "n_oa_cuts": 0,
                "n_abb_cuts": 0,
                "lp_status": "skipped (objective has unsupported NL terms)",
            }
            logger.debug(
                f"CompositeLB node {node.id}: LP relaxation SKIPPED "
                f"(objective has unsupported nonlinear terms)"
            )
            return

        # ── P0: extract tightened bounds and solve LP ──
        tightened_bounds, n_matched, n_missed = self._extract_tightened_bounds()
        logger.debug(
            f"CompositeLB node {node.id}: tightened bounds extracted — "
            f"{n_matched} vars matched, {n_missed} missed, "
            f"{len(tightened_bounds)} unique EF indices covered "
            f"(of {self._lp_context._pp.n_ef_vars} total)"
        )

        try:
            lp_result = self._lp_context.solve_node(
                node_state=node.state,
                current_UB=current_UB,
                mingap_abs=1e-3,
                mingap_rel=1e-2,
                max_oa_iters=self._max_oa_iters,
                tightened_var_bounds=tightened_bounds,
            )
        except Exception as e:
            logger.warning(f"LP relaxation solve_node failed: {e}")
            node.lb_problem.lp_relax_diagnostics = {
                "existing_lb": existing_lb,
                "lp_relax_lb": float("-inf"),
                "final_lb": existing_lb,
                "lb_source": "existing",
                "lp_relax_feasible": False,
                "n_oa_rounds": 0,
                "n_oa_cuts": 0,
                "n_abb_cuts": 0,
                "lp_status": f"error: {e}",
            }
            return

        lp_status = lp_result.get("lp_status", "N/A")

        # ── Case A: LP relaxation proved infeasibility ──
        if lp_status == "infeasible":
            node.lb_problem.feasible = False
            node.lb_problem.objective = float("inf")
            node.lb_problem.lp_relax_diagnostics = {
                "existing_lb": existing_lb,
                "lp_relax_lb": float("inf"),
                "final_lb": float("inf"),
                "lb_source": "lp_relax_infeasible",
                "lp_relax_feasible": False,
                "n_oa_rounds": lp_result.get("n_oa_rounds", 0),
                "n_oa_cuts": lp_result.get("n_oa_cuts", 0),
                "n_abb_cuts": lp_result.get("n_abb_cuts", 0),
                "lp_status": "infeasible",
            }
            logger.debug(
                f"CompositeLB node {node.id}: LP relaxation INFEASIBLE "
                f"→ node marked infeasible"
            )
            return

        # ── Case B: LP relaxation solved successfully ──
        if lp_result.get("feasible", False):
            lp_lb = lp_result.get("lp_relax_lb", float("-inf"))
            final_lb = max(existing_lb, lp_lb)
            node.lb_problem.objective = final_lb

            lb_source = "lp_relax" if lp_lb > existing_lb else "existing"
            node.lb_problem.lp_relax_diagnostics = {
                "existing_lb": existing_lb,
                "lp_relax_lb": lp_lb,
                "final_lb": final_lb,
                "lb_source": lb_source,
                "lp_relax_feasible": True,
                "n_oa_rounds": lp_result.get("n_oa_rounds", 0),
                "n_oa_cuts": lp_result.get("n_oa_cuts", 0),
                "n_abb_cuts": lp_result.get("n_abb_cuts", 0),
                "lp_status": lp_status,
            }
            logger.debug(
                f"CompositeLB node {node.id}: existing={existing_lb:.6g}, "
                f"lp_relax={lp_lb:.6g}, final={final_lb:.6g} "
                f"[source={lb_source}]"
            )
            return

        # ── Case C: LP solver returned non-optimal, non-infeasible ──
        node.lb_problem.lp_relax_diagnostics = {
            "existing_lb": existing_lb,
            "lp_relax_lb": float("-inf"),
            "final_lb": existing_lb,
            "lb_source": "existing",
            "lp_relax_feasible": False,
            "n_oa_rounds": lp_result.get("n_oa_rounds", 0),
            "n_oa_cuts": lp_result.get("n_oa_cuts", 0),
            "n_abb_cuts": lp_result.get("n_abb_cuts", 0),
            "lp_status": lp_status,
        }
        logger.debug(
            f"CompositeLB node {node.id}: LP relaxation non-conclusive "
            f"(status={lp_status}), falling back to existing_lb={existing_lb:.6g}"
        )

    # ── Write LP relaxation results to the subproblem log ──

    def _write_lp_relax_to_subproblem_log(self, node) -> None:
        """
        Append LP relaxation results to the per-subproblem log file.

        Called from solver.dispatch_lb_solve() AFTER apply_lp_relaxation().
        Uses the same log file as existing_bounder._subproblem_log.
        """
        log_path = getattr(self.existing_bounder, '_subproblem_log', None)
        if not log_path:
            return
        diag = getattr(node.lb_problem, 'lp_relax_diagnostics', None)
        if diag is None:
            return
        try:
            with open(log_path, "a") as f:
                f.write(f"  --- LP Relaxation (Node {node.id}) ---\n")
                f.write(f"  Status:          {diag.get('lp_status', 'N/A')}\n")
                f.write(f"  Existing LB:     {diag.get('existing_lb', float('nan')):.8g}\n")
                f.write(f"  LP Relax LB:     {diag.get('lp_relax_lb', float('nan')):.8g}\n")
                f.write(f"  Final LB:        {diag.get('final_lb', float('nan')):.8g}\n")
                f.write(f"  LB Source:       {diag.get('lb_source', '?')}\n")
                f.write(f"  LP Feasible:     {diag.get('lp_relax_feasible', '?')}\n")
                f.write(f"  OA Rounds:       {diag.get('n_oa_rounds', 0)}\n")
                f.write(f"  OA Cuts Added:   {diag.get('n_oa_cuts', 0)}\n")
                f.write(f"  aBB Cuts Added:  {diag.get('n_abb_cuts', 0)}\n")
                f.write(f"  -----------------------------------\n")
        except Exception as ex:
            logger.warning(f"Failed to write LP relax to subproblem log: {ex}")
