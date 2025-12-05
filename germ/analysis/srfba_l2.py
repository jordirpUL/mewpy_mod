from typing import Dict, Any, Union

from mewpy.germ.analysis import SRFBA  # or from .srfba import SRFBA, depending on your layout
from mewpy.solvers.solution import Status


class SRFBA_L2(SRFBA):
    """
    SRFBA with an additional L2-minimization step.

    Two-stage solve:
      1) Standard SRFBA to get optimal objective Z* under given regulatory state.
      2) Quadratic re-optimization: minimize sum(v_i^2) subject to
         - same mass balance and regulatory constraints
         - objective reaction fixed to Z*.
    """

    def _optimize(
        self,
        to_solver: bool = False,
        solver_kwargs: Dict[str, Any] | None = None,
        initial_state: Dict[str, float] | None = None,
        **kwargs
    ):
        if solver_kwargs is None:
            solver_kwargs = {}
        if initial_state is None:
            initial_state = {}

        # -------- Stage 1: regular SRFBA (maximize objective) --------
        # Force getting values so we can extract fluxes & objective
        self.build_solver()

        kw1 = dict(solver_kwargs)
        kw1.setdefault("get_values", True)

        sol1 = super()._optimize(
            to_solver=True,
            solver_kwargs=kw1,
            initial_state=initial_state,
            **kwargs,
        )

        if sol1.status != Status.OPTIMAL:
            # If SRFBA itself cannot find an optimal solution, just return it.
            return sol1

        Z = float(sol1.fobj)

        # For simplicity, handle only the usual case of a single objective reaction.
        obj_items = list(self.model.objective.items())
        if len(obj_items) != 1:
            # If there are multiple objective reactions, fall back to pure SRFBA solution.
            return sol1

        obj_rxn, coef = obj_items[0]
        coef = float(coef)
        if coef == 0.0:
            # Degenerate objective; nothing to fix.
            return sol1

        # Flux value of the objective reaction at optimum
        obj_flux = Z / coef

        # -------- Stage 2: QP – minimize sum(v_i^2) with objective fixed --------

        # Define flux variables only
        flux_vars = [rxn.id for rxn in self.model.yield_reactions()]

        # L2 objective: only on real fluxes
        quadratic = {(var, var): 1.0 for var in flux_vars}

        # Zero linear objective on only the flux variables
        linear = {var: 0.0 for var in flux_vars}

        # Constraints override (objective fixed + initial_state)
        constraints = {}
        constraints.update(initial_state)
        constraints[obj_rxn.id] = (obj_flux, obj_flux)

        sol2 = self.solver.solve(
            linear=linear,
            quadratic=quadratic,
            minimize=True,
            constraints=constraints,
            get_values=True,
        )


        # If QP fails for some reason, fall back to SRFBA solution.
        if sol2.status != Status.OPTIMAL:
            return sol1

        # Optionally, you can return a ModelSolution instead,
        # but this mirrors how SRFBA currently returns a Solver Solution.
        return sol2
