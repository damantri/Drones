from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.
    """
    csp.assignment_attempts = 0
    csp.backtracks = 0

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:
            csp.assignment_attempts += 1

            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)

                result = backtrack(assignment)
                if result is not None:
                    return result

                csp.unassign(var, assignment)

        csp.backtracks += 1
        return None

    return backtrack({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.
    """
    csp.assignment_attempts = 0
    csp.backtracks = 0

    def forward_check(var: str, assignment: dict[str, str]) -> bool:
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            new_domain: list[str] = []
            for val in csp.domains[neighbor]:
                if csp.is_consistent(neighbor, val, assignment):
                    new_domain.append(val)

            csp.domains[neighbor] = new_domain

            if not new_domain:
                return False

        return True

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(csp.domains[var]):
            csp.assignment_attempts += 1

            if not csp.is_consistent(var, value, assignment):
                continue

            saved_domains = {v: list(d) for v, d in csp.domains.items()}
            csp.assign(var, value, assignment)

            if forward_check(var, assignment):
                result = backtrack(assignment)
                if result is not None:
                    return result

            csp.unassign(var, assignment)
            csp.domains = saved_domains

        csp.backtracks += 1
        return None

    return backtrack({})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.
    """
    csp.assignment_attempts = 0
    csp.backtracks = 0

    def values_compatible(xi: str, vi: str, xj: str, vj: str) -> bool:
        temp = {xj: vj}
        if not csp.is_consistent(xi, vi, temp):
            return False

        temp = {xi: vi}
        if not csp.is_consistent(xj, vj, temp):
            return False

        return True

    def revise(xi: str, xj: str) -> bool:
        revised = False
        new_domain: list[str] = []

        for vi in csp.domains[xi]:
            supported = any(
                values_compatible(xi, vi, xj, vj) for vj in csp.domains[xj]
            )

            if supported:
                new_domain.append(vi)
            else:
                revised = True

        if revised:
            csp.domains[xi] = new_domain

        return revised

    def ac3(initial_arcs: list[tuple[str, str]] | None = None) -> bool:
        if initial_arcs is None:
            queue = deque(
                (xi, xj)
                for xi in csp.variables
                for xj in csp.get_neighbors(xi)
            )
        else:
            queue = deque(initial_arcs)

        while queue:
            xi, xj = queue.popleft()

            if revise(xi, xj):
                if not csp.domains[xi]:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        queue.append((xk, xi))

        return True

    if not ac3():
        return None

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(csp.domains[var]):
            csp.assignment_attempts += 1

            if not csp.is_consistent(var, value, assignment):
                continue

            saved_domains = {v: list(d) for v, d in csp.domains.items()}
            csp.assign(var, value, assignment)
            csp.domains[var] = [value]

            arcs = [
                (neighbor, var)
                for neighbor in csp.get_neighbors(var)
                if neighbor not in assignment
            ]

            if ac3(arcs):
                result = backtrack(assignment)
                if result is not None:
                    return result

            csp.unassign(var, assignment)
            csp.domains = saved_domains

        csp.backtracks += 1
        return None

    return backtrack({})


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.
    """
    csp.assignment_attempts = 0
    csp.backtracks = 0

    def select_unassigned_variable(assignment: dict[str, str]) -> str:
        unassigned = csp.get_unassigned_variables(assignment)
        return min(
            unassigned,
            key=lambda var: (
                len(csp.domains[var]),
                -sum(1 for n in csp.get_neighbors(var) if n not in assignment),
            ),
        )

    def order_domain_values(var: str, assignment: dict[str, str]) -> list[str]:
        legal_values = [
            v for v in csp.domains[var]
            if csp.is_consistent(var, v, assignment)
        ]
        return sorted(
            legal_values,
            key=lambda value: csp.get_num_conflicts(var, value, assignment),
        )

    def forward_check(var: str, assignment: dict[str, str]) -> bool:
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            new_domain = [
                val for val in csp.domains[neighbor]
                if csp.is_consistent(neighbor, val, assignment)
            ]
            csp.domains[neighbor] = new_domain

            if not new_domain:
                return False

        return True

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        var = select_unassigned_variable(assignment)

        for value in order_domain_values(var, assignment):
            csp.assignment_attempts += 1

            saved_domains = {v: list(d) for v, d in csp.domains.items()}
            csp.assign(var, value, assignment)

            if forward_check(var, assignment):
                result = backtrack(assignment)
                if result is not None:
                    return result

            csp.unassign(var, assignment)
            csp.domains = saved_domains

        csp.backtracks += 1
        return None

    return backtrack({})