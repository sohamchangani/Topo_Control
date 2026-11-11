"""Hard Poincare validator (no persistent homology decides success)."""
from __future__ import annotations
import math
from typing import Dict, List
import numpy as np
import torch


def hard_upward_crossings(
    traj: torch.Tensor,
    burn_in: int,
) -> List[np.ndarray]:
    """
    Return hard q1=0, dq1>0 Poincare crossings for each trajectory.

    Crossing state is represented as:
        [q2_wrapped, dq1, dq2]

    Linear interpolation is used between adjacent time steps.  Crossings are
    detected at q1 = 2*pi*k, so the diagnostic works for both librations and
    repeated rotations.
    """
    x = traj[burn_in:].detach().cpu().numpy()
    out: List[np.ndarray] = []

    for j in range(x.shape[1]):
        rows = []

        for k in range(x.shape[0] - 1):
            q1a = x[k, j, 0]
            q1b = x[k + 1, j, 0]

            # Upward crossings of q1 = 0 mod 2pi.  This is consistent with
            # the soft training section based on 1-cos(q1), and works for
            # both librations and rotations.
            if q1b <= q1a:
                continue

            two_pi = 2.0 * math.pi
            first_n = math.floor(q1a / two_pi) + 1
            last_n = math.floor(q1b / two_pi)

            # Include an exact boundary at q1a only if numerical noise places
            # it infinitesimally below the section; ordinary segments need no
            # special case.
            for n in range(first_n, last_n + 1):
                section_value = n * two_pi
                denom = q1b - q1a
                if abs(denom) <= 1e-12:
                    continue
                frac = (section_value - q1a) / denom
                if not (0.0 <= frac <= 1.0):
                    continue

                state = (
                    x[k, j]
                    + frac * (x[k + 1, j] - x[k, j])
                )

                if state[2] > 0.0:
                    q2_wrapped = math.atan2(
                        math.sin(state[1]),
                        math.cos(state[1]),
                    )
                    rows.append(
                        [q2_wrapped, state[2], state[3]]
                    )

        if rows:
            out.append(np.asarray(rows, dtype=float))
        else:
            out.append(np.empty((0, 3), dtype=float))

    return out


def poincare_evaluation(
    traj: torch.Tensor,
    burn_in: int,
    last_k: int,
    within_tol: float,
    between_tol: float,
    velocity_scale: float,
) -> Dict[str, float]:
    """
    Test whether trajectories converge to one common Poincare point.

    For each trajectory with enough crossings:
      - average the last K crossings in
            (cos q2, sin q2, dq1/v_scale, dq2/v_scale)
      - compute within-trajectory spread.

    Then compare each trajectory's limiting point with the common median
    limiting point across initial conditions.
    """
    crossings = hard_upward_crossings(
        traj,
        burn_in,
    )

    centers = []
    within_errors = []

    for j, arr in enumerate(crossings):
        if arr.shape[0] < last_k:
            continue

        tail = arr[-last_k:]
        q2 = tail[:, 0]
        dq1 = tail[:, 1]
        dq2 = tail[:, 2]

        z = np.stack(
            [
                np.cos(q2),
                np.sin(q2),
                dq1 / velocity_scale,
                dq2 / velocity_scale,
            ],
            axis=1,
        )

        center = z.mean(axis=0)
        within = np.sqrt(
            np.mean(
                np.sum(
                    (z - center[None, :]) ** 2,
                    axis=1,
                )
            )
        )

        centers.append(center)
        within_errors.append(float(within))

    if not centers:
        return {
            "poincare_success_rate": 0.0,
            "poincare_error": float("nan"),
            "fraction_with_enough_crossings": 0.0,
            "within_error_mean": float("nan"),
            "between_error_mean": float("nan"),
        }

    centers_arr = np.asarray(centers)
    common = np.median(centers_arr, axis=0)

    between_errors = np.linalg.norm(
        centers_arr - common[None, :],
        axis=1,
    )

    success = [
        (
            within_errors[i] <= within_tol
            and between_errors[i] <= between_tol
        )
        for i in range(len(centers))
    ]

    # Missing-crossing trajectories count as failures.
    success_count = int(np.sum(success))
    success_rate = success_count / len(crossings)

    combined = np.sqrt(
        np.asarray(within_errors) ** 2
        + between_errors ** 2
    )

    return {
        "poincare_success_rate": float(success_rate),
        "poincare_error": float(np.mean(combined)),
        "fraction_with_enough_crossings": (
            len(centers) / len(crossings)
        ),
        "within_error_mean": float(
            np.mean(within_errors)
        ),
        "between_error_mean": float(
            np.mean(between_errors)
        ),
    }
