"""Two-link underactuated pendulum (Acrobot): dynamics, energy, initial conditions, RK4 closed-loop rollout."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
import torch

if TYPE_CHECKING:  # type hints only
    from .controller import Controller

DEVICE = torch.device("cpu")


class Acrobot:
    """
    Two-link underactuated arm from the project one-pager.

    Angles:
        q1 = first link angle from downward vertical
        q2 = second link angle relative to first link

    State:
        x = [q1, q2, dq1, dq2]

    Input:
        elbow torque u
    """

    def __init__(
        self,
        g: float = 9.81,
        damping: float = 0.30,
        u_max: float = 8.0,
    ):
        self.g = float(g)
        self.damping = float(damping)
        self.u_max = float(u_max)

    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        q2 = q[..., 1]
        c2 = torch.cos(q2)

        M11 = 3.0 + 2.0 * c2
        M12 = 1.0 + c2
        M22 = torch.ones_like(q2)

        row1 = torch.stack([M11, M12], dim=-1)
        row2 = torch.stack([M12, M22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def coriolis(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        q2 = q[..., 1]
        dq1 = dq[..., 0]
        dq2 = dq[..., 1]
        s2 = torch.sin(q2)

        h1 = -s2 * (2.0 * dq1 * dq2 + dq2.square())
        h2 = s2 * dq1.square()
        return torch.stack([h1, h2], dim=-1)

    def gravity(self, q: torch.Tensor) -> torch.Tensor:
        q1 = q[..., 0]
        q2 = q[..., 1]

        g1 = self.g * (
            2.0 * torch.sin(q1)
            + torch.sin(q1 + q2)
        )
        g2 = self.g * torch.sin(q1 + q2)
        return torch.stack([g1, g2], dim=-1)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q = x[..., :2]
        dq = x[..., 2:]

        u = torch.clamp(u.squeeze(-1), -self.u_max, self.u_max)

        M = self.mass_matrix(q)
        h = self.coriolis(q, dq)
        G = self.gravity(q)

        tau = torch.stack(
            [torch.zeros_like(u), u],
            dim=-1,
        )

        rhs = tau - h - G - self.damping * dq
        ddq = torch.linalg.solve(
            M,
            rhs.unsqueeze(-1),
        ).squeeze(-1)

        return torch.cat([dq, ddq], dim=-1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mechanical energy consistent with the supplied M(q) and G(q).

        Potential is chosen as
            V = -g [2 cos q1 + cos(q1+q2)]
        so the hanging state q1=q2=0 has energy -3g.
        """
        q = x[..., :2]
        dq = x[..., 2:]

        M = self.mass_matrix(q)
        kinetic = 0.5 * torch.einsum(
            "...i,...ij,...j->...",
            dq,
            M,
            dq,
        )

        q1 = q[..., 0]
        q2 = q[..., 1]
        potential = -self.g * (
            2.0 * torch.cos(q1)
            + torch.cos(q1 + q2)
        )

        return kinetic + potential

    def sample_initial(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
        q_range: float = 0.60,
        dq_range: float = 0.40,
    ) -> torch.Tensor:
        q = (
            2.0
            * torch.rand(
                n, 2,
                generator=generator,
                dtype=torch.get_default_dtype(),
            )
            - 1.0
        ) * q_range

        dq = (
            2.0
            * torch.rand(
                n, 2,
                generator=generator,
                dtype=torch.get_default_dtype(),
            )
            - 1.0
        ) * dq_range

        return torch.cat([q, dq], dim=-1).to(DEVICE)


def rk4_step(
    plant: Acrobot,
    controller: Controller,
    x: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def f(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = controller(z)
        return plant.dynamics(z, u), u

    k1, u1 = f(x)
    k2, _ = f(x + 0.5 * dt * k1)
    k3, _ = f(x + 0.5 * dt * k2)
    k4, _ = f(x + dt * k3)

    x_next = x + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )
    return x_next, u1


def rollout(
    plant: Acrobot,
    controller: Controller,
    x0: torch.Tensor,
    dt: float,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
        traj:     [steps+1, N, 4]
        controls: [steps,   N, 1]
    """
    xs = [x0]
    us = []
    x = x0

    for _ in range(steps):
        x, u = rk4_step(
            plant,
            controller,
            x,
            dt,
        )
        xs.append(x)
        us.append(u)

    return torch.stack(xs, dim=0), torch.stack(us, dim=0)
