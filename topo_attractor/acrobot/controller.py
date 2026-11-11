"""Feedback policy and the shared energy-regulation warm start."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .plant import Acrobot


class Controller(nn.Module):
    def __init__(self, hidden: int = 64, u_max: float = 8.0):
        super().__init__()
        self.u_max = float(u_max)

        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Start with modest output magnitude.
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(x: torch.Tensor) -> torch.Tensor:
        q1, q2, dq1, dq2 = x.unbind(dim=-1)
        return torch.stack(
            [
                torch.cos(q1),
                torch.sin(q1),
                torch.cos(q2),
                torch.sin(q2),
                dq1,
                dq2,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.features(x))
        return self.u_max * torch.tanh(raw)


def warm_start_energy_controller(
    controller: Controller,
    plant: Acrobot,
    steps: int,
    energy_target: float,
    k_energy: float,
    batch_size: int = 1024,
    lr: float = 2e-3,
) -> None:
    """
    Train the MLP to imitate:
        u_seed = sat(k (E_target - E) dq2)

    This seeds sustained motion but does NOT specify a particular orbit.
    """
    if steps <= 0:
        return

    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    for _ in range(steps):
        q = (
            2.0
            * torch.rand(
                batch_size, 2,
                dtype=torch.get_default_dtype(),
            )
            - 1.0
        ) * 1.5

        dq = (
            2.0
            * torch.rand(
                batch_size, 2,
                dtype=torch.get_default_dtype(),
            )
            - 1.0
        ) * 3.0

        x = torch.cat([q, dq], dim=-1)
        E = plant.energy(x)

        target = (
            k_energy
            * (energy_target - E)
            * dq[:, 1]
        ).clamp(
            -plant.u_max,
            plant.u_max,
        ).unsqueeze(-1)

        pred = controller(x)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
