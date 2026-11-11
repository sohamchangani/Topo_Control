"""Objectives and training loop."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .plant import Acrobot, DEVICE, rollout
from .controller import Controller, warm_start_energy_controller
from .losses import (OINEUS_AVAILABLE, energy_loss, range_loss, soft_poincare_loss, make_topology_cloud,
                     differentiable_h1_diagram, one_loop_plateau_loss)


MODES = (
    "energy_only",
    "range_only",
    "poincare_only",
    "topology_only",
    "topology_energy",
    "topology_range",
    "topology_poincare",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def uses_topology(mode: str) -> bool:
    return mode in {
        "topology_only",
        "topology_energy",
        "topology_range",
        "topology_poincare",
    }


def uses_energy(mode: str) -> bool:
    return mode in {"energy_only", "topology_energy"}


def uses_range(mode: str) -> bool:
    return mode in {"range_only", "topology_range"}


def uses_poincare(mode: str) -> bool:
    return mode in {"poincare_only", "topology_poincare"}


@dataclass
class TrainRecord:
    epoch: int
    total: float
    energy: float
    range_loss: float
    poincare: float
    topology: float
    control: float
    plateau_start: float
    plateau_end: float
    plateau_width: float
    l1: float
    l2: float


def train(args):
    if uses_topology(args.mode) and not OINEUS_AVAILABLE:
        raise ImportError(
            "This mode requires Oineus: pip install oineus"
        )

    set_seed(args.seed)

    plant = Acrobot(
        g=args.g,
        damping=args.damping,
        u_max=args.u_max,
    )

    controller = Controller(
        hidden=args.hidden,
        u_max=args.u_max,
    ).to(DEVICE)

    # Common initialization for fair model comparisons.
    warm_start_energy_controller(
        controller,
        plant,
        steps=args.warm_start_steps,
        energy_target=args.energy_target,
        k_energy=args.warm_start_k,
        batch_size=args.warm_start_batch,
        lr=args.warm_start_lr,
    )

    train_gen = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed + 1001)

    fixed_x0 = plant.sample_initial(
        args.n_initial,
        generator=train_gen,
        q_range=args.initial_q_range,
        dq_range=args.initial_dq_range,
    )

    optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=args.lr,
    )

    history: List[TrainRecord] = []

    for epoch in range(1, args.epochs + 1):
        if args.resample_initials:
            gen = torch.Generator(
                device="cpu"
            ).manual_seed(
                args.seed + 2000 + epoch
            )
            x0 = plant.sample_initial(
                args.n_initial,
                generator=gen,
                q_range=args.initial_q_range,
                dq_range=args.initial_dq_range,
            )
        else:
            x0 = fixed_x0

        traj, controls = rollout(
            plant,
            controller,
            x0,
            args.dt,
            args.steps,
        )

        # Compute all diagnostics, but only activate requested losses.
        e_loss, _ = energy_loss(
            plant,
            traj,
            args.burn_in,
            args.energy_target,
        )

        r_loss, _, _, _, _ = range_loss(
            traj,
            args.burn_in,
            args.q1_amp_target,
            args.q2_amp_target,
        )

        p_loss, p_spread, _, _ = soft_poincare_loss(
            traj,
            args.burn_in,
            velocity_scale=args.velocity_scale,
            section_sigma=args.section_sigma,
            direction_sigma=args.direction_sigma,
            min_section_mass=args.min_section_mass,
            lambda_mass=args.lambda_section_mass,
            min_crossing_speed=args.min_crossing_speed,
            lambda_speed=args.lambda_crossing_speed,
        )

        if uses_topology(args.mode):
            cloud = make_topology_cloud(
                traj,
                args.burn_in,
                args.ph_points,
                args.velocity_scale,
            )

            dgm1 = differentiable_h1_diagram(
                cloud,
                max_diameter=args.max_diameter,
                n_threads=args.ph_threads,
            )

            (
                t_loss,
                plateau_start,
                plateau_end,
                plateau_width,
                l1,
                l2,
                lifetimes,
            ) = one_loop_plateau_loss(
                dgm1,
                cloud,
            )

            plateau_start_val = float(plateau_start.detach().item())
            plateau_end_val = float(plateau_end.detach().item())
            plateau_width_val = float(plateau_width.detach().item())
            l1_val = float(l1.detach().item())
            l2_val = float(l2.detach().item())
        else:
            t_loss = traj.sum() * 0.0
            plateau_start_val = float("nan")
            plateau_end_val = float("nan")
            plateau_width_val = float("nan")
            l1_val = float("nan")
            l2_val = float("nan")

        c_loss = controls.square().mean()

        total = (
            (args.lambda_energy * e_loss if uses_energy(args.mode) else 0.0)
            + (args.lambda_range * r_loss if uses_range(args.mode) else 0.0)
            + (args.lambda_poincare * p_loss if uses_poincare(args.mode) else 0.0)
            + (args.lambda_topology * t_loss if uses_topology(args.mode) else 0.0)
            + args.lambda_control * c_loss
        )

        optimizer.zero_grad()
        total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(),
            args.grad_clip,
        )

        if not torch.isfinite(
            torch.as_tensor(grad_norm)
        ):
            raise RuntimeError(
                f"Non-finite gradient at epoch {epoch}"
            )

        optimizer.step()

        history.append(
            TrainRecord(
                epoch=epoch,
                total=float(total.detach().item()),
                energy=float(e_loss.detach().item()),
                range_loss=float(r_loss.detach().item()),
                poincare=float(p_loss.detach().item()),
                topology=float(t_loss.detach().item()),
                control=float(c_loss.detach().item()),
                plateau_start=plateau_start_val,
                plateau_end=plateau_end_val,
                plateau_width=plateau_width_val,
                l1=l1_val,
                l2=l2_val,
            )
        )

        if (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:4d} "
                f"total={total.item():.5f} "
                f"E={e_loss.item():.5f} "
                f"range={r_loss.item():.5f} "
                f"P={p_loss.item():.5f} "
                f"topo={t_loss.item():.5f} "
                f"R={plateau_width_val:.4f} "
                f"[r0,r1]=[{plateau_start_val:.4f},{plateau_end_val:.4f}] "
                f"l1={l1_val:.4f} "
                f"l2={l2_val:.4f} "
                f"u2={c_loss.item():.4f}"
            )

    return controller, plant, history, fixed_x0
