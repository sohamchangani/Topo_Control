"""Screening evaluation on the held-out initial conditions, plots and CSV output."""
from __future__ import annotations
import csv, os
from typing import Dict, List, Sequence

import torch

from .plant import Acrobot, rollout
from .controller import Controller
from .losses import OINEUS_AVAILABLE, range_loss, make_topology_cloud, differentiable_h1_diagram, one_loop_plateau_loss
from .validator import hard_upward_crossings, poincare_evaluation
from .training import TrainRecord, uses_topology


def save_csv(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def final_evaluation(
    controller: Controller,
    plant: Acrobot,
    args,
) -> Dict[str, float]:
    gen = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed + 5001)

    eval_x0 = plant.sample_initial(
        args.eval_initial,
        generator=gen,
        q_range=args.eval_q_range,
        dq_range=args.eval_dq_range,
    )

    controller.eval()

    with torch.no_grad():
        traj, controls = rollout(
            plant,
            controller,
            eval_x0,
            args.dt,
            args.eval_steps,
        )

    burn = args.eval_burn_in
    late = traj[burn:]

    E = plant.energy(late)

    r_loss, q1_max, q1_min, q2_max, q2_min = range_loss(
        traj,
        burn,
        args.q1_amp_target,
        args.q2_amp_target,
    )

    p_metrics = poincare_evaluation(
        traj,
        burn,
        last_k=args.poincare_last_k,
        within_tol=args.poincare_within_tol,
        between_tol=args.poincare_between_tol,
        velocity_scale=args.velocity_scale,
    )

    summary: Dict[str, float] = {
        "mean_energy": float(E.mean().item()),
        "std_energy": float(E.std(unbiased=False).item()),
        "energy_error_abs": float(
            torch.mean(
                torch.abs(E - args.energy_target)
            ).item()
        ),
        "range_loss": float(r_loss.item()),
        "q1_max_mean": float(q1_max.mean().item()),
        "q1_min_mean": float(q1_min.mean().item()),
        "q2_max_mean": float(q2_max.mean().item()),
        "q2_min_mean": float(q2_min.mean().item()),
        "mean_u2": float(
            controls.square().mean().item()
        ),
        "rms_velocity": float(
            torch.sqrt(late[..., 2:].square().mean()).item()
        ),
        "max_abs_velocity": float(
            late[..., 2:].abs().max().item()
        ),
        **p_metrics,
    }

    # Evaluate topology for every mode if Oineus is available.
    if OINEUS_AVAILABLE:
        with torch.no_grad():
            cloud = make_topology_cloud(
                traj,
                burn,
                args.eval_ph_points,
                args.velocity_scale,
            )

        # Evaluation need not backpropagate, but using the same barcode objective
        # makes the training and evaluation topology diagnostics directly comparable.
        dgm1 = differentiable_h1_diagram(
            cloud,
            max_diameter=args.max_diameter,
            n_threads=args.ph_threads,
        )

        (
            _,
            plateau_start,
            plateau_end,
            plateau_width,
            l1,
            l2,
            lifetimes,
        ) = one_loop_plateau_loss(dgm1, cloud)

        cloud_diameter = float(
            torch.cdist(cloud, cloud, p=2).max().detach().item()
        ) if cloud.shape[0] > 1 else 0.0

        summary.update(
            {
                "h1_l1": float(l1.detach().item()),
                "h1_l2": float(l2.detach().item()),
                "one_loop_plateau_start": float(plateau_start.detach().item()),
                "one_loop_plateau_end": float(plateau_end.detach().item()),
                "one_loop_plateau_width": float(plateau_width.detach().item()),
                "topology_cloud_diameter": cloud_diameter,
                "plateau_width_over_diameter": (
                    float(plateau_width.detach().item()) / cloud_diameter
                    if cloud_diameter > 0.0 else 0.0
                ),
                "h1_bar_count": int(lifetimes.numel()),
            }
        )

    # Save evaluation ICs and per-trajectory ranges.
    outdir = args.run_dir

    save_csv(
        os.path.join(
            outdir,
            "evaluation_initial_conditions.csv",
        ),
        ["q1", "q2", "dq1", "dq2"],
        eval_x0.detach().cpu().numpy().tolist(),
    )

    range_rows = []
    for j in range(args.eval_initial):
        range_rows.append(
            [
                j,
                float(q1_max[j].item()),
                float(q1_min[j].item()),
                float(q2_max[j].item()),
                float(q2_min[j].item()),
            ]
        )

    save_csv(
        os.path.join(
            outdir,
            "evaluation_ranges.csv",
        ),
        [
            "trajectory",
            "q1_max",
            "q1_min",
            "q2_max",
            "q2_min",
        ],
        range_rows,
    )

    make_plots(
        traj,
        plant,
        args,
    )

    return summary


def make_plots(
    traj: torch.Tensor,
    plant: Acrobot,
    args,
) -> None:
    import matplotlib.pyplot as plt

    x = traj.detach().cpu().numpy()

    start = args.eval_burn_in
    n_show = min(args.plot_trajectories, x.shape[1])

    # q1-dq1 projection
    plt.figure(figsize=(6, 5))
    for j in range(n_show):
        plt.plot(
            x[start:, j, 0],
            x[start:, j, 2],
            linewidth=0.8,
        )
    plt.xlabel("q1")
    plt.ylabel("dq1")
    plt.title(f"{args.mode}: late q1-dq1 projection")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.run_dir, "phase_q1.png"),
        dpi=180,
    )
    plt.close()

    # q2-dq2 projection
    plt.figure(figsize=(6, 5))
    for j in range(n_show):
        plt.plot(
            x[start:, j, 1],
            x[start:, j, 3],
            linewidth=0.8,
        )
    plt.xlabel("q2")
    plt.ylabel("dq2")
    plt.title(f"{args.mode}: late q2-dq2 projection")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.run_dir, "phase_q2.png"),
        dpi=180,
    )
    plt.close()

    # Hard Poincare section
    crossings = hard_upward_crossings(
        traj,
        args.eval_burn_in,
    )

    plt.figure(figsize=(6, 5))
    for j in range(min(n_show, len(crossings))):
        arr = crossings[j]
        if len(arr):
            plt.scatter(
                arr[:, 0],
                arr[:, 2],
                s=12,
            )
    plt.xlabel("q2 at q1=0 upward crossing")
    plt.ylabel("dq2 at q1=0 upward crossing")
    plt.title("Poincare section")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            args.run_dir,
            "poincare_section.png",
        ),
        dpi=180,
    )
    plt.close()

    # Energy traces
    with torch.no_grad():
        E = plant.energy(traj).cpu().numpy()

    plt.figure(figsize=(7, 4))
    for j in range(min(n_show, E.shape[1])):
        plt.plot(E[:, j], linewidth=0.8)
    plt.axhline(
        args.energy_target,
        linestyle="--",
        linewidth=1.0,
        label="E target",
    )
    plt.xlabel("time step")
    plt.ylabel("mechanical energy")
    plt.title("Energy trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            args.run_dir,
            "energy_trajectories.png",
        ),
        dpi=180,
    )
    plt.close()


def plot_training(
    history: List[TrainRecord],
    args,
) -> None:
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in history]

    plt.figure(figsize=(7, 4))
    plt.plot(
        epochs,
        [r.total for r in history],
        label="total",
    )
    plt.plot(
        epochs,
        [r.energy for r in history],
        label="energy diagnostic",
    )
    plt.plot(
        epochs,
        [r.range_loss for r in history],
        label="range diagnostic",
    )
    plt.plot(
        epochs,
        [r.poincare for r in history],
        label="Poincare diagnostic",
    )
    if uses_topology(args.mode):
        plt.plot(
            epochs,
            [r.topology for r in history],
            label="topology",
        )

    plt.yscale("symlog", linthresh=1e-5)
    plt.xlabel("epoch")
    plt.ylabel("loss / diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            args.run_dir,
            "training_losses.png",
        ),
        dpi=180,
    )
    plt.close()

    if uses_topology(args.mode):
        # Main topology diagnostic: the longest beta_1=1 plateau.
        plt.figure(figsize=(7, 4))
        plt.plot(
            epochs,
            [r.plateau_width for r in history],
            label="one-loop plateau width R",
        )
        plt.plot(
            epochs,
            [r.plateau_start for r in history],
            label="plateau start r0",
        )
        plt.plot(
            epochs,
            [r.plateau_end for r in history],
            label="plateau end r1",
        )
        plt.xlabel("epoch")
        plt.ylabel("filtration scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.run_dir,
                "topology_one_loop_plateau.png",
            ),
            dpi=180,
        )
        plt.close()

        # Retain top-two persistence lifetimes as diagnostics only.
        plt.figure(figsize=(7, 4))
        plt.plot(
            epochs,
            [r.l1 for r in history],
            label="largest H1 lifetime l1",
        )
        plt.plot(
            epochs,
            [r.l2 for r in history],
            label="second H1 lifetime l2",
        )
        plt.xlabel("epoch")
        plt.ylabel("persistence lifetime")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.run_dir,
                "topology_lifetimes.png",
            ),
            dpi=180,
        )
        plt.close()
