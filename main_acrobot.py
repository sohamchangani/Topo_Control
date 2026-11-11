#!/usr/bin/env python3
"""
Double pendulum (Acrobot) experiments.

Train one controller for one objective and seed, screen it on 32 held-out initial conditions and, if it
passes, retest it on 10 shared batches of 32 new initial conditions.

    python main_acrobot.py --mode topology_only --seed 0
    python main_acrobot.py --mode range_only --seed 0 --burn_in 0 --eval_burn_in 0   # task-only objectives
    python main_acrobot.py --summarize                                               # pass / perfect table

Objectives (--mode): energy_only, range_only, poincare_only, topology_only, topology_energy, topology_range,
topology_poincare.  See README.md.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from topo_attractor.acrobot.training import MODES, uses_topology, train
from topo_attractor.acrobot.evaluation import save_csv, final_evaluation, plot_training
from topo_attractor.acrobot.retest import RETEST_SEED_BASE, retest, summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Acrobot attractor-control losses for L4DC."
    )

    p.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="objective to train (required unless --retest_only or --summarize)",
    )
    p.add_argument("--seed", type=int, default=0)

    # Plant.
    p.add_argument("--g", type=float, default=9.81)
    p.add_argument("--damping", type=float, default=0.30)
    p.add_argument("--u_max", type=float, default=8.0)

    # Controller.
    p.add_argument("--hidden", type=int, default=64)

    # Training rollout.
    p.add_argument("--n_initial", type=int, default=8)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--burn_in", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--resample_initials", action="store_true")
    p.add_argument("--initial_q_range", type=float, default=0.60)
    p.add_argument("--initial_dq_range", type=float, default=0.40)

    # Common energy warm start.
    p.add_argument("--warm_start_steps", type=int, default=300)
    p.add_argument("--warm_start_k", type=float, default=1.0)
    p.add_argument("--warm_start_batch", type=int, default=1024)
    p.add_argument("--warm_start_lr", type=float, default=2e-3)

    # Energy baseline.
    p.add_argument("--energy_target", type=float, default=-24.0)

    # Range baseline. These are research hyperparameters.
    p.add_argument("--q1_amp_target", type=float, default=1.0)
    p.add_argument("--q2_amp_target", type=float, default=1.0)

    # Differentiable Poincare baseline.
    p.add_argument("--section_sigma", type=float, default=0.20)
    p.add_argument("--direction_sigma", type=float, default=0.20)
    p.add_argument("--min_section_mass", type=float, default=0.02)
    p.add_argument("--lambda_section_mass", type=float, default=10.0)
    p.add_argument(
        "--min_crossing_speed",
        type=float,
        default=0.50,
        help="Minimum weighted positive dq1 on the soft Poincare section.",
    )
    p.add_argument(
        "--lambda_crossing_speed",
        type=float,
        default=10.0,
        help="Penalty weight preventing the equilibrium shortcut.",
    )

    # Fixed state metric for PH / Poincare embedding.
    p.add_argument("--velocity_scale", type=float, default=4.0)

    # PH topology.
    p.add_argument("--ph_points", type=int, default=96)
    p.add_argument("--eval_ph_points", type=int, default=192)
    p.add_argument("--ph_threads", type=int, default=4)
    p.add_argument(
        "--max_diameter",
        type=float,
        default=0.0,
        help="<=0 means use full sampled-cloud diameter",
    )
    # Loss weights.
    p.add_argument("--lambda_energy", type=float, default=1.0)
    p.add_argument("--lambda_range", type=float, default=10.0)
    p.add_argument("--lambda_poincare", type=float, default=10.0)
    p.add_argument("--lambda_topology", type=float, default=10.0)
    p.add_argument("--lambda_control", type=float, default=0.01)

    # Evaluation.
    p.add_argument("--eval_initial", type=int, default=32)
    p.add_argument("--eval_steps", type=int, default=1200)
    p.add_argument("--eval_burn_in", type=int, default=600)
    p.add_argument("--eval_q_range", type=float, default=0.75)
    p.add_argument("--eval_dq_range", type=float, default=0.50)

    p.add_argument("--poincare_last_k", type=int, default=5)
    p.add_argument("--poincare_within_tol", type=float, default=0.05)
    p.add_argument("--poincare_between_tol", type=float, default=0.10)

    p.add_argument("--plot_trajectories", type=int, default=16)
    p.add_argument("--print_every", type=int, default=10)

    p.add_argument(
        "--outdir",
        type=str,
        default="runs_acrobot",
    )

    # Retest of passing controllers and summary (see topo_attractor/acrobot/retest.py).
    p.add_argument("--no_retest", action="store_true",
                   help="do not retest a controller that passes screening")
    p.add_argument("--retest_sets", type=int, default=10)
    p.add_argument("--retest_seed_base", type=int, default=RETEST_SEED_BASE,
                   help="retest set k uses generator seed retest_seed_base + k (shared by all controllers)")
    p.add_argument("--retest_only", default=None, metavar="RUN_DIR",
                   help="only retest the saved controller in RUN_DIR")
    p.add_argument("--summarize", action="store_true",
                   help="only print and save the pass / perfect table for --outdir")

    return p


def validate_args(args) -> None:
    if args.burn_in >= args.steps:
        raise ValueError("burn_in must be smaller than steps")

    if args.eval_burn_in >= args.eval_steps:
        raise ValueError(
            "eval_burn_in must be smaller than eval_steps"
        )

    if args.velocity_scale <= 0:
        raise ValueError("velocity_scale must be positive")

    if args.section_sigma <= 0 or args.direction_sigma <= 0:
        raise ValueError(
            "section_sigma and direction_sigma must be positive"
        )
    if args.min_crossing_speed < 0:
        raise ValueError("min_crossing_speed must be nonnegative")
    if args.lambda_crossing_speed < 0:
        raise ValueError("lambda_crossing_speed must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        summarize(args.outdir)
        return
    if args.retest_only:
        retest(args.retest_only, n_sets=args.retest_sets, seed_base=args.retest_seed_base)
        return
    if args.mode is None:
        raise SystemExit("--mode is required (or use --retest_only / --summarize)")
    validate_args(args)

    args.run_dir = os.path.join(
        args.outdir,
        f"{args.mode}_b{args.damping:g}_seed{args.seed}",
    )
    os.makedirs(args.run_dir, exist_ok=True)

    print("=" * 72)
    print("Acrobot attractor-control experiment")
    print("=" * 72)
    print(f"mode              : {args.mode}")
    print(f"seed              : {args.seed}")
    print(f"warm_start_steps  : {args.warm_start_steps}")
    print(f"energy target     : {args.energy_target}")
    print(
        f"range targets     : q1={args.q1_amp_target}, "
        f"q2={args.q2_amp_target}"
    )
    if uses_topology(args.mode):
        print(
            "topology target   : maximize R = r1-r0, where [r0,r1) "
            "is the longest interval with beta_1(r)=1"
        )
        if args.max_diameter > 0:
            print(
                "WARNING           : positive max_diameter truncates the VR "
                "filtration and can censor H1 bars. Use 0 unless the cutoff "
                "is known to exceed all relevant death scales."
            )
    print()

    controller, plant, history, train_x0 = train(args)

    torch.save(
        {
            "state_dict": controller.state_dict(),
            "mode": args.mode,
            "seed": args.seed,
            "args": vars(args),
        },
        os.path.join(
            args.run_dir,
            "controller.pt",
        ),
    )

    save_csv(
        os.path.join(
            args.run_dir,
            "training_initial_conditions.csv",
        ),
        ["q1", "q2", "dq1", "dq2"],
        train_x0.detach().cpu().numpy().tolist(),
    )

    hist_rows = [
        [
            r.epoch,
            r.total,
            r.energy,
            r.range_loss,
            r.poincare,
            r.topology,
            r.control,
            r.plateau_start,
            r.plateau_end,
            r.plateau_width,
            r.l1,
            r.l2,
        ]
        for r in history
    ]

    save_csv(
        os.path.join(
            args.run_dir,
            "training_history.csv",
        ),
        [
            "epoch",
            "total",
            "energy_diagnostic",
            "range_diagnostic",
            "poincare_diagnostic",
            "topology_loss",
            "control_effort",
            "plateau_start_r0",
            "plateau_end_r1",
            "plateau_width_R",
            "l1",
            "l2",
        ],
        hist_rows,
    )

    plot_training(
        history,
        args,
    )

    summary = final_evaluation(
        controller,
        plant,
        args,
    )

    with open(
        os.path.join(
            args.run_dir,
            "summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nFinal evaluation")
    print("----------------")
    for key, value in summary.items():
        print(f"{key:36s}: {value}")

    print(f"\nSaved to: {args.run_dir}")

    # Screening passes only if all eval_initial arms settle; passing controllers are retested,
    # weights frozen, on the shared retest sets.
    if summary["poincare_success_rate"] == 1.0 and not args.no_retest:
        retest(args.run_dir, n_sets=args.retest_sets, seed_base=args.retest_seed_base)


if __name__ == "__main__":
    main()
