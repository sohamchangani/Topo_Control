"""Retest of controllers that pass screening, and the pass / perfect summary table.

A controller passes screening if all of its 32 screening arms settle (poincare_success_rate == 1.0 in
summary.json).  A passing controller is retested, with its weights frozen, on `n_sets` batches of new initial
conditions drawn from the same box as the screening set.  Retest set k uses the generator seed
`seed_base + k`, so every controller is retested on the same starting points.  A controller is *perfect* if
it passes screening and every retest arm settles.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from .plant import Acrobot, rollout
from .controller import Controller
from .validator import poincare_evaluation

RETEST_SEED_BASE = 50000


@torch.no_grad()
def retest(run_dir: str, n_sets: int = 10, seed_base: int = RETEST_SEED_BASE) -> dict:
    """Retest the controller saved in run_dir/controller.pt with the settings it was trained and screened
    with (test length, test burn-in, validator tolerances).  Writes run_dir/retest.json."""
    ckpt = torch.load(os.path.join(run_dir, "controller.pt"), map_location="cpu")
    a = argparse.Namespace(**ckpt["args"])

    plant = Acrobot(g=a.g, damping=a.damping, u_max=a.u_max)
    controller = Controller(hidden=a.hidden, u_max=a.u_max)
    controller.load_state_dict(ckpt["state_dict"])
    controller.eval()

    sets = []
    for k in range(1, n_sets + 1):
        gen = torch.Generator(device="cpu").manual_seed(seed_base + k)
        x0 = plant.sample_initial(a.eval_initial, generator=gen,
                                  q_range=a.eval_q_range, dq_range=a.eval_dq_range)
        traj, _ = rollout(plant, controller, x0, a.dt, a.eval_steps)
        m = poincare_evaluation(traj, a.eval_burn_in, last_k=a.poincare_last_k,
                                within_tol=a.poincare_within_tol, between_tol=a.poincare_between_tol,
                                velocity_scale=a.velocity_scale)
        settled = int(round(m["poincare_success_rate"] * a.eval_initial))
        sets.append(dict(set=k, generator_seed=seed_base + k, success_rate=m["poincare_success_rate"],
                         settled=settled, n=a.eval_initial))
        print(f"retest set {k:2d}: {settled}/{a.eval_initial} settled", flush=True)

    rates = [s["success_rate"] for s in sets]
    out = dict(seed_base=seed_base, n_sets=n_sets, success_rates=rates,
               n_sets_at_1=int(sum(r == 1.0 for r in rates)),
               all_sets_at_1=bool(all(r == 1.0 for r in rates)),
               settled=int(sum(s["settled"] for s in sets)),
               total=int(sum(s["n"] for s in sets)), sets=sets)
    with open(os.path.join(run_dir, "retest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"retest: {out['settled']}/{out['total']} arms settled, "
          f"{out['n_sets_at_1']}/{n_sets} sets with every arm settled", flush=True)
    return out


def summarize(outdir: str) -> str:
    """Pass / perfect / settled retest arms per objective, over every run in outdir.  Writes outdir/summary.md."""
    pattern = re.compile(r"^(?P<mode>.+)_b[^_]+_seed(?P<seed>\d+)$")
    groups: dict = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*", "summary.json"))):
        run_dir = os.path.dirname(f)
        m = pattern.match(os.path.basename(run_dir))
        if not m:
            continue
        screen = json.load(open(f))
        rt_path = os.path.join(run_dir, "retest.json")
        rt = json.load(open(rt_path)) if os.path.exists(rt_path) else None
        groups.setdefault(m["mode"], []).append((int(m["seed"]), screen, rt))
    if not groups:
        print("no summary.json found under", outdir)
        return ""

    lines = ["## Per objective", "",
             "| objective | seeds | mean screening | pass | perfect | settled retest arms (passing) |",
             "|---|---|---|---|---|---|"]
    detail = ["", "## Per run", "",
              "| objective | seed | screening | pass | retest settled | perfect |",
              "|---|---|---|---|---|---|"]
    for mode, runs in groups.items():
        runs.sort(key=lambda r: r[0])
        passed = [r for r in runs if r[1]["poincare_success_rate"] == 1.0]
        tested = [r for r in passed if r[2] is not None]
        perfect = [r for r in tested if r[2]["all_sets_at_1"]]
        arms = (f"{sum(r[2]['settled'] for r in tested)}/{sum(r[2]['total'] for r in tested)}"
                if tested else "---")
        mean = np.mean([r[1]["poincare_success_rate"] for r in runs])
        lines.append(f"| {mode} | {len(runs)} | {mean:.2f} | {len(passed)}/{len(runs)} | "
                     f"{len(perfect)}/{len(runs)} | {arms} |")
        for seed, screen, rt in runs:
            p = screen["poincare_success_rate"] == 1.0
            rts = f"{rt['settled']}/{rt['total']}" if rt else ("not run" if p else "---")
            detail.append(f"| {mode} | {seed} | {screen['poincare_success_rate']:.2f} | {'yes' if p else 'no'} | "
                          f"{rts} | {'yes' if rt and rt['all_sets_at_1'] else 'no'} |")
    txt = "\n".join(lines + detail)
    print(txt)
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt
