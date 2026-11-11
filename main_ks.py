#!/usr/bin/env python3
"""
Kuramoto-Sivashinsky experiments.

    python main_ks.py --mode setup                       # reference values + all initial-condition pools
    python main_ks.py --mode pipeline --loss topology_recurrence --hpsearch --hp_seeds 0,1,2 --use_best_val
    python main_ks.py --mode summarize

Modes: setup, calibrate, scan, train, test, pipeline, summarize.  Every field of Cfg
(topo_attractor/ks/config.py) is also a command-line flag.  See README.md.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

import torch

from topo_attractor.ks.config import Cfg, LOSSES, check_cfg
from topo_attractor.ks.plant import KS
from topo_attractor.ks.initial_conditions import get_pool, TRAIN_POOL, VAL_POOL, TEST_POOL
from topo_attractor.ks.reference import calibrate, scan, load_reference
from topo_attractor.ks.experiment import train, run_test, pipeline, summarize


def setup(c: Cfg, seeds=None):
    """Compute the reference values and every initial-condition pool once.  Run this before starting several
    pipelines in parallel, so that they only read these files instead of creating them at the same time."""
    ks = KS(c)
    load_reference(c, ks)
    for s in (seeds if seeds is not None else range(c.n_seeds)):
        get_pool(ks, c, c.pool_size, TRAIN_POOL + s)
    get_pool(ks, c, c.pool_size, VAL_POOL)
    get_pool(ks, c, c.pool_size, TEST_POOL)
    print(f"setup done: reference values and pools in {c.outdir}/")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mode", choices=["setup", "calibrate", "scan", "train", "test", "pipeline", "summarize"], required=True)
    p.add_argument("--loss", choices=LOSSES, default=None)
    p.add_argument("--policy", choices=["mlp", "const", "zero"], default="mlp", help="test mode")
    p.add_argument("--run_dir", default=None, help="test mode: run directory holding policy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None, help="pipeline: comma-separated training seeds (default 0..n_seeds-1)")
    num = [f.name for f in fields(Cfg) if f.type in ("float", "int") and f.name != "seed"]
    for k in num:
        p.add_argument(f"--{k}", type=type(getattr(Cfg(), k)), default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--amps", default=None)
    p.add_argument("--pattern", default=None)
    p.add_argument("--resample_initials", action="store_true", help="draw new training starts every epoch")
    p.add_argument("--use_best_val", action="store_true",
                   help="freeze the best validation checkpoint (checked every --val_every epochs) instead of the final epoch")
    p.add_argument("--hpsearch", action="store_true", help="pipeline: grid search over epochs x lr x hidden first")
    p.add_argument("--hp_epochs", default=None, help="comma-separated epoch counts (default 300,600,900,1200,1500)")
    p.add_argument("--hp_lrs", default=None, help="comma-separated learning rates (default 3e-4,1e-3,3e-3)")
    p.add_argument("--hp_hiddens", default=None, help="comma-separated hidden widths (default 32,64,128)")
    p.add_argument("--hp_seeds", default=None, help="comma-separated seeds for the search (default: all training seeds)")
    p.add_argument("--require_lyap", action="store_true", help="qualification also needs max Lyapunov <= lam_tol")
    p.add_argument("--ood", action="store_true", help="also report an out-of-distribution test set (not counted)")
    p.add_argument("--no_cert", action="store_true")
    p.add_argument("--no_screen", action="store_true", help="test even if validation success rate < 1.0")
    p.add_argument("--smoke", action="store_true", help="tiny settings to check the code runs")
    a = p.parse_args()
    c = Cfg(seed=a.seed)
    if a.smoke:
        c.epochs, c.n_initial, c.steps, c.burn_in, c.N_cloud = 3, 4, 200, 100, 32
        c.pool_size, c.pool_burn = 16, 20.0
        c.eval_initial = 6               # full eval length: the reference orbit needs it to settle
        c.n_seeds, c.n_test_seeds = 2, 2
        c.n_lyap, c.T_lyap, c.lyap_discard, c.N_cert = 2, 20.0, 5.0, 40
        c.val_every, c.hp_epochs, c.hp_lrs, c.hp_hiddens = 2, (2, 4), (1e-3, 3e-3), (16,)
        c.amps = (0.65, 1.0)
        c.outdir = "runs_smoke"
    for k in num + ["outdir", "loss"]:
        if getattr(a, k, None) is not None:
            setattr(c, k, getattr(a, k))
    ints = lambda s: tuple(int(x) for x in s.split(","))
    flts = lambda s: tuple(float(x) for x in s.split(","))
    if a.amps: c.amps = flts(a.amps)
    if a.pattern: c.pattern = flts(a.pattern)
    if a.hp_epochs: c.hp_epochs = ints(a.hp_epochs)
    if a.hp_lrs: c.hp_lrs = flts(a.hp_lrs)
    if a.hp_hiddens: c.hp_hiddens = ints(a.hp_hiddens)
    if a.hp_seeds: c.hp_seeds = ints(a.hp_seeds)
    if a.resample_initials: c.resample_initials = True
    if a.use_best_val: c.use_best_val = True
    if a.hpsearch: c.hpsearch = True
    if a.require_lyap: c.require_lyap = True
    if a.ood: c.ood = True
    if a.no_cert: c.cert = False
    if c.threads:
        torch.set_num_threads(c.threads)
    if c.hpsearch and a.mode != "pipeline":
        raise SystemExit("--hpsearch is used with --mode pipeline")
    check_cfg(c)
    return c, a


if __name__ == "__main__":
    c, a = parse()
    if a.mode == "setup":
        setup(c, [int(s) for s in a.seeds.split(",")] if a.seeds else None)
    elif a.mode == "calibrate":
        calibrate(c)
    elif a.mode == "scan":
        scan(c)
    elif a.mode == "train":
        train(c)
    elif a.mode == "test":
        if a.policy == "mlp":
            assert a.run_dir, "--run_dir required for --policy mlp"
            run_test(c, a.run_dir, "mlp", screen=not a.no_screen)
        else:
            tag = "zero" if a.policy == "zero" else f"const_A{c.amp:g}"
            run_test(c, f"{c.outdir}/{tag}", a.policy, screen=not a.no_screen)
    elif a.mode == "pipeline":
        pipeline(c, [int(s) for s in a.seeds.split(",")] if a.seeds else None, screen=not a.no_screen)
    else:
        summarize(c)
