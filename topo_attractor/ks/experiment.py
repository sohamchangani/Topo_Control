"""Protocol: training, validation (= screening) with best-checkpoint selection, retest on the shared test
sets, hyperparameter search and summary tables."""
from __future__ import annotations
import copy, glob, itertools, json, math, os, shutil, time, warnings
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from .config import Cfg, LOSSES, check_cfg
from .plant import KS
from .policies import build_policy, weights_sha
from .initial_conditions import training_ics, validation_ics, val_seed_of, test_ics, test_seeds
from .losses import compute_loss, check_recurrence_window
from .validator import evaluate_set, short
from .plots import plot_phase, plot_history
from .reference import load_reference, proto_of


def variant(c: Cfg):
    """Run-name prefix: the loss, plus _bestval when the best validation checkpoint is frozen."""
    return c.loss + ("_bestval" if c.use_best_val else "")


def val_success(policy, ks, c: Cfg, v0):
    """Validation success rate only (no Lyapunov / certificate); used to pick the best checkpoint."""
    fin, _ = evaluate_set(policy, ks, c, v0, lyap=False, cert=False)
    return fin["success_rate"]


def validate(c: Cfg, policy, ks, run_dir, kind="mlp", meta=None):
    """Frozen weights -> validation set of training seed c.seed (eval_initial starts drawn with generator
    val_seed_offset + seed, eval_steps timesteps).  Qualified iff success rate 1.0."""
    policy.eval()
    fin, Zw = evaluate_set(policy, ks, c, validation_ics(ks, c), cert=c.cert, n_threads=c.threads or 2)
    os.makedirs(run_dir, exist_ok=True)
    plot_phase(Zw, fin, f"{run_dir}/phase_validation.png")
    if meta is None and os.path.exists(f"{run_dir}/meta.json"):
        meta = json.load(open(f"{run_dir}/meta.json"))
    out = dict(kind=kind, val_seed=val_seed_of(c), success_rate=fin["success_rate"], qualified=fin["good"],
               weights_sha=weights_sha(policy), proto=proto_of(c), meta=meta or {},
               result={k: v for k, v in fin.items() if k != "settled"})
    json.dump(out, open(f"{run_dir}/validation.json", "w"), indent=1)
    print(short(f"validation (seed {val_seed_of(c)})", fin), "-> QUALIFIED" if fin["good"] else "-> not qualified",
          flush=True)
    if "certificate" in fin:
        cr = fin["certificate"]
        print(f"   certificate: R={cr['R']:.3f}  eps_hat={cr['eps_hat']:.3f}  "
              f"R > {c.cert_factor:g}*eps_hat: {cr['certified']}", flush=True)
    return out


def freeze(c: Cfg, policy, best, ep, ks, d, x0_fixed):
    """Save the controller frozen at epoch ep (final weights, or the best validation checkpoint so far)
    in directory d, then validate it."""
    os.makedirs(d, exist_ok=True)
    last = copy.deepcopy(policy.state_dict())
    if c.use_best_val:
        chosen, ck_ep, ck_val = best[2], best[1], best[0]
    else:
        chosen, ck_ep, ck_val = last, ep, None
    torch.save(last, f"{d}/policy_last.pt")
    torch.save(chosen, f"{d}/policy.pt")                              # the frozen controller
    torch.save(x0_fixed, f"{d}/training_initial_conditions.pt")
    meta = dict(loss=c.loss, seed=c.seed, epochs=ep, lr=c.lr, hidden=c.hidden,
                checkpoint="best_val" if c.use_best_val else "final", checkpoint_epoch=ck_ep,
                checkpoint_val_success=ck_val, resample_initials=c.resample_initials)
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    frozen = copy.deepcopy(policy)
    frozen.load_state_dict(chosen); frozen.requires_grad_(False)
    what = (f"best validation checkpoint: epoch {ck_ep} of {ep}, success {ck_val:.2f}" if c.use_best_val
            else f"final weights, epoch {ep}")
    print(f"saved {d}/policy.pt  ({what})", flush=True)
    validate(c, frozen, ks, d, meta=meta)


def train(c: Cfg, snapshots=None, run_dir=None):
    """One training seed: n_initial fixed starts; each epoch = (steps-timestep rollout -> loss on the last
    steps - burn_in timesteps -> one Adam step).  At every snapshot epoch E (default: only the last epoch)
    the frozen controller is the epoch-E weights, or with use_best_val the best validation checkpoint among
    epochs <= E.  It is saved and validated in run_dir (one snapshot) or run_dir/ep<E> (several)."""
    assert c.loss in LOSSES, f"--loss must be one of {LOSSES}"
    check_cfg(c)
    snaps = sorted(set(int(e) for e in (snapshots or [c.epochs])))
    c.epochs = snaps[-1]
    torch.manual_seed(c.seed); np.random.seed(c.seed)
    gen = torch.Generator().manual_seed(c.seed)              # FPS start + jitter of the PH cloud
    nt = c.threads or 2
    ks = KS(c)
    load_reference(c, ks)       # baselines need E_star / P_ref; the success rate needs v_ref for its floor
    c.P_ref *= c.P_scale
    if "recurrence" in c.loss:
        check_recurrence_window(c)
    x0_fixed = training_ics(ks, c)
    v_val = validation_ics(ks, c) if c.use_best_val else None
    run_dir = run_dir or f"{c.outdir}/{variant(c)}_seed{c.seed}"
    os.makedirs(run_dir, exist_ok=True)
    policy = build_policy(c, ks)
    opt = torch.optim.Adam(policy.parameters(), lr=c.lr)
    history, best, t0 = [], None, time.time()
    for ep in range(1, c.epochs + 1):
        v0 = training_ics(ks, c, epoch=ep) if c.resample_initials else x0_fixed
        loss, st = compute_loss(policy, ks, v0, c, gen, nt)
        opt.zero_grad()
        gn = float("nan")
        if torch.isfinite(loss) and loss.requires_grad:
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(policy.parameters(), c.clip))
            if math.isfinite(gn):
                opt.step()
            else:
                warnings.warn(f"epoch {ep}: non-finite gradient, step skipped (lower --tbptt)")
        st.update(epoch=ep, grad_norm=gn)
        if c.use_best_val and (ep % c.val_every == 0 or ep in snaps):
            sr = val_success(policy, ks, c, v_val)
            st["val_success"] = sr
            if best is None or sr >= best[0]:                 # ties -> the later epoch
                best = (sr, ep, copy.deepcopy(policy.state_dict()))
        history.append(st)
        if ep == 1 or ep % 10 == 0 or ep == c.epochs or "val_success" in st:
            extra = (f"R={st['R']:.3f} Rs={st['R_signed']:.2f} r_conn={st['r_conn']:.2f} l1={st['l1']:.2f} " if "R" in st else "") + \
                    (f"L_E={st['L_energy']:.3f} " if "L_energy" in st else "") + \
                    (f"L_rec={st['L_rec']:.3f} floor={st['L_floor']:.2f} " if "L_rec" in st else "") + \
                    (f"val={st['val_success']:.2f} " if "val_success" in st else "")
            print(f"[{c.loss} s{c.seed}] epoch {ep:4d} loss={st['loss']:.4f} {extra}"
                  f"eff={st['effort']:.3f} |g|={gn:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if ep in snaps:
            freeze(c, policy, best, ep, ks, run_dir if len(snaps) == 1 else f"{run_dir}/ep{ep}", x0_fixed)
    json.dump(dict(cfg={k: v for k, v in asdict(c).items()}, snapshots=snaps, history=history,
                   train_seconds=time.time() - t0), open(f"{run_dir}/history.json", "w"), indent=1)
    plot_history(history, f"{run_dir}/curves.png")
    return run_dir


def run_test(c: Cfg, run_dir, kind="mlp", screen=True):
    """Frozen weights -> 10 test seeds (each eval_initial fresh starts, eval_steps timesteps).
    Only controllers with validation success rate 1.0 are tested (unless screen=False)."""
    check_cfg(c)
    meta = json.load(open(f"{run_dir}/meta.json")) if os.path.exists(f"{run_dir}/meta.json") else {}
    if "seed" in meta:                                    # validate on the run's own seed + 5001
        c = copy.deepcopy(c); c.seed = meta["seed"]
    ks = KS(c); load_reference(c, ks)
    nt = c.threads or 2
    policy = build_policy(c, ks, kind, f"{run_dir}/policy.pt" if kind == "mlp" else None)
    policy.eval(); policy.requires_grad_(False)
    sha = weights_sha(policy)
    vpath = f"{run_dir}/validation.json"
    val = json.load(open(vpath)) if os.path.exists(vpath) else None
    if (val is None or val.get("weights_sha") != sha or val.get("proto") != proto_of(c)
            or val.get("val_seed") != val_seed_of(c)):
        val = validate(c, policy, ks, run_dir, kind)
    base = dict(kind=kind, weights_sha=sha, meta=meta, validation_success_rate=val["success_rate"],
                qualified=val["qualified"])
    if screen and not val["qualified"]:
        print(f"   validation success rate {val['success_rate']:.2f} < 1.0 -> not tested", flush=True)
        json.dump(dict(base, tested=False), open(f"{run_dir}/test.json", "w"), indent=1)
        return base
    sets = {}
    for s in test_seeds(c):
        fin, _ = evaluate_set(policy, ks, c, test_ics(ks, c, s), n_threads=nt)
        sets[str(s)] = {k: v for k, v in fin.items() if k != "settled"}
        print(short(f"test seed {s}", fin), flush=True)
    rates = [sets[str(s)]["success_rate"] for s in test_seeds(c)]
    out = dict(base, tested=True, test_seeds=test_seeds(c), success_rates=rates,
               mean_success_rate=float(np.mean(rates)),
               n_test_seeds_at_1=int(sum(r == 1.0 for r in rates)),
               all_test_seeds_at_1=bool(all(r == 1.0 for r in rates)),
               starts_settled=int(sum(sets[str(s)]["n_settled"] for s in test_seeds(c))),
               starts_total=int(sum(sets[str(s)]["n_starts"] for s in test_seeds(c))),
               sets=sets)
    if c.ood:
        fin, _ = evaluate_set(policy, ks, c, test_ics(ks, c, 99, "ood"), n_threads=nt)
        out["ood"] = {k: v for k, v in fin.items() if k != "settled"}
        print(short("out-of-distribution", fin), "(reported, not counted)", flush=True)
    print(f"   test: mean success rate {out['mean_success_rate']:.3f}, "
          f"{out['n_test_seeds_at_1']}/{len(rates)} test seeds at 1.0, "
          f"{out['starts_settled']}/{out['starts_total']} starts settled", flush=True)
    json.dump(out, open(f"{run_dir}/test.json", "w"), indent=1)
    return out


def pipeline(c: Cfg, seeds=None, screen=True):
    """One loss: for each training seed, train -> validate -> test if qualified; then summarize.
    With c.hpsearch the hyperparameters are chosen first (see hpsearch)."""
    seeds = list(seeds) if seeds is not None else list(range(c.n_seeds))
    if c.hpsearch:
        return hpsearch(c, seeds, screen)
    for s in seeds:
        cs = copy.deepcopy(c); cs.seed = s
        print(f"\n===== {variant(cs)}, training seed {s} =====", flush=True)
        run_dir = train(cs)
        run_test(copy.deepcopy(cs), run_dir, "mlp", screen)
    summarize(c)


def hp_tag(lr, hidden):
    return f"lr{lr:g}_h{hidden}"


def hpsearch(c: Cfg, seeds, screen=True):
    """Grid search over (lr, hidden, epochs), scored on VALIDATION only; the test seeds are never used.
    Each (lr, hidden) pair is trained once per search seed up to max(hp_epochs) with a snapshot (frozen +
    validated) at every epoch count in hp_epochs.  Ranking: most search seeds with validation success 1.0,
    then highest mean validation success, then fewer epochs, then smaller hidden width.  The winning
    configuration's controllers (one per training seed) are then tested exactly as in the normal pipeline.
    Finished (config, seed) runs are reused, so an interrupted search can be restarted with the same command."""
    assert c.loss in LOSSES, f"--loss must be one of {LOSSES}"
    search_seeds = list(c.hp_seeds) if c.hp_seeds else list(seeds)
    snaps = sorted(set(int(e) for e in c.hp_epochs))
    root = f"{c.outdir}/hpsearch/{variant(c)}"
    grid = list(itertools.product([float(x) for x in c.hp_lrs], [int(x) for x in c.hp_hiddens]))
    print(f"hyperparameter search for {variant(c)}: {len(grid)} (lr, hidden) pairs x epochs {snaps}, "
          f"search seeds {search_seeds}", flush=True)
    rows = []
    for i, (lr, h) in enumerate(grid):
        tag = hp_tag(lr, h)
        for s in search_seeds:
            d = f"{root}/{tag}/seed{s}"
            if all(os.path.exists(f"{d}/ep{e}/validation.json") for e in snaps) and os.path.exists(f"{d}/history.json"):
                print(f"[hpsearch] {tag} seed {s}: already done, reusing", flush=True)
                continue
            cs = copy.deepcopy(c); cs.seed, cs.lr, cs.hidden = s, lr, h
            print(f"\n===== hpsearch {variant(c)}: config {i + 1}/{len(grid)} ({tag}), seed {s} =====", flush=True)
            train(cs, snapshots=snaps, run_dir=d)
        for e in snaps:
            vals = [json.load(open(f"{root}/{tag}/seed{s}/ep{e}/validation.json")) for s in search_seeds]
            rates = [v["success_rate"] for v in vals]
            rows.append(dict(lr=lr, hidden=h, epochs=e, n_qualified=int(sum(v["qualified"] for v in vals)),
                             mean_val_success=float(np.mean(rates)), val_success=rates))
    rows.sort(key=lambda r: (-r["n_qualified"], -r["mean_val_success"], r["epochs"], r["hidden"]))
    best = rows[0]
    lines = [f"## Hyperparameter search: {variant(c)} (validation only; search seeds {search_seeds})", "",
             "| rank | lr | hidden | epochs | seeds with validation = 1.0 | mean validation success | per seed |",
             "|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        lines.append(f"| {k} | {r['lr']:g} | {r['hidden']} | {r['epochs']} | {r['n_qualified']}/{len(search_seeds)} | "
                     f"{r['mean_val_success']:.3f} | {' '.join(f'{x:.2f}' for x in r['val_success'])} |")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    os.makedirs(root, exist_ok=True)
    open(f"{root}/hpsearch.md", "w").write(txt + "\n")
    json.dump(dict(search_seeds=search_seeds, grid=dict(lrs=list(c.hp_lrs), hiddens=list(c.hp_hiddens), epochs=snaps),
                   ranking=rows, best=best), open(f"{root}/hpsearch.json", "w"), indent=1)
    print(f"\nselected: lr={best['lr']:g} hidden={best['hidden']} epochs={best['epochs']}", flush=True)

    # adopt the winner: one controller per training seed, then test as in the normal pipeline
    for s in seeds:
        cs = copy.deepcopy(c); cs.seed, cs.lr, cs.hidden, cs.epochs = s, best["lr"], best["hidden"], best["epochs"]
        dst = f"{c.outdir}/{variant(c)}_hp_seed{s}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if s in search_seeds:
            src = f"{root}/{hp_tag(best['lr'], best['hidden'])}/seed{s}"
            shutil.copytree(f"{src}/ep{best['epochs']}", dst)
            for f in ("history.json", "curves.png"):
                if os.path.exists(f"{src}/{f}"):
                    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
        else:
            print(f"\n===== {variant(c)} with selected hyperparameters, training seed {s} =====", flush=True)
            train(cs, run_dir=dst)
        run_test(cs, dst, "mlp", screen)
    summarize(c)


def summarize(c: Cfg):
    """Per-loss table over every <outdir>/<run>/validation.json (+ test.json)."""
    groups = {}
    for vf in sorted(glob.glob(f"{c.outdir}/*/validation.json")):
        rd = os.path.dirname(vf); name = os.path.basename(rd)
        mode = name.rsplit("_seed", 1)[0] if "_seed" in name else name
        tf = f"{rd}/test.json"
        groups.setdefault(mode, []).append((name, json.load(open(vf)), json.load(open(tf)) if os.path.exists(tf) else None))
    if not groups:
        print("no validation.json found under", c.outdir); return
    tested = lambda t: bool(t and t.get("tested"))
    lines = ["## Per loss", "",
             "| loss | seeds trained | mean validation success rate | qualified (validation = 1.0) | "
             "tested with all test seeds = 1.0 | mean test success rate | starts settled on test |",
             "|---|---|---|---|---|---|---|"]
    detail = ["", "## Per run", "",
              "| run | epochs | lr | hidden | checkpoint | validation success rate | qualified | "
              "test success rates (seed 1..n) | mean |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mode, lst in groups.items():
        n = len(lst)
        q = [x for x in lst if x[1]["qualified"]]
        qt = [t for _, _, t in q if tested(t)]
        mean_t = f"{np.mean([t['mean_success_rate'] for t in qt]):.3f}" if qt else "-"
        settled = sum(t["starts_settled"] for t in qt); total = sum(t["starts_total"] for t in qt)
        lines.append(f"| {mode} | {n} | {np.mean([v['success_rate'] for _, v, _ in lst]):.2f} | {len(q)}/{n} | "
                     f"{sum(t['all_test_seeds_at_1'] for t in qt)}/{len(qt)} | {mean_t} | {settled}/{total} |")
        for nm, v, t in lst:
            m = v.get("meta") or {}
            ck = (f"best val (ep {m['checkpoint_epoch']})" if m.get("checkpoint") == "best_val"
                  else m.get("checkpoint", "-"))
            rates = " ".join(f"{r:.2f}" for r in t["success_rates"]) if tested(t) else "not tested"
            mean = f"{t['mean_success_rate']:.3f}" if tested(t) else "-"
            detail.append(f"| {nm} | {m.get('epochs', '-')} | {m.get('lr', '-')} | {m.get('hidden', '-')} | {ck} | "
                          f"{v['success_rate']:.2f} | {'yes' if v['qualified'] else 'no'} | {rates} | {mean} |")
    txt = "\n".join(lines + detail)
    print(txt)
    open(f"{c.outdir}/summary.md", "w").write(txt + "\n")
