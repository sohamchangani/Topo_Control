"""Reference values for the baselines (E*, v_ref, P*), measured on the open-loop orbit, and the open-loop scan."""
from __future__ import annotations
import copy, json, os

from .config import Cfg
from .plant import KS
from .policies import ConstPolicy, ZeroPolicy
from .initial_conditions import calibration_ics
from .losses import energy_of
from .validator import evaluate_set, short


def ref_path(c): return f"{c.outdir}/reference.json"


def proto_of(c: Cfg):
    """Every setting the reference values depend on; reference.json is recomputed if any changes."""
    return dict(L=c.L, N=c.N, dt=c.dt, ctrl_every=c.ctrl_every, n_act=c.n_act, act_sigma=c.act_sigma,
                n_sens=c.n_sens, K_modes=c.K_modes, ic_amp=c.ic_amp, pool_burn=c.pool_burn,
                pool_every=c.pool_every, pool_size=c.pool_size, eval_initial=c.eval_initial,
                eval_steps=c.eval_steps, eval_burn_in=c.eval_burn_in,
                lag_max=c.lag_max, decor_thr=c.decor_thr, tol_rec=c.tol_rec, delta_rel=c.delta_rel,
                pattern=list(c.pattern))


def load_reference(c: Cfg, ks):
    d = json.load(open(ref_path(c))) if os.path.exists(ref_path(c)) else None
    if d is None or d.get("proto") != proto_of(c):
        print("[reference values missing or computed with other settings: running calibrate]", flush=True)
        calibrate(copy.deepcopy(c))
        d = json.load(open(ref_path(c)))
    c.E_star, c.v_ref, c.P_ref = d["E_star"], d["v_ref"], d["P_ref"]


def calibrate(c: Cfg):
    """Run the validator on known regimes (on the fixed calibration set) and derive the reference values handed to
    the BASELINES: E_star (mean energy), v_ref (temporal variance) and P_ref (period) of the open-loop orbit
    at amp 1.0.  v_ref also sets the oscillation floor of the success rate.
    Expected: u = 0 -> chaos, not settled; fixed forcing -> periodic orbits, but TWO coexisting mirror
    copies (the forcing pattern is symmetric), so success rate ~ 0.5."""
    c.E_star = c.v_ref = c.P_ref = 0.0
    ks = KS(c)
    v0 = calibration_ics(ks, c)
    ref = None
    for name, pol in [("u = 0 (chaos)", ZeroPolicy(c.n_act)),
                      ("fixed forcing A=0.65", ConstPolicy(0.65, c.pattern)),
                      ("fixed forcing A=1.00", ConstPolicy(1.0, c.pattern))]:
        fin, Zw = evaluate_set(pol, ks, c, v0, n_threads=c.threads or 2)
        print(short(name, fin), flush=True)
        if name.endswith("1.00"):
            ref = (fin, Zw)
    fin, Zw = ref
    starts = [j for j in range(Zw.shape[1]) if fin["settled"][j]]
    if not starts:
        raise RuntimeError("calibrate: no start settled under fixed forcing A=1.0, so the reference orbit "
                           "is undefined; check eval_steps / eval_burn_in / lag_max")
    sub = Zw[:, starts]
    E = float(energy_of(sub).mean())
    v = float(((sub - sub.mean(0)) ** 2).sum(-1).mean())
    d = dict(E_star=E, v_ref=v, P_ref=fin["period"], n_starts=len(starts),
             source="fixed forcing amp 1.0, medoid orbit, calibration set", proto=proto_of(c))
    os.makedirs(c.outdir, exist_ok=True)
    json.dump(d, open(ref_path(c), "w"), indent=1)
    print("reference values for baselines:", {k: v_ for k, v_ in d.items() if k != "proto"})


def scan(c: Cfg):
    """Open-loop baseline: success rate / distinct orbits vs amplitude, same validator, calibration set."""
    ks = KS(c); load_reference(c, ks)
    v0 = calibration_ics(ks, c)
    rows = []
    for amp in c.amps:
        fin, _ = evaluate_set(ConstPolicy(amp, c.pattern), ks, c, v0, n_threads=c.threads or 2)
        rows.append(dict(amp=amp, **{k: v for k, v in fin.items() if k not in ("settled", "lyapunov")}))
        print(short(f"fixed forcing A={amp:g}", fin), flush=True)
    os.makedirs(c.outdir, exist_ok=True)
    json.dump(rows, open(f"{c.outdir}/scan.json", "w"), indent=1)
