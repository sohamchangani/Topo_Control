"""Independent dynamical validator (no persistent homology decides success): periodicity from the return
error, common orbit from the Hausdorff distance to the medoid orbit.  Also the Lyapunov diagnostic and the
PH certificate, which are reported but never used for success."""
from __future__ import annotations
import math
import torch

from .config import Cfg
from .plant import rollout
from .losses import energy_of, fps_indices, ph_diagrams, plateau_loss


@torch.no_grad()
def return_stats(Zw, c: Cfg, rms_floor):
    """Per-start periodicity from the relative return error r(l) = |z(t+l)-z(t)|_rms / rms(z - mean).
    Periodic iff the start oscillates (rms >= floor), decorrelates (r reaches decor_thr) and later returns
    to within tol_rec. Returns (periodic bool (M,), rho (M,), period (M,), rms (M,))."""
    n, M, _ = Zw.shape
    zc = Zw - Zw.mean(0)
    rms = torch.sqrt((zc ** 2).sum(-1).mean(0))
    lmax = min(int(round(c.lag_max / c.dt_ctrl)), n // 2)
    r = torch.stack([torch.sqrt(((Zw[l:] - Zw[:-l]) ** 2).sum(-1).mean(0)) for l in range(1, lmax + 1)]) \
        / rms.clamp_min(1e-12)                                             # (lmax, M)
    rho = torch.full((M,), float("inf")); per = torch.zeros(M)
    for j in range(M):
        idx = torch.nonzero(r[:, j] >= c.decor_thr)
        if len(idx) == 0:
            continue
        l0 = int(idx[0]); k = int(torch.argmin(r[l0:, j]))
        rho[j] = r[l0 + k, j]; per[j] = (l0 + k + 1) * c.dt_ctrl
    periodic = (rms >= rms_floor) & (rho <= c.tol_rec)
    return periodic, rho, per, rms


@torch.no_grad()
def hausdorff_matrix(Zw):
    M = Zw.shape[1]
    H = torch.zeros(M, M)
    for i in range(M):
        for j in range(i + 1, M):
            D = torch.cdist(Zw[:, i], Zw[:, j])
            H[i, j] = H[j, i] = max(D.min(1).values.max(), D.min(0).values.max())
    return H


def rms_floor_of(c: Cfg):
    return c.rms_floor_frac * math.sqrt(c.v_ref) if c.v_ref > 0 else c.rms_floor_abs


@torch.no_grad()
def analyze_window(Zw, c: Cfg):
    """Zw: (n, M, D) scored window. Settled start = periodic AND within delta of the medoid orbit.
    n_orbits = number of distinct periodic orbits (leader clustering at tolerance delta)."""
    periodic, rho, per, rms = return_stats(Zw, c, rms_floor_of(c))
    M = Zw.shape[1]
    out = dict(frac_periodic=float(periodic.float().mean()), settled_fraction=0.0, n_orbits=0,
               orbit_sizes=[], period=float("nan"), rho_median=float(rho.median()),
               rms_median=float(rms.median()), settled=[False] * M, medoid=-1, delta=float("nan"))
    if not periodic.any():
        return out
    H = hausdorff_matrix(Zw)
    pidx = torch.nonzero(periodic).flatten()
    sub = H[pidx][:, pidx]
    med = int(pidx[int(torch.argmin(sub.median(1).values))])
    diam = float(torch.cdist(Zw[:, med], Zw[:, med]).max())
    delta = c.delta_rel * diam
    settled = periodic & (H[med] <= delta)
    leaders, sizes = [], []
    for j in pidx.tolist():
        for k, l in enumerate(leaders):
            if H[j, l] <= delta:
                sizes[k] += 1; break
        else:
            leaders.append(j); sizes.append(1)
    out.update(settled_fraction=float(settled.float().mean()), n_orbits=len(leaders),
               orbit_sizes=sorted(sizes, reverse=True), period=float(per[med]), medoid=med, delta=delta,
               settled=settled.tolist())
    return out


@torch.no_grad()
def lyapunov(policy, ks, v, c: Cfg, seed=0, delta=1e-7):
    """Finite-time largest Lyapunov exponent of the closed loop (twin trajectory, renormalised every
    timestep). The first lyap_discard time units are dropped (the random initial direction relaxes onto the
    leading direction; keeping that transient biases the estimate by O(1/T), about +0.03 at T = 100)."""
    gen = torch.Generator().manual_seed(seed)
    du = torch.randn(v.shape[0], c.N, generator=gen)
    du = delta * du / torch.linalg.norm(du, dim=1, keepdim=True)
    w = v + torch.fft.rfft(du, dim=-1)
    n, skip = c.n_ctrl(c.T_lyap), c.n_ctrl(c.lyap_discard)
    acc = torch.zeros(v.shape[0])
    for i in range(n):
        Fv = ks.forcing(policy(ks.obs(v))); Fw = ks.forcing(policy(ks.obs(w)))
        for _ in range(c.ctrl_every):
            v = ks.step(v, Fv); w = ks.step(w, Fw)
        d = torch.linalg.norm(ks.field(w) - ks.field(v), dim=1).clamp_min(1e-300)
        if i >= skip:
            acc += torch.log(d / delta)
        w = v + (w - v) * (delta / d)[:, None]
    return (acc / ((n - skip) * c.dt_ctrl)).tolist()


@torch.no_grad()
def certificate(Zw, c: Cfg, n_threads, seed=0):
    """Secondary, PH-based: R of the pooled window cloud (N_cert FPS points) and the covering radius
    eps_hat = max over all window points of the distance to the FPS subsample (the Hausdorff error of the
    subsample w.r.t. the dense cloud). Certified iff R > cert_factor * eps_hat."""
    gen = torch.Generator().manual_seed(seed)
    P = Zw.reshape(-1, Zw.shape[-1])
    idx = fps_indices(P, c.N_cert, gen)
    S = P[idx]
    eps_hat = float(torch.cdist(P, S).min(1).values.max())
    dgm0, dgm1 = ph_diagrams(S, n_threads)
    _, _, info = plateau_loss(dgm0, dgm1, S)
    return dict(R=info["R"], r0=info["r0"], r1=info["r1"], r_conn=info["r_conn"], l1=info["l1"], l2=info["l2"],
                eps_hat=eps_hat, certified=bool(info["R"] > c.cert_factor * eps_hat))


@torch.no_grad()
def evaluate_set(policy, ks, c: Cfg, v0, lyap=True, cert=False, n_threads=2):
    """Run one set of starts for eval_steps timesteps and score the last eval_steps - eval_burn_in.
    success_rate = fraction of the starts that settle on the common periodic orbit."""
    v_end, Z, A = rollout(policy, ks, v0, c.eval_steps, c)
    Zw, Aw = Z[c.eval_burn_in:], A[c.eval_burn_in:]
    fin = analyze_window(Zw, c)
    fin["success_rate"] = fin["settled_fraction"]
    fin["n_starts"] = int(v0.shape[0])
    fin["n_settled"] = int(sum(fin["settled"]))
    fin["mean_effort"] = float((Aw ** 2).mean())
    fin["energy_mean"] = float(energy_of(Zw).mean())
    fin["finite"] = bool(torch.isfinite(Z).all())
    if lyap:
        lam = lyapunov(policy, ks, v_end[:c.n_lyap], c)
        fin["lyapunov"] = lam; fin["lyapunov_max"] = max(lam)
        fin["lyap_ok"] = bool(max(lam) <= c.lam_tol)
    if cert:
        fin["certificate"] = certificate(Zw, c, n_threads)
    fin["good"] = bool(fin["success_rate"] == 1.0 and (fin.get("lyap_ok", True) if c.require_lyap else True))
    return fin, Zw


def short(name, r):
    return (f"{name:26s} success={r['success_rate']:.2f} ({r['n_settled']}/{r['n_starts']}) "
            f"periodic={r['frac_periodic']:.2f} orbits={r['n_orbits']} sizes={r['orbit_sizes']} "
            f"period={r['period']:.1f} lam={r.get('lyapunov_max', float('nan')):+.3f} "
            f"effort={r['mean_effort']:.3f}")
