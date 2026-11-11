"""Training losses: connected one-loop plateau (topology), energy and recurrence, and the full training loss."""
from __future__ import annotations
import torch

try:
    import oineus.diff as odiff
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install oineus   (differentiable Vietoris-Rips backend)") from e

from .config import Cfg
from .plant import rollout


@torch.no_grad()
def fps_indices(P, n, gen):
    """Farthest-point sampling (indices only)."""
    K = P.shape[0]
    if K <= n:
        return torch.arange(K)
    idx = [int(torch.randint(K, (1,), generator=gen))]
    d = torch.linalg.norm(P - P[idx[0]], dim=1)
    for _ in range(n - 1):
        j = int(torch.argmax(d)); idx.append(j)
        d = torch.minimum(d, torch.linalg.norm(P - P[j], dim=1))
    return torch.tensor(idx)


def make_cloud(Z, n, gen):
    """(..., D) -> differentiable (n, D) cloud: FPS gather + tiny jitter (general position)."""
    P = Z.reshape(-1, Z.shape[-1])
    P = P[fps_indices(P.detach(), n, gen)]
    return P + 1e-7 * torch.randn(P.shape, generator=gen)


def ph_diagrams(points, n_threads=2):
    """Differentiable H0/H1 diagrams (finite points) of the Vietoris-Rips filtration of a cloud.
    The full complex is used (max_diameter = diameter of the cloud) so no bar is censored.
    Pairwise-distance input is required because Oineus' coordinate backend is compiled for 2-5 dims."""
    dist = torch.cdist(points, points)
    md = float(dist.detach().max()) * 1.0001 + 1e-9
    fil = odiff.vr_filtration(dist, from_pwdists=True, max_dim=2, max_diameter=md, n_threads=n_threads)
    dg = odiff.persistence_diagram(fil, include_inf_points=False, n_threads=n_threads)
    empty = points.new_empty((0, 2))
    return (dg[0] if 0 in dg and dg[0].numel() else empty), (dg[1] if 1 in dg and dg[1].numel() else empty)


def plateau_loss(dgm0, dgm1, ref):
    """Width R of the longest filtration interval on which (beta_0, beta_1) = (1, 1).

    beta_0(r) = 1 exactly for r >= r_conn := largest finite H0 death (every H0 bar is born at 0 in a
    Rips filtration). beta_1(r) is swept from the H1 endpoints (events at identical values are grouped).
    For each maximal beta_1 = 1 run [s, e) the usable part is [max(s, r_conn), e), so

        R = max over runs of relu(e - max(s, r_conn)),     loss = -R   (raw, no normalisation).

    Because r_conn enters, a loop plus an isolated second cluster (a second attractor) has no usable
    plateau unless the loop survives until the clusters join.

    Training uses the SIGNED width  R_signed = max over runs of (e - max(s, r_conn))  (negative when the
    cloud is still disconnected where the loop lives). That keeps a gradient that pulls a stray start / second
    attractor in (it shrinks r_conn); with relu the loss would be flat exactly when one start is unconverged.
    The reported R is relu(R_signed). Returns (R_signed, R_signed, info) with info = detached floats.
    """
    zero = ref.sum() * 0.0
    info = dict(R=0.0, R_signed=0.0, r0=0.0, r1=0.0, r_conn=0.0, l1=0.0, l2=0.0, n_h1=0)
    r_conn = dgm0[:, 1].max() if dgm0.numel() else zero
    info["r_conn"] = float(r_conn.detach())
    if dgm1.numel() == 0:
        return zero, zero, info
    births, deaths = dgm1[:, 0], dgm1[:, 1]
    life = (deaths - births).clamp_min(0.0)
    ls, _ = torch.sort(life, descending=True)
    l1 = ls[0]
    info.update(l1=float(l1.detach()), l2=float(ls[1].detach()) if ls.numel() > 1 else 0.0, n_h1=int(life.numel()))

    ends = torch.cat([births, deaths])
    ev = torch.cat([torch.ones_like(births), -torch.ones_like(deaths)])
    se, order = torch.sort(ends)
    sv = ev[order]
    vals = se.detach().cpu().tolist()
    groups, g0 = [], 0
    for k in range(1, len(vals) + 1):
        if k == len(vals) or vals[k] != vals[g0]:
            groups.append((g0, k)); g0 = k
    u = torch.stack([se[a:b].mean() for a, b in groups])
    net = torch.stack([sv[a:b].sum() for a, b in groups])
    beta = torch.cumsum(net, 0)
    mask = (beta[:-1] == 1).detach().cpu().tolist()
    runs, k = [], 0
    while k < len(mask):
        if not mask[k]:
            k += 1; continue
        a = k
        while k + 1 < len(mask) and mask[k + 1]:
            k += 1
        runs.append((a, k)); k += 1
    if not runs:
        return zero, zero, info
    widths, starts = [], []
    for a, b in runs:
        s_ = torch.maximum(u[a], r_conn)
        widths.append(u[b + 1] - s_); starts.append(s_)
    W = torch.stack(widths)
    R, j = torch.max(W, 0)                    # signed width
    j = int(j)
    info.update(R_signed=float(R.detach()))
    if float(R.detach()) > 0:
        info.update(R=float(R.detach()), r0=float(starts[j].detach()), r1=float(u[runs[j][1] + 1].detach()))
    return R, R, info          # (R, R, info): caller applies sign/cap


def energy_of(Z):
    """0.5 |z|^2 = <u^2>_x restricted to the K retained modes (differentiable, from the PH coordinates)."""
    return 0.5 * (Z ** 2).sum(-1)


def recurrence_lags(c: Cfg, n):
    """Lag range [0.6 P_ref, 1.4 P_ref] in timesteps, clipped to fit a window of n timesteps."""
    lo = max(2, int(round(0.6 * c.P_ref / c.dt_ctrl)))
    hi = min(int(round(1.4 * c.P_ref / c.dt_ctrl)), n - 4)
    return lo, hi


def check_recurrence_window(c: Cfg):
    n = c.steps - c.burn_in
    lo, hi = recurrence_lags(c, n)
    if c.P_ref <= 0 or lo > hi:
        raise ValueError(
            f"recurrence loss: the scored training window ({n} timesteps = {n * c.dt_ctrl:g} time units) "
            f"cannot hold lags 0.6..1.4 x P_ref = {c.P_ref:g}; increase --steps or decrease --burn_in")


def recurrence_terms(Z, c: Cfg):
    """Per-rollout recurrence (period-window) loss + amplitude floor. Z: (n, M, D).
    r2(l) = mean_t |z(t+l) - z(t)|^2 over lags l in [0.6 P_ref, 1.4 P_ref]; use the best lag.
    Normalised by v_ref. The floor relu(1 - var/(0.5 v_ref)) excludes the trivial steady state."""
    lo, hi = recurrence_lags(c, Z.shape[0])
    r2 = torch.stack([((Z[l:] - Z[:-l]) ** 2).sum(-1).mean(0) for l in range(lo, hi + 1, 2)])   # (nl, M)
    rmin = r2.min(0).values / c.v_ref
    var = ((Z - Z.mean(0)) ** 2).sum(-1).mean(0)
    floor = torch.relu(1.0 - var / (0.5 * c.v_ref))
    return rmin.mean(), floor.mean()


def compute_loss(policy, ks, v0, c: Cfg, gen, n_threads):
    """One training rollout of `steps` timesteps from v0: the first `burn_in` are discarded (run without
    gradient), the loss is computed on the remaining steps - burn_in timesteps of all starts together."""
    with torch.no_grad():
        v, _, _ = rollout(policy, ks, v0, c.burn_in, c, record=False)
    _, Z, A = rollout(policy, ks, v.detach(), c.steps - c.burn_in, c, tbptt=c.tbptt)
    parts = c.loss.split("_")
    effort = (A ** 2).mean()
    loss = c.beta_effort * effort
    st = dict(effort=float(effort.detach()))
    if "topology" in parts:
        cloud = make_cloud(Z, c.N_cloud, gen)
        dgm0, dgm1 = ph_diagrams(cloud, n_threads)
        R, _, info = plateau_loss(dgm0, dgm1, cloud)
        R_eff = torch.clamp(R, max=c.R_cap) if c.R_cap > 0 else R
        l_top = -R_eff
        if c.cold_kappa > 0 and info["R"] == 0.0 and dgm1.numel():
            l_top = l_top - c.cold_kappa * (dgm1[:, 1] - dgm1[:, 0]).max()
        loss = loss + c.w_top * l_top
        st.update({k: v_ for k, v_ in info.items()}); st["L_top"] = float(l_top.detach())
    if "energy" in parts:
        l_e = ((energy_of(Z).mean(0) - c.E_star) ** 2).mean()
        loss = loss + c.w_energy * l_e
        st["L_energy"] = float(l_e.detach()); st["E_mean"] = float(energy_of(Z).mean().detach())
    if "recurrence" in parts:
        r, fl = recurrence_terms(Z, c)
        l_r = r + fl
        loss = loss + c.w_rec * l_r
        st["L_rec"] = float(r.detach()); st["L_floor"] = float(fl.detach())
    st["loss"] = float(loss.detach())
    return loss, st
