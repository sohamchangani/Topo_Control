"""Training losses: energy, range, soft Poincare section, and the loop-only plateau (topology)."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

try:
    import oineus.diff as odiff
    OINEUS_AVAILABLE = True
except ImportError:
    odiff = None
    OINEUS_AVAILABLE = False

from .plant import Acrobot


def energy_loss(
    plant: Acrobot,
    traj: torch.Tensor,
    burn_in: int,
    energy_target: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    late = traj[burn_in:]
    E = plant.energy(late)
    loss = (E - energy_target).square().mean()
    return loss, E


def range_loss(
    traj: torch.Tensor,
    burn_in: int,
    q1_amp_target: float,
    q2_amp_target: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Exact symmetric coordinate-range baseline.

    For every trajectory:
        max q1 -> +a1, min q1 -> -a1
        max q2 -> +a2, min q2 -> -a2

    This is intentionally a strong problem-specific geometric baseline.
    """
    late = traj[burn_in:]
    q1 = late[..., 0]
    q2 = late[..., 1]

    q1_max = q1.max(dim=0).values
    q1_min = q1.min(dim=0).values
    q2_max = q2.max(dim=0).values
    q2_min = q2.min(dim=0).values

    loss = (
        (q1_max - q1_amp_target).square()
        + (q1_min + q1_amp_target).square()
        + (q2_max - q2_amp_target).square()
        + (q2_min + q2_amp_target).square()
    ).mean()

    return loss, q1_max, q1_min, q2_max, q2_min


def soft_poincare_loss(
    traj: torch.Tensor,
    burn_in: int,
    velocity_scale: float,
    section_sigma: float,
    direction_sigma: float,
    min_section_mass: float,
    lambda_mass: float,
    min_crossing_speed: float,
    lambda_speed: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable surrogate for the Poincare-section collapse loss.

    Section:
        q1 = 0 mod 2pi, dq1 > 0

    Smooth section weight:
        near q1=0: exp(-(1-cos q1)/sigma^2)
        upward:     sigmoid(dq1 / direction_sigma)

    Crossing feature:
        z = (cos q2, sin q2, dq1 / velocity_scale, dq2 / velocity_scale)

    A q1=0 section of a 4D autonomous system is 3D; dq1 must be retained.
    The pooled weighted variance asks ALL initial conditions to have section
    points concentrated around one common point.

    A mass penalty prevents the trivial solution "never visit the section".
    A minimum positive crossing-speed penalty prevents the other trivial
    solution: sitting at an equilibrium on the section.
    """
    late = traj[burn_in:]

    q1 = late[..., 0]
    q2 = late[..., 1]
    dq1 = late[..., 2]
    dq2 = late[..., 3]

    # 1-cos(q1) is zero only at q1=0 mod 2pi, unlike sin(q1) which also
    # vanishes at pi.
    angular_distance = 1.0 - torch.cos(q1)

    w_section = torch.exp(
        -angular_distance / (section_sigma ** 2)
    )
    w_direction = torch.sigmoid(
        dq1 / direction_sigma
    )
    w = w_section * w_direction

    z = torch.stack(
        [
            torch.cos(q2),
            torch.sin(q2),
            dq1 / velocity_scale,
            dq2 / velocity_scale,
        ],
        dim=-1,
    )

    denom = w.sum() + 1e-10
    mean_z = (w.unsqueeze(-1) * z).sum(dim=(0, 1)) / denom

    sq_dist = (z - mean_z).square().sum(dim=-1)
    spread = (w * sq_dist).sum() / denom

    # Require every rollout to actually visit the section.
    per_traj_mass = w.mean(dim=0)
    mass_loss = F.relu(
        min_section_mass - per_traj_mass
    ).square().mean()

    # Require a genuine positive passage through the section.  This prevents
    # the controller from minimizing Poincare spread by simply parking at
    # q1=0 with zero velocity.
    per_traj_weight = w.sum(dim=0) + 1e-10
    per_traj_crossing_speed = (w * dq1).sum(dim=0) / per_traj_weight
    speed_loss = F.relu(
        min_crossing_speed - per_traj_crossing_speed
    ).square().mean()

    total = (
        spread
        + lambda_mass * mass_loss
        + lambda_speed * speed_loss
    )
    return total, spread, per_traj_mass, per_traj_crossing_speed


def make_topology_cloud(
    traj: torch.Tensor,
    burn_in: int,
    ph_points: int,
    velocity_scale: float,
) -> torch.Tensor:
    """
    Build a balanced pooled cloud from all trajectories.

    We embed angular variables periodically:
        (cos q1, sin q1, cos q2, sin q2, dq1/vs, dq2/vs)

    Each trajectory contributes approximately the same number of points.
    """
    late = traj[burn_in:]  # [L,N,4]
    L, N, _ = late.shape

    per_traj = max(1, ph_points // N)
    idx = torch.linspace(
        0,
        L - 1,
        steps=per_traj,
        device=late.device,
    ).round().long()

    samples = late[idx]  # [P,N,4]
    # Flatten time-major so truncation stays approximately balanced across
    # trajectories even when ph_points is not divisible by N.
    samples = samples.reshape(-1, 4)

    if samples.shape[0] > ph_points:
        samples = samples[:ph_points]

    q1, q2, dq1, dq2 = samples.unbind(dim=-1)

    cloud = torch.stack(
        [
            torch.cos(q1),
            torch.sin(q1),
            torch.cos(q2),
            torch.sin(q2),
            dq1 / velocity_scale,
            dq2 / velocity_scale,
        ],
        dim=-1,
    )

    # Break exact distance ties without materially changing geometry.
    return cloud + 1e-7 * torch.randn_like(cloud)


def differentiable_h1_diagram(
    points: torch.Tensor,
    max_diameter: float,
    n_threads: int,
) -> torch.Tensor:
    """
    Differentiable H1 persistence from a pairwise-distance matrix.

    Oineus' coordinate-based VR backend is compiled only for a limited
    set of ambient dimensions.  Our natural Acrobot embedding is already
    6D, and later forcing-phase augmentation would make it 8D.  Passing
    the differentiable pairwise-distance matrix instead removes that
    ambient-dimension restriction while preserving gradients back to the
    original state-space points.
    """
    if not OINEUS_AVAILABLE:
        raise RuntimeError(
            "Topology mode requires Oineus. Install with: pip install oineus"
        )

    # Keep the Euclidean metric differentiable with respect to `points`.
    distances = torch.cdist(points, points, p=2)

    if max_diameter <= 0:
        if points.shape[0] > 1:
            effective = (
                1.01
                * distances.detach().max().item()
            )
        else:
            effective = 1.0
    else:
        effective = max_diameter

    fil = odiff.vr_filtration(
        distances,
        from_pwdists=True,
        max_dim=2,
        max_diameter=effective,
        n_threads=n_threads,
    )

    dgms = odiff.persistence_diagram(
        fil,
        include_inf_points=False,
        n_threads=n_threads,
    )

    if 1 not in dgms or dgms[1].numel() == 0:
        return points.new_empty((0, 2))

    return dgms[1]


def one_loop_plateau_loss(
    dgm1: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Maximize the longest connected filtration interval on which beta_1(r) = 1.

    If [r0, r1) is the longest connected interval satisfying beta_1(r)=1,
    define

        R = r1 - r0,
        L_topo = -R.

    The interval is computed from the H1 barcode endpoints. Births contribute
    +1 and deaths contribute -1 to beta_1. Events occurring at exactly the
    same filtration value are grouped before the Betti sweep. This matters in
    Vietoris-Rips filtrations because several simplices can share one
    filtration value; a simultaneous birth/death with zero net Betti change
    should not artificially split one beta_1=1 plateau into two pieces.

    The endpoint ordering, grouping, and choice of the longest run are
    discrete. Away from combinatorial changes, however, the selected width is
    differentiable with respect to its two boundary endpoints, so gradients
    flow through Oineus back to the rollout cloud almost everywhere.

    Returns
    -------
    loss : Tensor
        -R, to be minimized.
    r0, r1, R : Tensor
        Start, end, and width of the longest connected beta_1=1 plateau.
    l1, l2 : Tensor
        Largest and second-largest H1 lifetimes, retained only as diagnostics.
    lifetimes : Tensor
        All H1 lifetimes.

    Note
    ----
    If there is no H1 bar, R=0 and the PH term has no loop-creation gradient.
    This is the same fundamental cold-start limitation as other barcode losses.
    """
    zero = reference.sum() * 0.0

    if dgm1.numel() == 0:
        empty = reference.new_empty((0,))
        return zero, zero, zero, zero, zero, zero, empty

    births = dgm1[:, 0]
    deaths = dgm1[:, 1]
    lifetimes = (deaths - births).clamp_min(0.0)

    lifetimes_sorted, _ = torch.sort(lifetimes, descending=True)
    l1 = lifetimes_sorted[0]
    if lifetimes_sorted.numel() >= 2:
        l2 = lifetimes_sorted[1]
    else:
        l2 = zero

    # Sort all barcode events by filtration value.
    endpoints = torch.cat([births, deaths], dim=0)
    events = torch.cat(
        [torch.ones_like(births), -torch.ones_like(deaths)],
        dim=0,
    )
    sorted_endpoints, order = torch.sort(endpoints)
    sorted_events = events[order]

    # Group events at identical filtration values. Grouping is a discrete
    # combinatorial operation; using the mean of tied endpoint tensors keeps a
    # sensible gradient through a shared boundary value.
    endpoint_values = sorted_endpoints.detach().cpu().tolist()
    groups = []
    g0 = 0
    for k in range(1, len(endpoint_values) + 1):
        if k == len(endpoint_values) or endpoint_values[k] != endpoint_values[g0]:
            groups.append((g0, k))
            g0 = k

    unique_endpoints = torch.stack(
        [sorted_endpoints[a:b].mean() for a, b in groups]
    )
    net_events = torch.stack(
        [sorted_events[a:b].sum() for a, b in groups]
    )

    if unique_endpoints.numel() < 2:
        return zero, zero, zero, zero, l1, l2, lifetimes

    # beta_after[k] is beta_1 on [u_k, u_{k+1}) under the standard half-open
    # barcode convention, after applying all events at u_k.
    beta_after = torch.cumsum(net_events, dim=0)
    one_loop_mask = (beta_after[:-1] == 1).detach().cpu().tolist()

    # Find maximal consecutive runs of intervals on which beta_1=1.  This is
    # more precise than taking the widest single gap between endpoints because
    # a zero-net event at one filtration value leaves beta_1 unchanged and the
    # plateau should continue through it.
    runs = []
    k = 0
    while k < len(one_loop_mask):
        if not one_loop_mask[k]:
            k += 1
            continue
        a = k
        while k + 1 < len(one_loop_mask) and one_loop_mask[k + 1]:
            k += 1
        b = k
        runs.append((a, b))
        k += 1

    if not runs:
        return zero, zero, zero, zero, l1, l2, lifetimes

    run_widths = torch.stack(
        [unique_endpoints[b + 1] - unique_endpoints[a] for a, b in runs]
    )
    R, run_idx = torch.max(run_widths, dim=0)
    chosen = int(run_idx.detach().item())
    a, b = runs[chosen]
    r0 = unique_endpoints[a]
    r1 = unique_endpoints[b + 1]

    loss = -R
    return loss, r0, r1, R, l1, l2, lifetimes
