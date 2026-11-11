"""Initial conditions: uncontrolled-attractor pools and low-mode fields, with separate pools and generators
for training, validation (= screening), calibration and retest sets."""
from __future__ import annotations
import hashlib, json, math, os
import torch

from .config import Cfg
from .plant import KS


def lowmode_ics(M, ks: KS, amp, gen):
    x = torch.as_tensor(ks.x)
    u = torch.zeros(M, ks.c.N)
    for m in range(1, 5):
        a = amp * torch.randn(M, 1, generator=gen)
        ph = 2 * math.pi * torch.rand(M, 1, generator=gen)
        u = u + a * torch.cos(2 * math.pi * m * x[None] / ks.c.L + ph)
    return torch.fft.rfft(u, dim=-1)


@torch.no_grad()
def attractor_pool(ks: KS, c: Cfg, n, seed):
    """States on the UNCONTROLLED attractor (what the controller is switched on into)."""
    gen = torch.Generator().manual_seed(seed)
    per = 8
    v = lowmode_ics(math.ceil(n / per), ks, c.ic_amp, gen)
    F = torch.zeros_like(v)
    for _ in range(int(c.pool_burn / c.dt)):
        v = ks.step(v, F)
    out, gap = [], int(c.pool_every / c.dt)
    for _ in range(per):
        for _ in range(gap):
            v = ks.step(v, F)
        out.append(v.clone())
    pool = torch.cat(out, 0)[:n]
    if not torch.isfinite(pool.real).all():
        raise RuntimeError("attractor pool contains non-finite values; reduce dt")
    return pool


def get_pool(ks, c: Cfg, n, seed):
    """Pool cache keyed by every parameter that affects it."""
    key = json.dumps([c.L, c.N, c.dt, c.ic_amp, c.pool_burn, c.pool_every, n, seed])
    path = f"{c.outdir}/pool_{hashlib.md5(key.encode()).hexdigest()[:10]}.pt"
    os.makedirs(c.outdir, exist_ok=True)
    if os.path.exists(path):
        return torch.load(path)
    pool = attractor_pool(ks, c, n, seed)
    torch.save(pool, path)
    return pool


def sample_ics(M, ks, c: Cfg, gen, pool, kind="mix"):
    """mix: first M//2 starts are states of the uncontrolled attractor, the rest random low-mode fields."""
    take = lambda m: pool[torch.randint(len(pool), (m,), generator=gen)]
    if kind == "ood":
        return lowmode_ics(M, ks, c.ic_amp_ood, gen)
    if kind == "lowmode":
        return lowmode_ics(M, ks, c.ic_amp, gen)
    if kind == "attractor":
        return take(M)
    m1 = M // 2
    return torch.cat([take(m1), lowmode_ics(M - m1, ks, c.ic_amp, gen)], 0)


TRAIN_POOL, VAL_POOL, TEST_POOL = 1000, 7000, 9000


def training_ics(ks, c: Cfg, epoch=None):
    """The n_initial training starts of seed c.seed.  epoch=None: the fixed set (generator seed + 1001)."""
    pool = get_pool(ks, c, c.pool_size, TRAIN_POOL + c.seed)
    s = c.seed + 1001 if epoch is None else 100000 + 10000 * c.seed + epoch
    return sample_ics(c.n_initial, ks, c, torch.Generator().manual_seed(s), pool)


def val_seed_of(c: Cfg):
    return c.val_seed_offset + c.seed


def validation_ics(ks, c: Cfg):
    """Validation set of training seed c.seed: eval_initial starts, generator val_seed_offset + seed."""
    pool = get_pool(ks, c, c.pool_size, VAL_POOL)
    return sample_ics(c.eval_initial, ks, c, torch.Generator().manual_seed(val_seed_of(c)), pool)


def calibration_ics(ks, c: Cfg):
    """Fixed set for calibrate / scan (independent of training, validation and test sets)."""
    pool = get_pool(ks, c, c.pool_size, VAL_POOL)
    return sample_ics(c.eval_initial, ks, c, torch.Generator().manual_seed(60000), pool)


def test_ics(ks, c: Cfg, test_seed, kind="mix"):
    """Test seed k: eval_initial starts, identical for every controller."""
    pool = get_pool(ks, c, c.pool_size, TEST_POOL)
    return sample_ics(c.eval_initial, ks, c, torch.Generator().manual_seed(50000 + test_seed), pool, kind)


def test_seeds(c: Cfg):
    return list(range(1, c.n_test_seeds + 1))
