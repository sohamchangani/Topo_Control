"""Feedback policy (MLP on point sensors) and open-loop reference policies."""
from __future__ import annotations
import hashlib
import torch
import torch.nn as nn

from .config import Cfg


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, n_act, hidden, c_max):
        super().__init__()
        self.c_max = c_max
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_act))
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)      # start near u = 0 (uncontrolled chaos)

    def forward(self, o):
        return self.c_max * torch.tanh(self.net(o))


class ConstPolicy(nn.Module):
    """Fixed open-loop forcing c = amp * pattern."""
    def __init__(self, amp, pattern):
        super().__init__()
        self.register_buffer("c", amp * torch.tensor(pattern, dtype=torch.get_default_dtype()))

    def forward(self, o):
        return self.c.expand(o.shape[0], -1)


class ZeroPolicy(nn.Module):
    def __init__(self, n_act):
        super().__init__(); self.n_act = n_act

    def forward(self, o):
        return o.new_zeros(o.shape[0], self.n_act)


def weights_sha(policy) -> str:
    """Fingerprint of the weights, stored with validation and test results to show they are identical."""
    h = hashlib.sha256()
    for k, t in sorted(policy.state_dict().items()):
        h.update(k.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def build_policy(c: Cfg, ks, kind="mlp", load=None):
    if kind == "zero":
        return ZeroPolicy(c.n_act)
    if kind == "const":
        return ConstPolicy(c.amp, c.pattern)
    if load:
        sd = torch.load(load, map_location="cpu")
        pol = MLPPolicy(c.n_sens, c.n_act, int(sd["net.0.weight"].shape[0]), c.c_max)   # width from the file
        pol.load_state_dict(sd)
        return pol
    return MLPPolicy(c.n_sens, c.n_act, c.hidden, c.c_max)
