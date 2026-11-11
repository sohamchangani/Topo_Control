"""Actuated Kuramoto-Sivashinsky equation: batched differentiable ETDRK4 solver and closed-loop rollout."""
from __future__ import annotations
import numpy as np
import torch

from .config import Cfg


class KS:
    """Batched differentiable ETDRK4 (Kassam-Trefethen) solver. State v: (M, N//2+1) complex."""

    def __init__(self, c: Cfg):
        self.c, N, L, dt = c, c.N, c.L, c.dt
        self.Nh = N // 2 + 1
        k = 2 * np.pi / L * np.arange(self.Nh)
        lin = k**2 - k**4
        r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
        LR = dt * lin[:, None] + r[None]
        Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
        f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, 1))
        f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, 1))
        f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, 1))
        g = -0.5j * k * (np.arange(self.Nh) < N / 3)                 # 2/3-rule dealiasing
        T = lambda a: torch.as_tensor(a)
        self.E, self.E2 = T(np.exp(dt * lin)), T(np.exp(dt * lin / 2))
        self.Q, self.f1, self.f2, self.f3, self.g = T(Q), T(f1), T(f2), T(f3), T(g)
        # actuators: periodic Gaussians, spatial mean removed (never drive the k=0 mode)
        self.x = np.arange(N) * L / N
        xa = (np.arange(c.n_act) + 0.5) * L / c.n_act
        d = (self.x[None] - xa[:, None] + L / 2) % L - L / 2
        B = np.exp(-d**2 / (2 * c.act_sigma**2))
        B = B - B.mean(1, keepdims=True)
        self.B = T(B)
        self.Bh = torch.fft.rfft(self.B, dim=-1)
        self.x_act = xa
        self.sens_idx = torch.arange(c.n_sens) * (N // c.n_sens)

    def nonlin(self, v, F):
        return self.g * torch.fft.rfft(torch.fft.irfft(v, n=self.c.N, dim=-1) ** 2, dim=-1) + F

    def step(self, v, F):
        Nv = self.nonlin(v, F)
        a = self.E2 * v + self.Q * Nv;             Na = self.nonlin(a, F)
        b = self.E2 * v + self.Q * Na;             Nb = self.nonlin(b, F)
        cc = self.E2 * a + self.Q * (2 * Nb - Nv); Nc = self.nonlin(cc, F)
        return self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3

    def forcing(self, a):
        return a.to(self.Bh.dtype) @ self.Bh

    def field(self, v):
        return torch.fft.irfft(v, n=self.c.N, dim=-1)

    def obs(self, v):
        return self.field(v)[:, self.sens_idx]

    def coords(self, v):
        """Normalised Fourier coordinates 2 v_k / N, k = 1..K, as (Re, Im): u = sum Re(z_k e^{ikx})."""
        z = 2.0 * v[..., 1:self.c.K_modes + 1] / self.c.N
        return torch.cat([z.real, z.imag], dim=-1)


def rollout(policy, ks: KS, v, n_ctrl, c: Cfg, record=True, tbptt=0):
    """Closed loop with zero-order hold for n_ctrl timesteps. Returns v, Z (n,M,2K), A (n,M,n_act)."""
    Z, A = [], []
    for i in range(n_ctrl):
        if tbptt and i > 0 and i % tbptt == 0:
            v = v.detach()
        a = policy(ks.obs(v))
        F = ks.forcing(a)
        for _ in range(c.ctrl_every):
            v = ks.step(v, F)
        if record:
            Z.append(ks.coords(v)); A.append(a)
    st = lambda xs: torch.stack(xs, 0) if xs else None
    return v, st(Z), st(A)
