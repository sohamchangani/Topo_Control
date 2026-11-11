"""Diagnostic plots."""
from __future__ import annotations
import numpy as np


def plot_phase(Zw, res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = Zw.numpy(); P = Z.reshape(-1, Z.shape[-1]); mu = P.mean(0)
    _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for j in range(Z.shape[1]):
        Y = (Z[:, j] - mu) @ Vt[:2].T
        ax.plot(Y[:, 0], Y[:, 1], ".", ms=2, alpha=0.6, c="C0" if res["settled"][j] else "C3")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"scored window; blue = settled, red = not ({res['settled_fraction']:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_history(history, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ["loss", "R", "r_conn", "l1", "effort", "L_energy", "L_rec", "grad_norm"]
            if any(k in h for h in history)]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for ax, k in zip(axs.ravel(), keys):
        ax.plot([h["epoch"] for h in history], [h.get(k, np.nan) for h in history]); ax.set_title(k); ax.set_xlabel("epoch")
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
