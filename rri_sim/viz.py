"""Plotting helpers (matplotlib)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]


def hydrograph(times, sims: dict, obs=None, rain=None, title="", path=None, ylabel="Q [m³/s]"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if obs is not None:
        ax.plot(times, obs, "k.", ms=3, label="observed")
    for (label, q), c in zip(sims.items(), ["#1f6feb", "#d1495b", "#2a9d8f", "#e9c46a"]):
        ax.plot(times, q, color=c, lw=1.5, label=label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if rain is not None:
        ax2 = ax.twinx()
        ax2.bar(times, rain, width=1 / 24, color="#6c8ebf", alpha=0.5, align="edge")
        ax2.set_ylim(np.nanmax(rain) * 3, 0)
        ax2.set_ylabel("basin rain [mm/h]")
    ax.legend(loc="center right")
    fig.autofmt_xdate()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def depth_map(depth, domain, title="", path=None, vmax=5.0, river=True, obs_outline=None):
    from matplotlib.colors import ListedColormap
    tf = domain.transform
    ny, nx = depth.shape
    ext = [tf.c, tf.c + nx * tf.a, tf.f + ny * tf.e, tf.f]
    fig, ax = plt.subplots(figsize=(7, 7 * ny / max(nx, 1) + 0.5))
    z = np.where(domain.mask, domain.z, np.nan)
    ax.imshow(z, extent=ext, cmap="Greys", alpha=0.5)
    d = np.where(depth > 0.05, depth, np.nan)
    im = ax.imshow(d, extent=ext, cmap="Blues", vmin=0, vmax=vmax)
    if river and domain.river is not None:
        r = domain.river
        x = tf.c + (r.rj + 0.5) * tf.a
        y = tf.f + (r.ri + 0.5) * tf.e
        ax.scatter(x, y, s=0.5, c="navy", alpha=0.4)
    if obs_outline is not None:
        ax.contour(obs_outline.astype(float), levels=[0.5], colors="red", linewidths=0.8,
                   extent=ext, origin="upper")
    plt.colorbar(im, ax=ax, shrink=0.7, label="max inundation depth [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
