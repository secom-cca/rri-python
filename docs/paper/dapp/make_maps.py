"""Placement and simulated-effect maps of selected states of the v4 Adaptive DAPP (full RRI, not the surrogate).

States (decision vectors parsed from the initial-action labels of the run):
    A  recommended initial action (#2 in v4)
    B  A + the most frequent additions by 2050 on its pathway (paddy-field dam upstream 100 %, levee downstream +2 m)
    C  a low-cost Pareto initial action that waits (#24 in v4)
RRI runs: no measures and A, B, C under the current climate and 4 deg C (rain x1.2, sea level +0.70 m), nominal effect
coefficients (planner's belief).  Maps are cached in data/processed/dapp/<run>/maps/.

    .venv/bin/python docs/paper/dapp/make_maps.py [--run abukuma_dapp_adaptive_v4] [--rec 2] [--alt 24] [--plot-only]
    v2 (#73 vs #2): --run abukuma_dapp_adaptive --explore abukuma_explore_v2 --rec 73 --alt 2 --add "堤防・下流 +2 m" --tag _v2
Writes docs/paper/dapp/figs/fig6_placement.{pdf,png} and fig7_effect_maps.{pdf,png}.
"""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="abukuma_dapp_adaptive_v4")
ap.add_argument("--rec", type=int, default=2)
ap.add_argument("--alt", type=int, default=24)
ap.add_argument("--add", default="田んぼダム・上流 100%;堤防・下流 +2 m", help="additions of state B (';'-separated)")
ap.add_argument("--plot-only", action="store_true")
ap.add_argument("--explore", default="abukuma_explore_v4", help="exploration whose RRI variables the run uses")
ap.add_argument("--tag", default="", help="suffix of the figure names (e.g. _v2)")
a = ap.parse_args()
OUT = ROOT / "data/processed/dapp" / a.run / "maps"
OUT.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).parent / "figs"
A = json.load(open(ROOT / "data/processed/dapp" / a.run / "results.json", encoding="utf-8"))
label = {r["node"]: r["label"] for r in A["table"]}
log = lambda m: print(m, flush=True)  # noqa: E731

EFF = {"levee_effect": 0.9, "retention_effect": 0.85, "paddy_effect": 0.6, "forest_effect": 0.6, "raise_effect": 0.8,
       "forest_mgmt_effect": 1.0}
CLIM = {"current": {"rain_factor": 1.0, "sea_level_rise_m": 0.0, "rain_gamma": 1.0},
        "4C": {"rain_factor": 1.2, "sea_level_rise_m": 0.70, "rain_gamma": 1.0}}


def parts(lab):
    return {} if lab == "施策なし" else dict(p.split(" ", 1) for p in lab.split(" + "))


STATES = {"A": parts(label[a.rec]), "C": parts(label[a.alt])}
STATES["B"] = {**STATES["A"], **dict(p.split(" ", 1) for p in a.add.split(";"))}
STATES = {k: STATES[k] for k in "ABC"}
NAMES = {"A": f"#{a.rec} initial", "B": f"#{a.rec} + by 2050", "C": f"#{a.alt} initial"}


def b64(s, dt, shape):
    return np.frombuffer(base64.b64decode(s), dt).reshape(shape).astype(float)


if not a.plot_only:
    from rri_sim.rdm import candidates as CA
    from rri_sim.rdm.evaluate import gparams_for
    from rri_sim.roadmap.physics import rri_evaluator
    ev = rri_evaluator(a.explore, log)
    V, s = ev.variables, ev.s
    shape = s.d.shape

    def xvec(comp):
        x = [0] * len(V)
        for i, v in enumerate(V):
            if v.name in comp:
                o = comp[v.name]
                x[i] = float(o.rstrip("%")) / 100 if v.continuous else v.options.index(o)
        assert set(comp) <= {v.name for v in V}, set(comp) - {v.name for v in V}
        return x

    static = {"mask": s.d.mask, "z": np.where(s.d.mask, s.d.z, np.nan), "landclass": s.d.landclass,
              "riv_i": s.d.river.ri, "riv_j": s.d.river.rj, "riv_area": s.d.river.area_km2,
              "zones": np.stack([ev.zm[z] for z in ("上流", "中流", "下流")]),
              "class_names": np.array(s.class_names)}
    cells = {}
    for k, comp in STATES.items():
        x = xvec(comp)
        for v, xi in zip(V, x):
            if v.active(xi):
                c = v.cells[:v.n_cells(xi)] if v.continuous else v.cells
                cells[f"{k}|{v.name}|{comp[v.name]}"] = np.asarray(c, int)
    np.savez_compressed(OUT / "static.npz", **static, **{"cells::" + n: c for n, c in cells.items()})
    for cn, c in CLIM.items():
        for k in ["0"] + list(STATES):
            f = OUT / f"{k}_{cn}.npz"
            if f.exists():
                continue
            x = [0] * len(V) if k == "0" else xvec(STATES[k])
            sim = {**c, **EFF}
            eds = CA.edits(V, x, sim)
            log(f"RRI {k} {cn}: {CA.describe(V, x) or ['施策なし']}")
            res = s.run(ev.event, eds, {**gparams_for(sim), **ev.gextra})
            np.savez_compressed(f, depth=b64(res["max_depth_cm"], "<u2", shape) / 100, damage=b64(res["damage"], "<f4", shape),
                                damage_total=res["damage_total_oku"], pop_hazard=res["sub_indicators"]["flooded_population"],
                                embankment=np.asarray(res["embankment"]), x=np.asarray(x, float))
            log(f"  被害 {res['damage_total_oku']:.1f} 億円・浸水人口 {res['sub_indicators']['flooded_population']:.0f}")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "font.family": "DejaVu Sans"})
S = np.load(OUT / "static.npz", allow_pickle=True)
mask, z = S["mask"], S["z"]
ny, nx = mask.shape
zones = S["zones"]
ZN = ["Upstream", "Midstream", "Downstream"]
ZC = ["#dbe9f6", "#fdf0d5", "#e3f1e0"]
cellsets = {k[7:]: S[k] for k in S.files if k.startswith("cells::")}
riv_major = S["riv_area"] >= 50.0


def base_map(ax, zone_fill=True, river="#4a7fb5"):
    if zone_fill:
        img = np.full((ny, nx, 4), 0.0)
        for zi in range(3):
            img[zones[zi]] = matplotlib.colors.to_rgba(ZC[zi])
        ax.imshow(img, interpolation="nearest")
        for zi in range(3):
            ax.contour(zones[zi].astype(float), [0.5], colors="#9a9a9a", linewidths=0.4)
    else:
        hs = np.where(mask, z, np.nan)
        ax.imshow(hs, cmap="Greys", alpha=0.25, interpolation="nearest")
    ax.contour(mask.astype(float), [0.5], colors="#3a3a3a", linewidths=0.6)
    ri, rj = S["riv_i"][riv_major], S["riv_j"][riv_major]
    ax.scatter(rj, ri, s=0.6, c=river, marker="s", lw=0, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(ny - 0.5, -0.5)


# measure styles: (english, colour, marker or None for area fill)
STY = {"堤防": ("Levee raising", "#c0392b", "s"), "遊水地": ("Retention basins", "#1f4e9c", "D"),
       "田んぼダム": ("Paddy-field dam", "#7cb342", None), "森林化": ("Afforestation", "#1b5e20", None),
       "住宅対策": ("Housing (R raise / M relocate)", "#8e44ad", "^"), "排水ポンプ（下水道）": ("Urban pump", "#e67e22", "o"),
       "排水機場（農業）": ("Agricultural pump", "#b9770e", "o"), "雨水貯留（公共）": ("Public storage", "#00897b", "P"),
       "雨水貯留（各戸）": ("Household storage", "#26a69a", "x")}


def draw_state(ax, k, title, highlight=()):
    base_map(ax)
    present = set()
    for key, c in cellsets.items():
        st, name, opt = key.split("|")
        if st != k:
            continue
        meas = name.split("・")[0].rstrip("0123456789")
        en, col, mk = STY[meas]
        present.add(meas)
        ii, jj = np.unravel_index(c, (ny, nx))
        hl = name in highlight
        if mk is None:
            img = np.zeros((ny, nx, 4)); img[ii, jj] = matplotlib.colors.to_rgba(col, 0.85 if opt == "100%" else 0.5)
            ax.imshow(img, interpolation="nearest", zorder=2)
        else:
            sz = {"s": 7, "D": 22, "^": 26, "o": 9, "P": 14, "x": 10}[mk]
            dx = 1.2 if meas.startswith("排水") else 0.0      # pumps share cells with public storage
            ax.scatter(jj + dx, ii, s=sz * (1.8 if hl else 1), marker=mk, c=col, lw=0 if mk != "x" else 0.8,
                       edgecolors="#000000" if hl else None, zorder=4)
            if meas in ("堤防", "住宅対策"):
                txt = opt.replace("嵩上げ +1.0 m", "R").replace("移転", "M")
                ax.annotate(txt + (" (added)" if hl else ""), (np.median(jj), np.median(ii)),
                            xytext=(-8, 0) if np.median(jj) > 0.6 * nx else (6, -2), textcoords="offset points",
                            ha="right" if np.median(jj) > 0.6 * nx else "left", fontsize=6.5, color=col, fontweight="bold" if hl else "normal", zorder=6)
    ax.set_title(title, loc="left")
    return present


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(FIG / f"{name}.pdf")


# fig 6: placement
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.9))
pres = set()
for ax, k in zip(axs, "ABC"):
    hl = {n for n in STATES["B"] if STATES["B"][n] != STATES["A"].get(n)} if k == "B" else ()
    pres |= draw_state(ax, k, f"({'abc'['ABC'.index(k)]}) {NAMES[k]}", hl)
h = [Patch(fc=ZC[i], ec="#9a9a9a", label=ZN[i]) for i in range(3)]
h += [Patch(fc=STY[m][1], label=STY[m][0]) if STY[m][2] is None else
      Line2D([], [], marker=STY[m][2], ls="", color=STY[m][1], label=STY[m][0]) for m in STY if m in pres]
h += [Line2D([], [], marker="s", ls="", color="#4a7fb5", ms=3, label="River channel (≥ 50 km²)")]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(wspace=0.04)
save(fig, "fig6_placement" + a.tag)

# fig 7: simulated effects (rows: climate x quantity, columns: states)
R = {(k, cn): np.load(OUT / f"{k}_{cn}.npz") for k in ["0", "A", "B", "C"] for cn in CLIM}
dcm = ListedColormap(["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7f7f7", "#fcbba1", "#fb6a4a", "#a50f15"])
dnorm = BoundaryNorm([-2, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 2], dcm.N)
rows = [(cn, q) for cn in CLIM for q in ("depth", "damage")]
fig, axs = plt.subplots(len(rows), 3, figsize=(6.6, 10.4))
CL = {"current": "Current climate", "4C": "4 °C (rain ×1.2, SLR +0.70 m)"}
summary = {}
for r, (cn, q) in enumerate(rows):
    for c, k in enumerate("ABC"):
        ax = axs[r, c]
        base_map(ax, zone_fill=False, river="#d0d0d0")
        d = R[(k, cn)][q] - R[("0", cn)][q]
        if q == "depth":
            d = np.where(mask & ((R[(k, cn)]["depth"] > 0.01) | (R[("0", cn)]["depth"] > 0.01)), d, np.nan)
            im = ax.imshow(d, cmap=dcm, norm=dnorm, interpolation="nearest", zorder=2)
        else:
            d = np.where(mask & (np.abs(d) > 1e3), d / 1e6, np.nan)     # million JPY per cell
            im2 = ax.imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(0, -50, 50), interpolation="nearest", zorder=2)
        t0, t1 = float(R[("0", cn)]["damage_total"]), float(R[(k, cn)]["damage_total"])
        p0, p1 = float(R[("0", cn)]["pop_hazard"]), float(R[(k, cn)]["pop_hazard"])
        summary[f"{k}|{cn}"] = {"damage0": t0, "damage": t1, "pop0": p0, "pop": p1}
        if q == "damage":
            ax.set_xlabel(f"Damage {t0:.0f} → {t1:.0f} (100 M JPY)\nPop. ≥ 0.5 m {p0:,.0f} → {p1:,.0f}", fontsize=6.6, labelpad=2)
        if r == 0:
            ax.set_title(f"({'abc'[c]}) {NAMES[k]}", loc="left")
        if c == 0:
            ax.set_ylabel(f"{CL[cn]}\n{'Δ max depth' if q == 'depth' else 'Δ damage'}", fontsize=7.5)
fig.subplots_adjust(wspace=0.03, hspace=0.16, right=0.84)
cb = fig.colorbar(im, cax=fig.add_axes([0.87, 0.56, 0.015, 0.3]), extend="both")
cb.set_label("Δ max inundation depth [m]\n(with − without measures)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
cb = fig.colorbar(im2, cax=fig.add_axes([0.87, 0.14, 0.015, 0.3]), extend="both")
cb.set_label("Δ damage per cell [million JPY]\n(with − without measures)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
save(fig, "fig7_effect_maps" + a.tag)
(OUT / "summary.json").write_text(json.dumps({"states": STATES, "names": NAMES, "climates": CLIM, "effects": EFF, "totals": summary},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
