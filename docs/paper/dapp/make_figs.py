"""Figures for the Adaptive DAPP paper outline (docs/paper/dapp/dapp_outline.tex) from the 11.11 runs.

    .venv/bin/python docs/paper/dapp/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
A = json.load(open(ROOT / "data/processed/dapp/abukuma_dapp_adaptive/results.json", encoding="utf-8"))

plt.rcParams.update({"font.family": ["Hiragino Kaku Gothic Pro"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, GRAY, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#a3a29b"

MEAS = [("排水ポンプ（下水道）", "Urban drainage pump"), ("排水機場（農業）", "Agricultural drainage pump"),
        ("雨水貯留（公共）", "Public rainwater storage"), ("雨水貯留（各戸）", "Household rainwater storage"),
        ("田んぼダム", "Paddy-field dam"), ("森林化", "Afforestation"), ("住宅対策", "Housing"), ("堤防", "Levee"),
        ("遊水地", "Retention basin ")]
ZONE = {"上流": "upstream", "中流": "midstream", "下流": "downstream"}
OPT = [("嵩上げ", "raising"), ("移転", "relocation"), ("縮小（0%）", "scale back (0%)"), ("農地へ再転換", "back to farmland"),
       ("貯留深", "depth")]


def en(label):
    """Japanese action label (catalog) -> English, e.g. '堤防・下流 +2 m' -> 'Levee (downstream) +2 m'."""
    name, _, opt = label.partition(" ")
    meas, _, zone = name.partition("・")
    for j, e in MEAS:
        meas = meas.replace(j, e)
    for j, e in OPT:
        opt = opt.replace(j, e)
    return f"{meas.strip()} ({ZONE.get(zone, zone)}) {opt}".strip()



def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- fig 1: initial actions with their best adaptive rule: Coverage vs mean cost (colour = remaining options)
T = A["table"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
cm = matplotlib.colormaps["viridis"]
for r in T:
    a = r["adaptive"]
    rec = r.get("recommended")
    ax.scatter(a["mean_cost"], a["coverage"], s=70 if rec else (40 if r.get("pareto") else 22),
               color=cm(a["remaining_mean"] / 0.8), edgecolor="#0b0b0b" if (rec or r.get("pareto")) else "none",
               lw=1.2 if rec else 0.6, marker="D" if rec else "o", zorder=4)
    lab = "No measures" if r["node"] == 0 else f"#{r['node']}"
    if r.get("pareto") or rec or r["node"] == 81:
        ax.annotate(lab, (a["mean_cost"], a["coverage"]), xytext=(4, -9 if r["node"] == 1 else 3), textcoords="offset points",
                    fontsize=7, color=GRAY)
ax.set_xlabel("Mean cumulative cost, PV [100 million JPY]")
ax.set_ylabel("Coverage (128 evaluation scenarios)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(0, 0.8))
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
cb.set_label("Remaining future options (mean over 2020–2100)", fontsize=7)
h = [plt.Line2D([], [], marker="o", ls="", mfc="#cccccc", mec="#0b0b0b", label="Pareto (black edge)"),
     plt.Line2D([], [], marker="D", ls="", mfc="#cccccc", mec="#0b0b0b", ms=7, label="Recommended")]
ax.legend(handles=h, loc="lower right", fontsize=7)
save(fig, "fig1_initial_actions")

# ---- fig 2: Adaptive Pathways Maps of the recommended initial action (#73) and of #2
cat = A["catalog"]
cmap = matplotlib.colormaps["RdYlBu"]


def pathways(ax, node, title):
    tree = A["trees"][str(node)]
    tot = tree[0]["n"]
    minN = max(1, round(0.03 * tot))
    Tn = [b for b in tree if b["n"] >= minN]
    first = {}
    for b in Tn:
        if b["kind"] == "act":
            first[b["a"]] = min(first.get(b["a"], 9999), b["year"])
    rows = ["init"] + [f"a{x}" for x in sorted(first, key=lambda x: (first[x], x))]
    key = lambda b: f"a{b['a']}" if b["kind"] == "act" else "init"  # noqa: E731
    cnt, off = {}, {}
    for b in Tn:
        k = key(b); cnt[k] = cnt.get(k, -1) + 1; off[b["id"]] = ((cnt[k] % 5) - 2) * 0.06
    Y = lambda b: -rows.index(key(b)) + off.get(b["id"], 0)  # noqa: E731
    for b in Tn:
        ch = [c for c in Tn if c.get("parent") == b["id"]]
        end = 2100 if (b.get("end") or not ch) else max(c["year"] for c in ch)
        ax.plot([b["year"], end], [Y(b), Y(b)], color=cmap(b["met_share"]), lw=0.8 + 6 * b["n"] / tot, solid_capstyle="butt", zorder=2)
        for c in ch:
            ax.plot([c["year"], c["year"]], [Y(b), Y(c)], color=cmap(c["met_share"]), lw=0.8 + 6 * c["n"] / tot, zorder=2)
    for b in Tn:
        ax.scatter(b["year"], Y(b), s=14, facecolor="white", edgecolor="#0b0b0b", lw=0.7, zorder=5)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([f"Initial action #{node}" if k == "init" else en(cat[int(k[1:])]) for k in rows], fontsize=8.5)
    ax.set_xlim(2018, 2102); ax.set_xticks(range(2020, 2101, 10)); ax.grid(axis="y", visible=False)
    ax.set_ylim(-len(rows) + 0.5, 0.5)
    ax.set_title(f"{title}; branches with < {minN} scenarios omitted", fontsize=9, color=GRAY, loc="left")
    return len(rows)


rec = next(r for r in A["table"] if r.get("recommended"))
n73 = len({b["a"] for b in A["trees"][str(rec["node"])] if b["kind"] == "act" and b["n"] >= 4}) + 1
n2 = len({b["a"] for b in A["trees"]["2"] if b["kind"] == "act" and b["n"] >= 4}) + 1
fig, axs = plt.subplots(2, 1, figsize=(7.4, 0.26 * (n73 + n2) + 1.4), gridspec_kw={"height_ratios": [n73, n2]}, sharex=True)
pathways(axs[0], rec["node"], f"(a) Initial action #{rec['node']} (recommended)")
pathways(axs[1], 2, "(b) Initial action #2")
fig.subplots_adjust(left=0.38, right=0.88, top=0.94, bottom=0.09, hspace=0.25)
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
cax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
cb = fig.colorbar(sm, cax=cax); cb.set_label("Share of scenarios meeting\nthe Target on the branch", fontsize=7)
cb.ax.tick_params(labelsize=7)
fig.text(0.01, 0.005, "Line width = number of scenarios (of 128); circle = action taken (median year)", fontsize=7, color=GRAY)
save(fig, "fig2_pathways_map")

# ---- fig 3: time course of #73 and #2 (adaptive rule) with the 11.9 pathway of #73 as reference
yrs = A["years"]
R2 = next(r for r in A["table"] if r["node"] == 2)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for o, lab, col, ls in [(rec["adaptive"]["options"], f"#{rec['node']} adaptive rule", ORANGE, "-"),
                        (R2["adaptive"]["options"], "#2 adaptive rule", BLUE, "-"),
                        (rec["robust"]["options"], f"#{rec['node']} 11.9 pathway (reference)", MUTED, "--")]:
    axs[0].plot(yrs[:len(o["met_so_far"])], o["met_so_far"], color=col, ls=ls, label=lab)
    axs[1].plot(yrs[:len(o["cost_cum"])], o["cost_cum"], color=col, ls=ls)
    axs[2].plot(yrs[:len(o["remaining"]["mean"])], o["remaining"]["mean"], color=col, ls=ls)
    if ls == "-":
        axs[2].fill_between(yrs[:len(o["remaining"]["mean"])], o["remaining"]["p10"], o["remaining"]["p90"], color=col, alpha=0.12, lw=0)
axs[0].set_title("Share meeting the Target so far", fontsize=8)
axs[1].set_title("Mean cumulative cost, PV [100 M JPY]", fontsize=8)
axs[2].set_title("Remaining future options (mean, 10–90%)", fontsize=8)
for a in (axs[0], axs[2]):
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[0].legend(fontsize=6.5)
fig.tight_layout()
save(fig, "fig3_time_course")

# ---- fig 4: planning climate range (standard runs, default target; values of docs/11 11.11.6)
R = ["Low", "Medium", "High", "Low–High"]
cov_all = [0.53, 0.66, 0.70, 0.66]; cov_in = [1.00, 0.77, 0.33, 0.66]; plan = [1.00, 0.83, 0.42, 0.71]
above = [0.48, 0.44, np.nan, np.nan]; below = [np.nan, 0.98, 0.74, np.nan]; cost = [345, 432, 523, 440]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(4)
axs[0].plot(x, plan, "s--", color=MUTED, label="Expected at planning (search scenarios)")
axs[0].plot(x, cov_all, "o-", color=BLUE, label="All (128 evaluation scenarios)")
axs[0].plot(x, cov_in, "^", color=GREEN, ms=7, label="Within the range")
axs[0].plot(x, above, "v", color=ORANGE, ms=7, label="More severe than the range")
axs[0].plot(x, below, "v", color=GRAY, mfc="white", ms=7, label="Milder than the range")
axs[0].set_xticks(x, R); axs[0].set_ylabel("Coverage"); axs[0].set_ylim(0, 1.05)
axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
axs[1].bar(x, cost, color=BLUE, width=0.6)
axs[1].set_xticks(x, R); axs[1].set_ylabel("Mean cumulative cost, PV [100 M JPY]")
for i, c in enumerate(cost):
    axs[1].text(i, c + 8, f"{c}", ha="center", fontsize=7)
fig.tight_layout()
save(fig, "fig4_planning_range")
# ---- fig 5: target ambition x planning range: early preparedness (initial cost) vs future flexibility (remaining options)
# GUI re-searches (fast: 12 search scenarios), recommended initial action; data/processed/dapp/abukuma_dapp_adaptive_gui_<id>/summary.json
RUNS = {("Low", 1.0): "2a17943e", ("Medium", 1.0): "7caa85a9", ("High", 1.0): "a2cdd5b5", ("Low–High", 1.0): "a6a2aeb5",
        ("Low", 1.2): "87707e18", ("Medium", 1.2): "ff42afaf", ("High", 1.2): "6f2a9fad", ("Low–High", 1.2): "4b525724",
        ("Low", 1.4): "420dff8e", ("Medium", 1.4): "bce3f30b", ("High", 1.4): "7ff5c4af", ("Low–High", 1.4): "56d21622"}
MK = {"Low": "v", "Medium": "s", "High": "^", "Low–High": "o"}
TC = {1.0: BLUE, 1.2: GREEN, 1.4: ORANGE}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for (rng, tg), rid in RUNS.items():
    s = json.load(open(ROOT / f"data/processed/dapp/abukuma_dapp_adaptive_gui_{rid}/summary.json", encoding="utf-8"))["recommended"]
    axs[0].scatter(s["initial_cost"], s["remaining_mean"], marker=MK[rng], color=TC[tg], s=40, edgecolor="#0b0b0b", lw=0.4, zorder=3)
    axs[0].annotate(f"#{s['node']}", (s["initial_cost"], s["remaining_mean"]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=GRAY)
    axs[1].scatter(s["initial_cost"], s["coverage"], marker=MK[rng], color=TC[tg], s=40, edgecolor="#0b0b0b", lw=0.4, zorder=3)
axs[0].set_xlabel("Initial cost of the recommended initial action\n[100 M JPY] (early preparedness)"); axs[0].set_ylabel("Remaining future options (mean)")
axs[1].set_xlabel("Initial cost of the recommended initial action\n[100 M JPY]"); axs[1].set_ylabel("Coverage (128 evaluation scenarios)")
for a in axs:
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"Damage Target {t:.1f}×") for t, c in TC.items()] + \
    [plt.Line2D([], [], marker=m, ls="", mfc="white", mec=GRAY, label=r) for r, m in MK.items()]
fig.legend(handles=h, fontsize=6.5, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig5_target_preparedness")
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
