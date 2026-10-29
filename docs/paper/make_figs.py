"""Figures for the paper plan (docs/paper/paper_plan.tex) from the quick RDM / SA / DAPP runs.

    .venv/bin/python docs/paper/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
R = json.load(open(ROOT / "data/processed/rdm/abukuma_quick/results.json", encoding="utf-8"))
S = json.load(open(ROOT / "data/processed/rdm/abukuma_quick/sensitivity.json", encoding="utf-8"))
D = json.load(open(ROOT / "data/processed/dapp/abukuma_dapp_quick/results.json", encoding="utf-8"))

plt.rcParams.update({"font.family": ["Hiragino Kaku Gothic Pro"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#a3a29b"
TEXT2 = "#52514e"
STAKE = list(R["robustness"]["stakeholders"])
LBL = {"rain_factor": "降雨倍率", "rain_gamma": "降雨強度分布 γ", "paddy_effect": "田んぼダム効果",
       "sea_level_rise_m": "海面上昇", "cost_factor": "事業費係数", "asset_factor": "資産額係数"}
CLIM = {"current": "現在気候", "2C": "2℃", "4C": "4℃"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---- Fig. 2 Pareto front (nominal scenario)
arc, pol = R["archive"], R["policies"]
par = set(R["pareto"])
fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs = [a["cost"] for i, a in enumerate(arc) if i not in par and a["feasible"]]
ys = [a["damage"] for i, a in enumerate(arc) if i not in par and a["feasible"]]
ax.scatter(xs, ys, s=14, c=MUTED, label="評価した案（非 Pareto）", zorder=2)
pa = [arc[i] for i in sorted(par)]
sc = ax.scatter([a["cost"] for a in pa], [a["damage"] for a in pa], s=34, c=[a["ecology"] for a in pa],
                cmap="Blues", vmin=63.5, vmax=65.0, edgecolors="#2a78d6", linewidths=0.8, zorder=3, label="Pareto 解")
fig.colorbar(sc, ax=ax, pad=0.02).set_label("Ecology", color=TEXT2)
ax.axvline(R["config"]["budget_oku"], color=TEXT2, lw=0.8, ls="--")
ax.text(R["config"]["budget_oku"], 760, "予算 ", color=TEXT2, va="top", ha="right", fontsize=7.5)
mark = {"施策なし": "施策なし（国）", "案12": "案12（総合）", "案10": "案10（上流）", "案11": "案11（下流）", "案6": "案6（住民）"}
for p in pol:
    if p["label"] in mark:
        a = arc[p["archive"]]
        off = (-10, -12) if p["label"] == "案6" else (5, 4)
        ax.annotate(mark[p["label"]], (a["cost"], a["damage"]), xytext=off, textcoords="offset points", fontsize=7.5)
ax.set_xlabel("事業費 [億円, PV]")
ax.set_ylabel("被害額 [億円]（名目シナリオ）")
ax.legend(loc="lower left", fontsize=7.5)
save(fig, "fig2_pareto")

# ---- Fig. 3 relative max regret (stakeholder x Pareto policy) + cross regret
thr = [R["config"]["stakeholders"][s].get("regret_threshold", 0.05) for s in STAKE]
mx = np.array([R["robustness"]["stakeholders"][s]["summary"]["max"] for s in STAKE]) / np.array(thr)[:, None]
labels = [p["label"] for p in pol]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [3.2, 1.25], "wspace": 0.45})
ax = axs[0]
im = ax.imshow(np.minimum(mx, 10), cmap="Oranges", vmin=0, vmax=10, aspect="auto")
ax.set_xticks(range(len(labels)), [l.replace("施策なし", "なし") for l in labels], rotation=60, fontsize=7)
ax.set_yticks(range(len(STAKE)), STAKE)
ax.grid(False)
for j, s in enumerate(STAKE):
    k = R["robustness"]["stakeholders"][s]["minimax"]
    ax.add_patch(plt.Rectangle((k - .5, j - .5), 1, 1, fill=False, ec="#0b0b0b", lw=1.4))
k = R["robustness"]["overall"]["minimax"]
ax.add_patch(plt.Rectangle((k - .5, -.5), 1, len(STAKE), fill=False, ec="#2a78d6", lw=1.6, ls="--"))
ax.set_title("(a) 最大 Regret ÷ 許容閾値（□＝各主体の Minimax、破線＝総合推奨）", fontsize=8, loc="left")
fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
cr = np.array(S["preference"]["conflict"]["cross_max_regret"])
ax = axs[1]
im2 = ax.imshow(cr, cmap="Oranges", vmin=0, vmax=0.4)
ax.set_xticks(range(4), STAKE, rotation=45, fontsize=7)
ax.set_yticks(range(4), STAKE, fontsize=7)
ax.grid(False)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cr[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#0b0b0b" if cr[i, j] < 0.25 else "white")
ax.set_title("(b) 対立：列の主体の推奨案での\n　　行の主体の最大 Regret", fontsize=8, loc="left")
save(fig, "fig3_regret")

# ---- Fig. 4 Sobol indices
so = S["sobol"]
inp = so["inputs"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
outs = [("damage", "被害額"), ("reduction", "被害軽減額"), ("pop_hazard", "浸水人口*")]
w = 0.26
for o_i, (o, ol) in enumerate(outs):
    st = np.array(so["results"][o]["ST"])
    ci = np.array(so["results"][o]["ST_ci"]) if "ST_ci" in so["results"][o] else None
    x = np.arange(len(inp)) + (o_i - 1) * w
    ax.bar(x, st, w * 0.9, color=CAT[o_i], label=ol)
    if ci is not None:
        ax.errorbar(x, st, yerr=[st - ci[:, 0], ci[:, 1] - st], fmt="none", ecolor=TEXT2, lw=0.7, capsize=1.5)
ax.set_xticks(range(len(inp)), [LBL[i] for i in inp], rotation=30, ha="right")
ax.set_ylabel("全効果指数 $S_T$（95%CI）")
ax.set_title("(a) 総合推奨案（案12）の出力", fontsize=8, loc="left")
ax.legend(fontsize=7.5, ncol=3, loc="upper right")
ax = axs[1]
M = np.array([so["results"][f"regret:{s}"]["ST"] for s in STAKE])
M1 = np.array([so["results"][f"regret:{s}"]["S1"] for s in STAKE])
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
ax.grid(False)
for i in range(len(STAKE)):
    for j in range(len(inp)):
        ax.text(j, i, f"{M[i, j]:.2f}\n({max(M1[i, j], 0):.2f})", ha="center", va="center", fontsize=5.8,
                color="white" if M[i, j] > 0.5 else "#0b0b0b")
ax.set_xticks(range(len(inp)), [LBL[i] for i in inp], rotation=30, ha="right")
ax.set_yticks(range(len(STAKE)), STAKE)
ax.set_title("(b) 各主体の推奨案の Regret：$S_T$（$S_1$）", fontsize=8, loc="left")
save(fig, "fig4_sobol")

# ---- Fig. 5 decision sensitivity
dec = S["decision"]
plabs = S["policies"]
fig, ax = plt.subplots(figsize=(6.0, 2.2))
for j, s in enumerate(STAKE):
    left = 0
    for p, f in enumerate(dec[s]["best_frequency"]):
        if f > 0:
            ax.barh(j, f, left=left, color=CAT[p], edgecolor="white", linewidth=1.0)
            if f > 0.08:
                ax.text(left + f / 2, j, f"{f:.0%}", ha="center", va="center", fontsize=7, color="white")
            left += f
    ax.text(1.02, j, f"名目最良 {plabs[dec[s]['nominal_best']]}・逆転 {dec[s]['rank_reversal_rate']:.0%}",
            va="center", fontsize=7, color=TEXT2)
ax.set_yticks(range(len(STAKE)), STAKE)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("各案が予算内で最良となる割合（代理モデル上の 4,096 サンプル）")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=CAT[p]) for p in range(len(plabs))], plabs, ncol=5, fontsize=7,
          loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="y", visible=False)
save(fig, "fig5_decision")

# ---- Fig. 6 DAPP damage ratio time series
strs, clims, yrs = D["strategies"], D["climates"], D["years"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
thr_atp = D["config"]["atp"]["damage_ratio"]
for ci, c in enumerate(clims):
    ax = axs[ci]
    for si, s in enumerate(strs):
        r = D["records"][f"{s}|{c}"]
        y = [p["damage_ratio"] for p in r["periods"]]
        ax.plot([p["year"] + 5 for p in r["periods"]], y, color=CAT[si], lw=1.8, marker="o", ms=3, label=s)
        if r["atp"]["damage"]:
            ax.scatter([r["atp"]["damage"] + 5], [y[yrs.index(r["atp"]["damage"])]], marker="x", s=36, color=CAT[si], zorder=5)
    ax.axhline(thr_atp, color=TEXT2, lw=0.8, ls="--")
    ax.set_title(CLIM[c], fontsize=8.5, loc="left")
    ax.set_xticks([2025, 2045, 2065, 2085])
axs[0].text(2026, thr_atp + 0.03, f"ATP 閾値 ×{thr_atp}", fontsize=7, color=TEXT2)
axs[0].set_ylabel("代表洪水の被害額\n（対策なし・現在気候 = 1）")
axs[2].legend(fontsize=7, loc="upper left")
save(fig, "fig6_dapp_series")

# ---- Fig. 7 pathways (Gantt) under 4C
c = "4C"
vars_ = [v["name"] for v in D["variables"]]
used = sorted({sg["var"] for s in strs for sg in D["pathways"][f"{s}|{c}"]["segments"]}, key=vars_.index)
fig, axs = plt.subplots(1, len(strs) - 1, figsize=(7.2, 2.9), sharey=True)
for k, s in enumerate(strs[1:]):
    ax = axs[k]
    pw = D["pathways"][f"{s}|{c}"]
    col = CAT[k + 1]
    for sg in pw["segments"]:
        y = used.index(sg["var"])
        ax.barh(y, sg["complete"] - sg["start"], left=sg["start"], height=0.6, color=col, alpha=0.3)
        ax.barh(y, sg["end"] - sg["complete"], left=sg["complete"], height=0.6, color=col)
        ax.plot([sg["start"]], [y], marker="o", ms=3.5, color=col)
    for t in pw["triggers"]:
        ax.axvline(t["year"], color=TEXT2, lw=0.8, ls=":")
    if pw["atp"]["damage"]:
        ax.axvline(pw["atp"]["damage"], color="#e34948", lw=1.0)
        ax.text(pw["atp"]["damage"], len(used) - 0.4, " ATP", color="#e34948", fontsize=7)
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2060, 2100])
    ax.set_title(s, fontsize=8.5, loc="left")
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(used)), used, fontsize=7)
axs[0].invert_yaxis()
save(fig, "fig7_pathways")

# ---- Fig. 8 surrogate validation
val = S["surrogate"]["validation"]
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.6))
for ax, (o, ol) in zip(axs, [("damage", "被害額 [億円]"), ("pop_hazard", "浸水人口 [人]")]):
    allv = []
    for pi, (p, d) in enumerate(val.items()):
        ax.scatter(d[o]["y_val"], d[o]["yhat_val"], s=16, color=CAT[pi], label=p)
        allv += d[o]["y_val"] + d[o]["yhat_val"]
    lo, hi = min(allv), max(allv)
    ax.plot([lo, hi], [lo, hi], color=TEXT2, lw=0.8, ls="--")
    rr = [d[o]["selected"]["rel_rmse"] for d in val.values()]
    ax.set_title(f"{ol}  相対 RMSE {min(rr):.0%}–{max(rr):.0%}", fontsize=8, loc="left")
    ax.set_xlabel("RRI（検証 LHS）")
axs[0].set_ylabel("代理モデル予測")
axs[1].legend(fontsize=6.5)
save(fig, "fig8_surrogate")
print("figures ->", OUT)
