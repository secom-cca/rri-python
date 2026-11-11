"""Initial-action distribution and Adaptive Pathways Maps (DAPP) of the forecast DAPP (v6).

    .venv/bin/python docs/paper/dapp/make_pathways_v6.py [run_name]

figs/v6_initial_actions.pdf : (a) nominal Pareto set of initial candidates and the 15 selected,
                              (b) each initial action under uncertainty (cheapest / highest-Coverage rule of its front)
figs/v6_pathways_map.pdf    : pathways maps of representative initial actions under their highest-Coverage rule
                              (branches built from the per-scenario action records)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "abukuma_dapp_forecast_v6"
R = json.load(open(ROOT / f"data/processed/dapp/{RUN}/results.json", encoding="utf-8"))
V, LV = R["variables"], R["levels"]

plt.rcParams.update({"font.family": ["Hiragino Sans", "Hiragino Kaku Gothic Pro"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
GRAY = "#52514e"
MEAS = {"levee": "Levee", "retention": "Retention basins", "paddy_dam": "Paddy-field dams", "forest_mgmt": "Forest management",
        "relocate": "Housing", "pump_urban": "Urban drainage pump", "pump_agri": "Agricultural drainage pump",
        "storage_public": "Public storage", "storage_private": "Household storage"}
ZONE = {"上流": "up", "中流": "mid", "下流": "down"}
OPT = {"嵩上げ": "raising", "移転": "relocation", "なし": "none"}


def act_label(i, li):
    v = V[i]
    name = MEAS.get(v["type"], v["name"].split("・")[0])
    if v["type"] == "retention":
        return f"{name} (up)"
    opt = f"{LV[i][li]:.0%}" if v["continuous"] else v["options"][li]
    for j, e in OPT.items():
        opt = opt.replace(j, e)
    opt = opt.replace(" mm/h", " mm/h").replace("+1.0 m", "+1 m")
    return f"{name} ({ZONE.get(v['zone'], v['zone'])}) {opt}"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


T = R["table"]
REP = [7, 10]           # representative initial actions (cheap paddy-field dams / early levee)
SHOW = {0, 4, 7, 2, 13, 10, 8, 1, 3}
SHOWB = {0, 4, 7, 10, 5}

# ---------------------------------------------------------------- fig: initial actions
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"wspace": 0.42})
ax = axs[0]
F0 = R["initial_front"]
x = np.array([f["initial_cost"] for f in F0]); y = np.array([f["ead_pv"] for f in F0]) / 1e3
ax.scatter(x, y, s=8, color="#c9c8c2", label=f"Pareto set on the nominal world ({len(F0)})", zorder=2)
cm = matplotlib.colormaps["viridis"]
for r in T:
    ax.scatter(r["initial_cost"], r["nominal"]["ead_pv"] / 1e3, s=30, color=cm(r["initial_cost"] / 500), edgecolor="#0b0b0b",
               lw=0.6, zorder=4)
    if r["init"] in SHOW:
        ax.annotate(f"#{r['init']}", (r["initial_cost"], r["nominal"]["ead_pv"] / 1e3), xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=GRAY)
ax.scatter([], [], s=30, color="#888", edgecolor="#0b0b0b", lw=0.6, label="Selected initial actions (15)")
ax.set_xlabel("Initial cost [100 M JPY]")
ax.set_ylabel("PV of annual expected damage,\n2020–2100, nominal world [100 G JPY]")
ax.set_title("(a) Candidates, no adaptation\n(nominal world)", fontsize=8.5, loc="left", color=GRAY)
ax.legend(fontsize=7, loc="upper right")

ax = axs[1]
for r in T:
    fr = sorted(r["front"], key=lambda f: f["mean_cost"])
    c = cm(r["initial_cost"] / 500)
    ax.plot([f["mean_cost"] for f in fr], [f["coverage"] for f in fr], color=c, lw=0.7, alpha=0.6, zorder=2)
    lo, hi = fr[0], max(fr, key=lambda f: (f["coverage"], -f["mean_cost"]))
    ax.scatter(lo["mean_cost"], lo["coverage"], s=16, color=c, marker="o", edgecolor="none", zorder=3)
    ax.scatter(hi["mean_cost"], hi["coverage"], s=34, color=c, marker="D", edgecolor="#0b0b0b", lw=0.6, zorder=4)
    if r["init"] in SHOWB:
        dy = {0: 3, 4: -10, 7: 4, 10: -11, 5: -11}.get(r["init"], 3)
        ax.annotate(f"#{r['init']}", (hi["mean_cost"], hi["coverage"]), xytext=(3, dy), textcoords="offset points", fontsize=7, color=GRAY)
ax.scatter([], [], s=16, color="#888", label="Cheapest rule"); ax.scatter([], [], s=34, marker="D", color="#888", edgecolor="#0b0b0b",
                                                                          lw=0.6, label="Highest-Coverage rule")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Mean cumulative cost, PV [100 M JPY]")
ax.set_ylabel("Target Coverage (256 scenarios)")
ax.set_title("(b) With adaptation\n(front of each initial action)", fontsize=8.5, loc="left", color=GRAY)
ax.legend(fontsize=7, loc="lower right")
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(0, 500))
cb = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Initial cost [100 M JPY]", fontsize=7)
save(fig, "v6_initial_actions")


# ---------------------------------------------------------------- fig: pathways maps
def tree(rule):
    """Trie of the per-scenario action sequences: node -> n, met share, median year."""
    met = [fv is None for fv in rule["first_violation"]]
    X = [w["rain_2100"] for w in R["worlds"]]
    nodes = {(): {"n": len(met), "met": sum(met), "x": sum(X), "years": [2020], "seq": ()}}
    for s, ev in enumerate(rule["events"]):
        seq = ()
        for e in ev:
            if e["kind"] != "act":
                continue
            seq = seq + ((e["var"], e["level"]),)
            nd = nodes.setdefault(seq, {"n": 0, "met": 0, "x": 0.0, "years": [], "seq": seq})
            nd["n"] += 1; nd["met"] += met[s]; nd["x"] += X[s]; nd["years"].append(e["year"])
    for nd in nodes.values():
        nd["year"] = float(np.median(nd["years"]))
    return nodes


cmap = matplotlib.colormaps["YlOrRd"]
XN = matplotlib.colors.Normalize(1.0, 1.3)
col = lambda b: cmap(0.12 + 0.88 * XN(b["x"] / b["n"]))  # noqa: E731


def pathways(ax, r, rule, title, min_share=0.03):
    nodes = tree(rule)
    tot = nodes[()]["n"]
    minN = max(1, round(min_share * tot))
    Tn = {k: v for k, v in nodes.items() if v["n"] >= minN}
    first = {}
    for k, v in Tn.items():
        if k:
            first[k[-1]] = min(first.get(k[-1], 9999), v["year"])
    rows = [()] + sorted(first, key=lambda a: (first[a], a))
    cnt, off = {}, {}
    for k in sorted(Tn, key=lambda k: (len(k), Tn[k]["year"])):
        rk = k[-1] if k else ()
        cnt[rk] = cnt.get(rk, -1) + 1; off[k] = ((cnt[rk] % 5) - 2) * 0.07
    Y = lambda k: -rows.index(k[-1] if k else ()) + off[k]  # noqa: E731
    for k, b in Tn.items():
        ch = [c for c in Tn if len(c) == len(k) + 1 and c[:len(k)] == k]
        stay = b["n"] - sum(Tn[c]["n"] for c in ch)      # scenarios ending on this branch (or on omitted children)
        end = 2100 if stay > 0 or not ch else max(Tn[c]["year"] for c in ch)
        ax.plot([b["year"], end], [Y(k), Y(k)], color=col(b), lw=0.8 + 6 * b["n"] / tot,
                solid_capstyle="butt", zorder=2)
        for c in ch:
            ax.plot([Tn[c]["year"]] * 2, [Y(k), Y(c)], color=col(Tn[c]), lw=0.8 + 6 * Tn[c]["n"] / tot, zorder=2)
    for k, b in Tn.items():
        ax.scatter(b["year"], Y(k), s=14, facecolor="white", edgecolor="#0b0b0b", lw=0.7, zorder=5)
        if k:
            ax.annotate(f"{b['n']}" + (f" ({b['met'] / b['n']:.0%} met)" if b["met"] < b["n"] else ""), (b["year"], Y(k)), xytext=(3, 2.5), textcoords="offset points", fontsize=6, color=GRAY)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([f"Initial action #{r['init']}" if a == () else act_label(*a) for a in rows], fontsize=8)
    ax.set_xlim(2018, 2102); ax.set_xticks(range(2020, 2101, 10)); ax.grid(axis="y", visible=False)
    ax.set_ylim(-len(rows) + 0.5, 0.6)
    p = rule["policy"]
    ax.set_title(f"{title}\nrule: ρ {p['rho']}, look-ahead {p['lookahead']} yr, {p['k']} action(s)/decision"
                 f"{', wait' if p['wait'] else ''}; Coverage {rule['coverage']:.0%}, mean cost {rule['mean_cost']:.0f}",
                 fontsize=7.5, color=GRAY, loc="left")
    return len(rows)


def best(r):
    return max(r["front"], key=lambda f: (f["coverage"], -f["mean_cost"]))


def cheap_at(r, cov):
    ok = [f for f in r["front"] if f["coverage"] >= cov]
    return min(ok, key=lambda f: f["mean_cost"]) if ok else None


panels = []
for n in REP:
    r = next(t for t in T if t["init"] == n)
    panels.append((r, best(r), f"#{n} highest Coverage"))
r7 = next(t for t in T if t["init"] == 7)
mid = cheap_at(r7, 0.8)
if mid is not None and mid is not best(r7):
    panels.insert(1, (r7, mid, "#7 cheapest rule with Coverage >= 80%"))
sizes = []
for r, rule, _ in panels:
    nodes = tree(rule); tot = nodes[()]["n"]; m = max(1, round(0.03 * tot))
    sizes.append(len({k[-1] for k, v in nodes.items() if k and v["n"] >= m}) + 1)
fig, axs = plt.subplots(len(panels), 1, figsize=(7.6, 0.27 * sum(sizes) + 0.55 * len(panels) + 0.8),
                        gridspec_kw={"height_ratios": sizes}, sharex=True)
for i, (ax, (r, rule, title)) in enumerate(zip(axs, panels)):
    pathways(ax, r, rule, f"({'abc'[i]}) {title}")
fig.subplots_adjust(left=0.30, right=0.86, top=0.95, bottom=0.07, hspace=0.55)
sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cmap(np.linspace(0.12, 1, 256))), norm=XN)
cax = fig.add_axes([0.89, 0.3, 0.015, 0.4])
cb = fig.colorbar(sm, cax=cax); cb.set_label("Mean true rainfall factor in 2100\nof the scenarios on the branch", fontsize=7)
cb.ax.tick_params(labelsize=7)
fig.text(0.01, 0.02, "Line width / number = scenarios (of 256) taking the branch (share meeting the Target 2030–2100 if < 100%); circle = median decision year; branches < 3% of scenarios omitted.\n"
         "Initial action #7: paddy-field dams (up 100%, mid 50%); #10: levee down +1 m, paddy-field dams up.",
         fontsize=6.5, color=GRAY, va="top")
save(fig, "v6_pathways_map")

print("panels:", [(r["init"], rule["policy"], round(rule["coverage"], 3), round(rule["mean_cost"])) for r, rule, _ in panels])
