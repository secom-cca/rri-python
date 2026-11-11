"""Figures and tables of the revised paper (model revision 11.12: 1K-RRI, no effect coefficients, forecast and learning,
frequency-based benefits) from the runs abukuma_dapp_forecast_v6* and abukuma_dapp_forecast_v5.

    .venv/bin/python docs/paper/dapp/make_figs_v6.py
Writes figs/v6_*.pdf/png and tab_v6_*.tex.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
DAPP = ROOT / "data/processed/dapp"
plt.rcParams.update({"font.family": ["Hiragino Kaku Gothic Pro"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, GRAY, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#c0392b", "#7c3aed"
MEAS = [("排水ポンプ（下水道）", "Urban pump"), ("排水機場（農業）", "Agri. pump"), ("雨水貯留（公共）", "Public storage"),
        ("雨水貯留（各戸）", "Household storage"), ("田んぼダム", "Paddy-field dam"), ("森林整備", "Forest management"),
        ("住宅対策", "Housing"), ("堤防", "Levee"), ("遊水地", "Retention basins")]
ZONE = {"上流": "up", "中流": "mid", "下流": "down"}


def en_var(name):
    m, _, z = name.partition("・")
    for j, e in MEAS:
        m = m.replace(j, e)
    return f"{m}, {ZONE.get(z, z)}"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load(name):
    p = DAPP / name / "results.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


R6 = load("abukuma_dapp_forecast_v6")
R5 = load("abukuma_dapp_forecast_v5")

# ---- 1K-RRI hydrographs (2019)
V = json.load(open(ROOT / "data/processed/k1rri_validation_abukuma_2019_1000.json", encoding="utf-8"))
gs = ["阿久津", "本宮", "伏黒", "江尻"]
GEN = {"阿久津": "Akutsu", "本宮": "Motomiya", "伏黒": "Fushiguro", "江尻": "Ejiri"}
fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9), sharex=True)
for ax, g in zip(axs, gs):
    sim = np.array(V["sim"][g], float); obs = np.array([np.nan if x is None else x for x in V["obs"][g]], float)
    t = np.arange(len(sim))
    ax.plot(t, obs, color=GRAY, lw=1.0, label="Observed"); ax.plot(t, sim, color=BLUE, lw=1.0, label="1K-RRI")
    ax.set_title(f"{GEN[g]} (NSE {V['gauges'][g]['nse']:.2f})", fontsize=8, loc="left")
    ax.set_xlabel("Hours from 11 Oct 2019")
axs[0].set_ylabel("Discharge [m³/s]"); axs[0].legend(fontsize=7, loc="upper right")
save(fig, "v6_hydrographs")

# ---- damage-frequency curve and annual expected damage
ME = json.load(open(HERE / "measure_effects_v6.json", encoding="utf-8"))
F6 = R6["config"]["forecast"]["frequency"] if R6 else None
if F6:
    from rri_sim.roadmap.physics import SurrogatePhysics
    ph = SurrogatePhysics("abukuma_explore_v6", log=lambda *a: None)
    T, rT = np.array(F6["T"], float), np.array(F6["ratio"], float)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for rf, slr, lab, c in ((1.0, 0.0, "Current climate", BLUE), (1.1, 0.38, "2 °C (×1.1, +0.38 m)", ORANGE), (1.2, 0.70, "4 °C (×1.2, +0.70 m)", RED)):
        y = ph.predict(np.zeros((len(T), len(ph.layout))), {"rain_factor": rf * rT, "sea_level_rise_m": np.full(len(T), slr), "rain_gamma": np.ones(len(T))})["damage"]
        from rri_sim import frequency as FQ
        ax.plot(1 / T, y / 1e3, "o-", color=c, ms=3, lw=1.2, label=f"{lab}: EAD {FQ.ead(y, T):.0f}")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Annual exceedance probability"); ax.set_ylabel("Event damage [10¹¹ JPY]")
    ax.legend(fontsize=6.5, loc="upper left", title="No measures; EAD [100 M JPY/yr]", title_fontsize=6.5)
    save(fig, "v6_damage_frequency")


# ---- fronts per initial action (v6)
def fronts(R, ax, highlight=(), lw=1.0, alpha=1.0, color=None, label_all=False):
    cmap = matplotlib.colormaps["viridis"]
    ic = np.array([r["initial_cost"] for r in R["table"]])
    for r in R["table"]:
        f = r["front"]
        c = color or cmap(r["initial_cost"] / max(ic.max(), 1))
        ax.plot([x["mean_cost"] for x in f], [100 * x["coverage"] for x in f], "-o", color=c, ms=2.2, lw=lw if r["init"] not in highlight else 2.2,
                alpha=alpha, zorder=3 if r["init"] in highlight else 2)
        if r["init"] in highlight:
            best = max(f, key=lambda x: x["coverage"])
            ax.annotate(f"#{r['init']}", (best["mean_cost"], 100 * best["coverage"]), xytext=(3, -8), textcoords="offset points", fontsize=7, color=GRAY)
    return ic


if R6:
    # highlighted initial actions in categorical colours (legend, no in-plot numbers); others thin, by group
    HL = [(0, "No measures", "#52514e"), (7, "Paddy-field dams (up 100%, mid 50%)", "#2a78d6"),
          (10, "Levee down +1 m, paddy-field dams up", "#eb6834"), (1, "Levee down +1 m, paddy-field dams, raising up, storage", "#c0392b")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    hl = {n for n, _, _ in HL}
    for r in R6["table"]:
        if r["init"] in hl:
            continue
        lev = "堤防" in r["label"]
        f = sorted(r["front"], key=lambda x: x["mean_cost"])
        ax.plot([x["mean_cost"] for x in f], [100 * x["coverage"] for x in f], "-", color="#f2b38f" if lev else "#9cc3ec", lw=0.9, zorder=2)
    for n, lab, c in HL:
        r = next(t for t in R6["table"] if t["init"] == n)
        f = sorted(r["front"], key=lambda x: x["mean_cost"])
        ax.plot([x["mean_cost"] for x in f], [100 * x["coverage"] for x in f], "-o", color=c, ms=2.6, lw=2.0, zorder=4,
                label=f"#{n} {lab} (initial cost {r['initial_cost']:.0f})")
    ax.plot([], [], color="#9cc3ec", lw=0.9, label="Other initial actions without a levee")
    ax.plot([], [], color="#f2b38f", lw=0.9, label="Other initial actions with a levee")
    ax.set_xlabel("Mean cumulative cost, PV [100 M JPY]"); ax.set_ylabel("Target Coverage [%]"); ax.set_ylim(0, 102)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=1, handlelength=1.8)
    save(fig, "v6_fronts")

    # time course of the highest-coverage rule of selected initial actions
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    yrs = [y + 10 for y in R6["years"]]
    for i, c in zip((0, 7, 10, 1), (GRAY, GREEN, BLUE, ORANGE)):
        r = R6["table"][i]; f = max(r["front"], key=lambda x: (x["coverage"], -x["mean_cost"]))
        lab = f"#{i} (initial {r['initial_cost']:.0f})"
        axs[0].plot(yrs, [100 * v for v in f["met_so_far"]], "-o", ms=2.5, color=c, label=lab)
        axs[1].plot(yrs, f["cost_path"], "-o", ms=2.5, color=c, label=lab)
    axs[0].set_ylabel("Scenarios meeting the Target so far [%]"); axs[0].set_ylim(0, 102)
    axs[1].set_ylabel("Mean cumulative cost, PV [100 M JPY]"); axs[1].legend(fontsize=7)
    for a in axs:
        a.set_xlabel("Year")
    save(fig, "v6_time_course")

    # actions by decade (highest-coverage rule of #7 and #10)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for ax, i in zip(axs, (7, 10)):
        r = R6["table"][i]; f = max(r["front"], key=lambda x: (x["coverage"], -x["mean_cost"]))
        cnt = {}
        for evs in f["events"]:
            for e in evs:
                if e["kind"] == "act":
                    k = en_var(R6["variables"][e["var"]]["name"])
                    cnt.setdefault(k, {}); cnt[k][e["year"]] = cnt[k].get(e["year"], 0) + 1
        names = sorted(cnt, key=lambda k: -sum(cnt[k].values()))[:8]
        M = np.array([[cnt[n].get(y, 0) for y in R6["years"]] for n in names])
        ax.imshow(M, cmap="Blues", aspect="auto", vmin=0)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xticks(range(len(R6["years"]))); ax.set_xticklabels(R6["years"], fontsize=6.5, rotation=45)
        ax.grid(False)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if M[a, b]:
                    ax.text(b, a, M[a, b], ha="center", va="center", fontsize=6, color="white" if M[a, b] > M.max() / 2 else GRAY)
        ax.set_title(f"#{i}: {f['policy']['rho']:.1f} × Target, look-ahead {f['policy']['lookahead']} yr ({100 * f['coverage']:.0f}%)", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "v6_actions")

# ---- benefit valuation: representative event (v5) vs frequency-based (v6)
if R5 and R6:
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for R, c, lab in ((R5, MUTED := "#a3a29b", "Benefit = 1/100 event × 0.01 (v5)"), (R6, BLUE, "Benefit = annual expected damage (v6)")):
        pts = np.array([[f["mean_cost"], 100 * f["coverage"]] for r in R["table"] for f in r["front"]])
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=c, label=lab, alpha=0.8)
    ax.set_xlabel("Mean cumulative cost, PV [100 M JPY]"); ax.set_ylabel("Target Coverage [%]"); ax.set_ylim(0, 102); ax.legend(fontsize=7, loc="lower right")
    save(fig, "v6_valuation")

# ---- sensitivity: planning prior and target
CASES = [("abukuma_dapp_forecast_v6_prior_low", "Prior ×1.05, 0.29 m (mild)", GREEN), ("abukuma_dapp_forecast_v6", "Prior ×1.10, 0.38 m (base)", BLUE),
         ("abukuma_dapp_forecast_v6_prior_high", "Prior ×1.25, 0.62 m (severe)", RED)]
TCASES = [("abukuma_dapp_forecast_v6_t10", "Damage Target 1.0 ×", RED), ("abukuma_dapp_forecast_v6", "1.2 × (base)", BLUE),
          ("abukuma_dapp_forecast_v6_t14", "1.4 ×", GREEN)]


def envelope(R):
    pts = sorted([(f["mean_cost"], f["coverage"]) for r in R["table"] for f in r["front"]])
    env, best = [], -1
    for c, v in pts:
        if v > best:
            env.append((c, v)); best = v
    return np.array(env)


sens = {}
if all(load(n) for n, *_ in CASES + TCASES):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, cases, title in ((axs[0], CASES, "(a) Planning assumption (prior of the forecast)"), (axs[1], TCASES, "(b) Target ambition")):
        for n, lab, c in cases:
            R = load(n); e = envelope(R)
            ax.step(e[:, 0], 100 * e[:, 1], where="post", color=c, lw=1.4, label=lab)
            sens[n] = R
        ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("Mean cumulative cost, PV [100 M JPY]"); ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("Best Target Coverage [%]"); axs[0].set_ylim(0, 102)
    save(fig, "v6_sensitivity")


# ---- tables
def tex_initial_actions(R):
    T = R["table"]; Vv = R["variables"]
    used = [i for i in range(len(Vv)) if any(r["levels"][i] for r in T)]

    def cell(i, li):
        if not li:
            return ""
        v = Vv[i]
        if v["type"] == "relocate":
            return "R" if li == 1 else "M"
        if v["type"] == "levee":
            return v["options"][li].replace(" m", "")
        if v["continuous"]:
            return f"{100 * R['levels'][i][li]:.0f}"
        return r"\checkmark"
    lines = [r"\begin{table}[tbp]", r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
             r"\caption{Composition of the 15 candidate initial actions (Pareto-efficient under the planner's nominal values; run \texttt{abukuma\_dapp\_forecast\_v6}). "
             r"Levee: raised height [m]; paddy-field dam, forest management, household storage: implementation rate [\%]; housing: R = raising +1 m, M = relocation; \checkmark: built}",
             r"\label{tab:v6inits}", r"\begin{tabular}{@{}l" + "c" * len(T) + "@{}}", r"\toprule",
             "Initial action & " + " & ".join(f"\\#{r['init']}" for r in T) + r" \\",
             "Initial cost$^{\\dagger}$ & " + " & ".join(f"{r['initial_cost']:.0f}" for r in T) + r" \\",
             "EAD, PV$^{\\dagger}$ (nominal) & " + " & ".join(f"{r['nominal']['ead_pv'] / 1e3:.1f}k" for r in T) + r" \\", r"\midrule"]
    for i in used:
        lines.append(en_var(Vv[i]["name"]) + " & " + " & ".join(cell(i, r["levels"][i]) for r in T) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\footnotesize\raggedright $^{\dagger}$100 M JPY. up/mid/down: upstream/midstream/downstream zone.", r"\end{table}"]
    (HERE / "tab_v6_initial_actions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_measures():
    rows = ME["rows"]
    lines = [r"\begin{table}[tbp]", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Single-measure effects (v6 surrogate, other measures absent; no effect coefficients). Costs [100 M JPY] from \texttt{costs.py}; "
             r"change in the representative-event damage [100 M JPY] and the flooded population, and in the annual expected damage (EAD) [100 M JPY/yr]}",
             r"\label{tab:v6measures}", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Measure & Level & Initial & Annual & $\Delta D$ current & $\Delta D$ 4\,\textdegree C & $\Delta$pop. current & $\Delta$EAD current & $\Delta$EAD 4\,\textdegree C \\", r"\midrule"]
    for r in rows:
        lev = (r["level"].replace("阿武隈川上流遊水地群（350 ha, 1500 万m³）", "350 ha, 15 Mm\\textsuperscript{3}").replace("嵩上げ +1.0 m", "raising +1 m")
               .replace("移転", "relocation").replace("%", r"\%"))
        lines.append(f"{en_var(r['name'])} & {lev} & {r['initial']:.0f} & {r['annual']:.1f} & {r['d_cur']:.0f} & {r['d_4c']:.0f} & {r['p_cur']:.0f} & {r['ead_cur']:.0f} & {r['ead_4c']:.0f} \\\\")
    b = ME["base"]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\footnotesize\raggedright Without measures: damage " + f"{b['current'][0]:.0f} (current) / {b['4C'][0]:.0f} (4\\,\\textdegree C, rain ×1.2, sea level +0.70 m), "
              f"flooded population {b['current'][1]:.0f}, EAD {b['current_ead']:.0f} / {b['4C_ead']:.0f}. "
              r"Positive $\Delta$EAD of some measures (retention basins, midstream levee, public storage) is an error of the surrogate for small floods (Section~\ref{sec:limits}).",
              r"\end{table}"]
    (HERE / "tab_v6_measures.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if R6:
    tex_initial_actions(R6)
tex_measures()

# ---- numbers for the text
summary = {}
for n in ["abukuma_dapp_forecast_v6", "abukuma_dapp_forecast_v5"] + [c[0] for c in CASES + TCASES]:
    R = load(n)
    if not R:
        continue
    summary[n] = [{"init": r["init"], "initial_cost": round(r["initial_cost"]), "label": r["label"],
                   "max_cov": round(max(f["coverage"] for f in r["front"]), 3),
                   "cost_at_max": round(min(f["mean_cost"] for f in r["front"] if f["coverage"] >= max(g["coverage"] for g in r["front"]) - 1e-9)),
                   "min_cost": round(min(f["mean_cost"] for f in r["front"])), "cov_at_min": round(min(r["front"], key=lambda f: f["mean_cost"])["coverage"], 3),
                   "hv": round(r["convergence"]["hv_ratio_min"], 3)} for r in R["table"]]
(HERE / "v6_numbers.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", sorted(summary))
