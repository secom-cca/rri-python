"""Table of the candidate initial actions (composition by measure) for the paper outline.

    .venv/bin/python docs/paper/dapp/initial_actions_table.py [run name, default abukuma_dapp_adaptive]
Writes docs/paper/dapp/tab_initial_actions.tex (\\input by dapp_outline.tex).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
run = sys.argv[1] if len(sys.argv) > 1 else "abukuma_dapp_adaptive"
A = json.load(open(ROOT / f"data/processed/dapp/{run}/results.json", encoding="utf-8"))
T = sorted(A["table"], key=lambda r: (r["initial_cost"], r["node"]))

# rows: (Japanese variable name, English label); a measure appears as "<name> <option>" in the labels
ROWS = [("堤防・上流", "Levee, up"), ("堤防・中流", "Levee, mid"), ("堤防・下流", "Levee, down"),
        ("遊水地3・下流", "Retention basin 3, down"),
        ("田んぼダム・上流", "Paddy-field dam, up"), ("田んぼダム・中流", "Paddy-field dam, mid"),
        ("田んぼダム・下流", "Paddy-field dam, down"),
        ("森林化・上流", "Afforestation, up"), ("森林化・中流", "Afforestation, mid"), ("森林化・下流", "Afforestation, down"),
        ("住宅対策・上流", "Housing, up"), ("住宅対策・下流", "Housing, down"),
        ("排水ポンプ（下水道）・上流", "Urban pump, up"), ("排水ポンプ（下水道）・下流", "Urban pump, down"),
        ("排水機場（農業）・上流", "Agricultural pump, up"),
        ("雨水貯留（公共）・上流", "Public storage, up"),
        ("雨水貯留（各戸）・上流", "Household storage, up"), ("雨水貯留（各戸）・下流", "Household storage, down")]


def cell(opt):
    """Option text -> short cell entry."""
    if opt.startswith("嵩上げ"):
        return "R"
    if opt == "移転":
        return "M"
    if opt.startswith("貯留深"):
        return r"\checkmark"
    return opt.replace(" mm/h", "").replace(" mm", "").replace(" m", "").replace("%", "")


comp = []
for r in T:
    parts = {} if r["label"] == "施策なし" else dict(p.split(" ", 1) for p in r["label"].split(" + "))
    comp.append(parts)
used = [x for x in ROWS if any(x[0] in c for c in comp)]
missing = {k for c in comp for k in c} - {x[0] for x in ROWS}
assert not missing, missing

head = " & ".join(("\\textbf{" if r.get("recommended") else "") + ("None" if r["node"] == 0 else f"\\#{r['node']}")
                  + ("}" if r.get("recommended") else "") for r in T)
mark = " & ".join("◎" if r.get("recommended") else ("★" if r.get("pareto") else "") for r in T)
cost = " & ".join(f"{r['initial_cost']:.0f}" for r in T)
lines = [r"\begin{table}[tbp]",
         r"\centering\scriptsize\setlength{\tabcolsep}{2.6pt}",
         r"\caption{Composition of the 16 candidate initial actions (baseline run \texttt{" + run.replace("_", r"\_") + r"}; ordered by initial cost). "
         r"Levee: raised height [m]; paddy-field dam, afforestation, household storage: implementation rate [\%]; housing: R = raising +1 m, M = relocation; "
         r"pumps: added drainage rate [mm/h]; public storage: depth [mm]; \checkmark: built. ★ Pareto, ◎ recommended (Table~\ref{tab:init})}",
         r"\label{tab:initacts}",
         r"\begin{tabular}{@{}l" + "c" * len(T) + "@{}}",
         r"\toprule",
         "Initial action & " + head + r" \\",
         " & " + mark + r" \\",
         "Initial cost$^{\\dagger}$ & " + cost + r" \\",
         r"\midrule"]
for jp, en in used:
    lines.append(en + " & " + " & ".join(cell(c[jp]) if jp in c else "" for c in comp) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\par\footnotesize\raggedright $^{\dagger}$100 M JPY. up/mid/down: upstream/midstream/downstream zone. Retention basin 3 is one of the automatically selected downstream sites of exploration v2 (replaced by the Upper Abukuma retention basins from v3).",
          r"\end{table}"]
out = Path(__file__).parent / "tab_initial_actions.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
