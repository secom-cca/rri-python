"""End-to-end GUI check with Playwright (uses the installed Google Chrome).

Requires the server running:  python -m rri_sim.gui --no-browser
python scripts/gui_smoke_test.py [outdir]
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/gui")
BASIN = sys.argv[2] if len(sys.argv) > 2 else None
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8765/"


def wait_progress(page, text, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = page.inner_text("#pmsg")
        if text in msg:
            return msg
        if "エラー" in msg:
            raise RuntimeError(msg)
        time.sleep(1)
    raise TimeoutError(page.inner_text("#pmsg"))


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(URL)
    if BASIN:
        wait_progress(page, "現況の計算完了")
        page.select_option("#basin", BASIN)
        time.sleep(2)
    print("baseline:", wait_progress(page, "現況の計算完了", timeout=3600))
    page.screenshot(path=str(OUT / "01_baseline_flood.png"))

    # paint measures: paddy dam over the centre, levee along the river, raise houses
    canvas = page.locator("#view canvas[data-engine]")
    box = canvas.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def paint(tool, pts, radius="3"):
        page.click(f"#tools button[data-tool={tool}], #ops button[data-tool={tool}]")
        page.fill("#brush", radius) if False else page.evaluate(
            "r => { const b = document.getElementById('brush'); b.value = r; b.dispatchEvent(new Event('input')); }", radius)
        page.mouse.move(*pts[0])
        page.mouse.down()
        for q in pts[1:]:
            page.mouse.move(*q, steps=8)
        page.mouse.up()

    paint("paddy_dam", [(cx - 200, cy - 80), (cx + 50, cy - 40), (cx + 200, cy + 20)], "6")
    paint("forest", [(cx - 60, cy + 40), (cx + 60, cy + 90)], "4")
    paint("retention", [(cx - 20, cy - 20), (cx + 10, cy - 10)], "1")
    paint("levee", [(cx - 300, cy - 150), (cx, cy), (cx + 300, cy + 100)], "2")
    paint("raise", [(cx + 100, cy - 60), (cx + 160, cy - 30)], "2")
    page.click("#modes button[data-mode=landuse]")
    page.screenshot(path=str(OUT / "02_painted_measures.png"))
    print("edits:", page.inner_text("#edits"))

    paint("relocate", [(cx + 100, cy - 60), (cx + 140, cy - 40)], "1")
    page.click("#run")
    print("scenario:", wait_progress(page, "完了: 被害額", timeout=3600))
    page.screenshot(path=str(OUT / "06_costs.png"))
    page.click("#tabs button[data-tab=indic]")
    page.screenshot(path=str(OUT / "07_indicators.png"))
    print("indicators:", page.inner_text("#liv")[:200].replace("\n", " | "), "||", page.inner_text("#eco")[:160].replace("\n", " | "))
    page.click("#tabs button[data-tab=flow]")
    page.screenshot(path=str(OUT / "08_flow.png"))
    page.click("#tabs button[data-tab=damage]")
    print("costs:", page.inner_text("#costs")[:300].replace("\n", " | "))
    page.click("#modes button[data-mode=diff]")
    page.screenshot(path=str(OUT / "03_diff.png"))
    page.click("#modes button[data-mode=damage]")
    page.screenshot(path=str(OUT / "04_damage.png"))
    page.click("#modes button[data-mode=flood]")
    page.evaluate("() => { const t = document.getElementById('time'); t.value = 30; t.dispatchEvent(new Event('input')); }")
    page.screenshot(path=str(OUT / "05_flood_frame30.png"))
    print("KPIs:", page.inner_text("#kpis").replace("\n", " | "))
    print("JS errors:", errors or "none")
    browser.close()
