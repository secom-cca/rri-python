"""Browser check of the climate controls (Playwright, installed Chrome).  Server must run."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/gui_climate")
OUT.mkdir(parents=True, exist_ok=True)


def wait(page, text, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = page.inner_text("#pmsg")
        if text in m:
            return m
        if "エラー" in m:
            raise RuntimeError(m)
        time.sleep(1)
    raise TimeoutError(page.inner_text("#pmsg"))


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = b.new_page(viewport={"width": 1600, "height": 950})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.goto("http://127.0.0.1:8765/")
    print(wait(page, "現況の計算完了"))
    page.select_option("#climate", "4C")
    time.sleep(1)
    print("climate info:", page.inner_text("#climate-info"))
    print(wait(page, "現況の計算完了"))
    page.screenshot(path=str(OUT / "01_4C_baseline.png"))
    print("impact:", page.inner_text("#climate-impact").replace("\n", " | ")[:700])
    # a measure under 4C
    page.click("#tools button[data-tool=levee]")
    box = page.locator("#view canvas[data-engine]").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate("() => { const b = document.getElementById('brush'); b.value = 2; b.dispatchEvent(new Event('input')); }")
    page.mouse.move(cx - 300, cy - 150); page.mouse.down(); page.mouse.move(cx, cy, steps=10); page.mouse.move(cx + 300, cy + 100, steps=10); page.mouse.up()
    page.click("#run")
    print(wait(page, "完了: 被害額"))
    page.click("#tabs button[data-tab=flow]")
    page.screenshot(path=str(OUT / "02_4C_flow.png"))
    print("JS errors:", errs or "none")
    b.close()
