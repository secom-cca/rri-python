"""Launch the GUI:  python -m rri_sim.gui [--port 8765] [--no-browser]"""
import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    url = f"http://{a.host}:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"RRI-python GUI: {url}")
    uvicorn.run("rri_sim.gui.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
