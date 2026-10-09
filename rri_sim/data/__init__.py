"""Public-data acquisition (GSI, MLIT, JMA, 国土数値情報).

All downloaders cache raw files under ``data/raw`` and never re-download a file
that already exists.  Downloaded content is parsed as data only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
USER_AGENT = "rri-sim research (python-requests)"


def http_get(url, timeout=60, retries=4, headers=None):
    """GET with retries and exponential backoff; returns bytes."""
    import time
    import requests
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent": "Mozilla/5.0"})
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            time.sleep(2 ** k * 2)
    raise last
