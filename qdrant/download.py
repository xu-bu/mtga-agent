"""
Step 1: Download Scryfall bulk data (oracle_cards — one entry per unique card).
Run this once. Output: data/oracle_cards.json (~100MB)
"""

import json
import os
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "oracle_cards.json")


def get_bulk_download_url() -> str:
    """Fetch the latest oracle_cards bulk file URL from Scryfall."""
    print("Fetching bulk data index from Scryfall...")
    resp = requests.get("https://api.scryfall.com/bulk-data", timeout=30)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "oracle_cards":
            return entry["download_uri"]
    raise RuntimeError("Could not find oracle_cards bulk entry")


def download(url: str, dest: str):
    print(f"Downloading {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="oracle_cards.json"
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Saved to {dest}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} already exists — delete it to re-download.")
    else:
        url = get_bulk_download_url()
        download(url, OUTPUT_FILE)