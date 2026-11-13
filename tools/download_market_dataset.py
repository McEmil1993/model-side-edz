"""Download the explicitly licensed Fish-gres source archives, verifying provider SHA-256.

Downloads are cached for reproducibility; no web-search thumbnails or generated fish are used.
The cache is local dataset data and is excluded from Git.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/download-sources.json"


def checksum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(species, cache):
    destination = cache / species["archive"]
    if destination.exists() and checksum(destination) == species["sha256"]:
        print(f"Cached verified archive: {species['id']}", flush=True)
        return destination
    staging = destination.with_suffix(".rar.part")
    url = f"https://data.mendeley.com/public-files/datasets/76cr3wfhff/files/{species['fileId']}/file_downloaded"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Fish2Dish-dataset-research/0.1"})
            received, last_report = 0, 0
            with urllib.request.urlopen(request, timeout=90) as response, staging.open("wb") as output:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > species["bytes"]:
                        raise ValueError("Download exceeded provider-reported size")
                    output.write(block)
                    if received - last_report >= 30 * 1024 * 1024:
                        print(f"{species['id']}: {received // 1048576}/{species['bytes'] // 1048576} MiB", flush=True)
                        last_report = received
            if received != species["bytes"] or checksum(staging) != species["sha256"]:
                raise ValueError("Archive does not match provider size/SHA-256")
            staging.replace(destination)
            print(f"Verified download: {species['id']}", flush=True)
            return destination
        except Exception as error:
            print(f"Retry {attempt+1}/3 {species['id']}: {type(error).__name__}: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--species", help="Optional single class ID")
    args = parser.parse_args()
    source = json.loads(CONFIG.read_text(encoding="utf-8"))
    cache = ROOT / "data/source-archives"
    cache.mkdir(parents=True, exist_ok=True)
    selected = [c for c in source["classes"] if c.get("eligible", True) and (not args.species or c["id"] == args.species)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(3, args.workers))) as executor:
        list(executor.map(lambda item: download(item, cache), selected))
