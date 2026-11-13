"""Auditable grouped splits. No automatic scraping or assumed folder labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def prepare(csv_path: Path, registry_path: Path, output: Path, seed: int = 42) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("reviewed") is not True:
        raise ValueError("Review taxonomy and labeled examples in species.json before preparing training data.")
    labels = sorted(c["id"] for c in registry["classes"])
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("At least two unique reviewed classes required.")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen_hashes: set[str] = set()
    seen_visual: list[tuple[int, str, str]] = []
    reviewed_pairs = {frozenset(item['sha256']) for item in registry.get('distinctImageReviews', []) if item.get('reason') and len(item['sha256']) == 2}
    group_labels: dict[str, str] = {}
    for row in rows:
        if not all(row.get(key, "").strip() for key in ("path", "label", "group", "source", "license")):
            raise ValueError("Every image needs path, label, independent group, source and license.")
        if row["label"] not in labels:
            raise ValueError(f"Unreviewed class: {row['label']}")
        path = (csv_path.parent / row["path"]).resolve()
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum in seen_hashes:
            raise ValueError(f"Exact duplicate image: {path}")
        seen_hashes.add(checksum)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 64:
                raise ValueError(f"Image too small: {path}")
            # Cheap review aid, not a complete semantic near-duplicate detector.
            values = list(image.convert("L").resize((8, 8)).getdata())
            mean = sum(values) / 64
            ahash = sum((v >= mean) << i for i, v in enumerate(values))
            if any((ahash ^ old).bit_count() <= 2 and group != row["group"] and frozenset([checksum, old_checksum]) not in reviewed_pairs for old, group, old_checksum in seen_visual):
                raise ValueError(f"Potential near duplicate across groups: {path}. Review/group captures before splitting.")
            seen_visual.append((ahash, row["group"], checksum))
        if row["group"] in group_labels and group_labels[row["group"]] != row["label"]:
            raise ValueError("A capture group must not be split across labels. Review the group definition.")
        group_labels[row["group"]] = row["label"]
        grouped[row["label"]][row["group"]].append({**row, "path": str(path), "sha256": checksum})
    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label in labels:
        groups = list(grouped[label])
        if len(groups) < 3:
            raise ValueError(f"{label}: at least three independent groups required; substantially more are needed for reliable evaluation.")
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * .15))
        n_val = max(1, round(len(groups) * .15))
        for index, group in enumerate(groups):
            split = "test" if index < n_test else "validation" if index < n_test + n_val else "train"
            splits[split].extend(grouped[label][group])
    result = {"schemaVersion": 1, "seed": seed, "labels": labels, "splits": splits}
    version = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result["datasetVersion"] = version
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("config/species.json"))
    parser.add_argument("--output", type=Path, default=Path("data/splits.json"))
    args = parser.parse_args()
    result = prepare(args.csv, args.registry, args.output)
    print(json.dumps({"datasetVersion": result["datasetVersion"], "counts": {k: len(v) for k, v in result["splits"].items()}}))
