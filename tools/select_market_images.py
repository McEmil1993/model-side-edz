"""Select 50 original, distinct source photos per class and build provenance/review artifacts.

Requires UnRAR (or bsdtar) on PATH. It reads individual members to stdout rather
than extracting arbitrary archive paths into the workspace. No augmentation is used.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from download_market_dataset import ROOT, CONFIG, checksum

DATA = ROOT / "data"
FONT_PATH = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"


def hashes(image):
    gray = ImageOps.exif_transpose(image).convert("L")
    small = np.asarray(gray.resize((9, 8)), dtype=np.float32)
    dhash = sum(int(value) << index for index, value in enumerate((small[:, 1:] > small[:, :-1]).flat))
    pixels = np.asarray(gray.resize((32, 32)), dtype=np.float64)
    indices = np.arange(32)
    cosine = np.cos((2 * indices[:, None] + 1) * indices[None, :] * np.pi / 64)
    coefficients = (cosine.T @ pixels @ cosine)[:8, :8].flatten()[1:]
    median = np.median(coefficients)
    phash = sum(int(value) << index for index, value in enumerate(coefficients > median))
    return dhash, phash


def tool():
    unrar = shutil.which("unrar") or shutil.which("UnRAR.exe")
    if not unrar:
        candidate = Path("C:/Program Files/WinRAR/UnRAR.exe")
        if candidate.exists():
            unrar = str(candidate)
    if unrar:
        return unrar, True
    tar = shutil.which("tar")
    if tar:
        return tar, False
    raise RuntimeError("Install UnRAR or bsdtar to read the licensed source archives.")


def members(executable, is_unrar, archive):
    command = [executable, "lb", "-c-", str(archive)] if is_unrar else [executable, "-tf", str(archive)]
    result = subprocess.run(command, capture_output=True, check=True, timeout=90, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    names = result.stdout.decode("utf-8", errors="replace").splitlines()
    valid = []
    for name in names:
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or ":" in normalized:
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            valid.append(name)
    return valid


def read_member(executable, is_unrar, archive, name):
    command = [executable, "p", "-inul", "-p-", str(archive), name] if is_unrar else [executable, "-xOf", str(archive), name]
    result = subprocess.run(command, capture_output=True, check=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not result.stdout or len(result.stdout) > 25 * 1024 * 1024:
        raise ValueError("Unexpected image member size")
    return result.stdout


def contact_sheets(records, label):
    output = DATA / "review"
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT_PATH), 15) if FONT_PATH.exists() else ImageFont.load_default()
    heading = ImageFont.truetype(str(FONT_PATH), 24) if FONT_PATH.exists() else font
    for start in range(0, len(records), 25):
        page = records[start:start + 25]
        sheet = Image.new("RGB", (1250, 1170), "#eef5f6")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 14), f"{label} | source labels | {start+1}-{start+len(page)}", fill="#102a43", font=heading)
        for i, record in enumerate(page):
            x, y = (i % 5) * 250, 60 + (i // 5) * 220
            draw.rectangle((x + 5, y, x + 245, y + 212), fill="white")
            with Image.open(DATA / record["path"]) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (236, 182))
                sheet.paste(thumb, (x + (250 - thumb.width) // 2, y + (184 - thumb.height) // 2))
            draw.text((x + 12, y + 187), Path(record["path"]).name, fill="#102a43", font=font)
        sheet.save(output / f"{label}-{start//25+1}.jpg", quality=92)


def select(per_class=50):
    source = json.loads(CONFIG.read_text(encoding="utf-8"))
    executable, is_unrar = tool()
    exclusions_file = ROOT / "config/download-exclusions.json"
    exclusions = json.loads(exclusions_file.read_text()) if exclusions_file.exists() else {}
    records, report, used = [], [], set()
    for species in source["classes"]:
        if not species.get('eligible',True):
            continue
        archive = DATA / "source-archives" / species["archive"]
        if not archive.exists() or checksum(archive) != species["sha256"]:
            raise ValueError(f"Missing/unverified source archive: {archive}")
        candidates = members(executable, is_unrar, archive)
        random.Random(20261112).shuffle(candidates)
        accepted, fingerprints, rejected, pixel_hashes = [], [], [], set()
        output = DATA / "raw" / species["id"]
        output.mkdir(parents=True, exist_ok=True)
        for member in candidates:
            if member in exclusions.get(species["id"], []):
                continue
            raw = read_member(executable, is_unrar, archive, member)
            sha = hashlib.sha256(raw).hexdigest()
            if sha in used:
                rejected.append({"member": member, "reason": "exact_duplicate"})
                continue
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    if min(width, height) < 128:
                        raise ValueError("undersized")
                    dhash, phash = hashes(image)
                    rgb=ImageOps.exif_transpose(image).convert('RGB')
                    pixel_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
                    if pixel_hash in pixel_hashes:
                        rejected.append({'member':member,'reason':'identical_decoded_pixels'})
                        continue
                    if any((dhash ^ d).bit_count() <= 3 and (phash ^ p).bit_count() <= 5 for d, p in fingerprints):
                        rejected.append({"member": member, "reason": "near_duplicate_candidate"})
                        continue
            except (OSError, ValueError) as error:
                rejected.append({"member": member, "reason": str(error)})
                continue
            path = output / f"{species['id']}-{len(accepted)+1:03}.png"
            # One PNG per original source photo; lossless RGB conversion from TIFF.
            # No augmentation or extra transformed copies are counted as new samples.
            rgb.save(path,format='PNG')
            used.add(sha)
            pixel_hashes.add(pixel_hash)
            fingerprints.append((dhash, phash))
            timestamp=re.search(r'IMG_(\d{8})_(\d{6})',member)
            capture_date=timestamp.group(1) if timestamp else 'unknown-session'
            row = {"path": path.relative_to(DATA).as_posix(), "label": species["id"],
                   "group": f"fishgres-v2-{species['id']}-{capture_date}",
                   "source": "Prasetyo, Suciati & Fatichah (2021), Fish-gres V2, doi:10.17632/76cr3wfhff.2",
                   "license": source["license"], "scientific_name": species["scientificName"],
                   "local_name": species["localName"], "original_member": member,
                   "source_url": f"{source['page']}/files/{species['fileId']}",
                   "license_url": source["licenseUrl"], "sha256": checksum(path), "original_sha256":sha, "pixel_sha256":pixel_hash,"width": width, "height": height,
                   "group_basis":"inferred filename capture date; not certified specimen identity",
                   "specimen_id_known": "false", "label_status": "research-source-labeled; visual screening required"}
            accepted.append(row)
            if len(accepted) == per_class:
                break
        if len(accepted) != per_class:
            raise ValueError(f"{species['id']}: only {len(accepted)} suitable distinct images; do not pad with duplicates")
        contact_sheets(accepted, species["id"])
        records.extend(accepted)
        report.append({"id": species["id"], "accepted": len(accepted), "candidates": len(candidates), "rejected": rejected, "archiveSha256": species["sha256"]})
        print(f"Selected {len(accepted)} verified image files: {species['id']}; rejected {len(rejected)} duplicate/invalid candidates", flush=True)
    with (DATA / "images.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (DATA / 'provenance/fishgres-selected.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (DATA / "provenance/selection-report.json").parent.mkdir(parents=True, exist_ok=True)
    (DATA / "provenance/selection-report.json").write_text(json.dumps({"seed": 20261112, "source": source, "classes": report, "count": len(records), "specimenGroupsKnown": False}, indent=2), encoding="utf-8")
    html = ['<!doctype html><meta charset="utf-8"><title>Fish2Dish dataset review</title><style>body{font-family:Arial;margin:30px;background:#eef5f6;color:#102a43}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}figure{background:white;margin:0;padding:8px}img{width:100%;height:180px;object-fit:contain}figcaption{font-size:12px;padding:8px}h1,h2{margin-top:32px}</style><h1>Fish2Dish — 250 source-labeled photos</h1><p>CC BY 4.0 · Fish-gres V2 · Photos captured in Indonesia. Dataset author labels; not expert certification. Independent specimens/sessions are unknown.</p>']
    for species in source["classes"]:
        html.append(f"<h2>{species['localName']} — {species['scientificName']}</h2><section>")
        for row in records:
            if row["label"] == species["id"]:
                html.append(f'<figure><img loading="lazy" src="../{row["path"]}" alt="{row["scientific_name"]}"><figcaption>{Path(row["path"]).name}</figcaption></figure>')
        html.append("</section>")
    (DATA / "review/index.html").write_text("\n".join(html), encoding="utf-8")
    return records


if __name__ == "__main__":
    select()
