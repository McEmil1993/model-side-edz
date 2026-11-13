# Downloaded starter dataset — 5 classes × 50 photos

Acquired and screened on November 12, 2026. **250 selected photographs**, approximately **102.6 MiB**. No generated images, augmentations or extra transformed copies are counted as independent photos.

| Folder/class ID | Local name | Exact source label | Count |
| --- | --- | --- | --- |
| `data/raw/bangus/` | Bangus | Chanos chanos | 50 |
| `data/raw/tilapia/` | Nile tilapia | Oreochromis niloticus | 50 |
| `data/raw/hito/` | Hito / African sharptooth catfish | Clarias gariepinus | 50 |
| `data/raw/dalag/` | Dalag / striped snakehead | Channa striata | 50 |
| `data/raw/karpa/` | Karpa / common carp | Cyprinus carpio | 50 |

Market names can cover additional species. These exact labels do not extend classifier support to every fish called hito, tilapia or karpa.

## Where to look

- [dataset.csv](dataset.csv): tracked primary manifest, with paths relative to model-side; source URLs, authors/credits, licenses, checksums and grouping metadata.
- [data/images.csv](data/images.csv): local convenience manifest with paths relative to data.
- [Offline preview gallery](data/review/index.html): all 250 selected photos with credits and individual source/license links.
- [Audit report](data/provenance/accepted-dataset-audit.json): counts, decode/checksum results, duplicate screening and limitations.
- [Selected observation IDs](config/selected-observations.json): explicit screened iNaturalist photo/observation selection.

The local data directory, including source archives and screening candidates, is intentionally ignored by Git. Keep a backup. `dataset.csv`, source configuration and restoration scripts are tracked so the selected photos can be recovered where the original files remain available.

## Sources and attribution

### Bangus and Nile tilapia

[Fish-gres V2, Mendeley Data](https://data.mendeley.com/datasets/76cr3wfhff/2), DOI **10.17632/76cr3wfhff.2**. Creators: **Eko Prasetyo, Nanik Suciati and Chastine Fatichah**. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

These photos were captured in traditional markets in Gresik, East Java, Indonesia. They are **not Philippine-captured market photos**. Fifty source photos per retained species were selected; original TIFFs were converted losslessly to displayable RGB PNGs, one output per source photo, without resizing or augmentation. Archive and original-member checksums are recorded.

Requested related citation: Prasetyo, E., Suciati, N., & Fatichah, C. (2021). Multi-level residual network VGGNet for fish species classification. [Article DOI](https://doi.org/10.1016/j.jksuci.2021.05.015).

### Hito, dalag and karpa

Licensed photographs associated with exact-species **research-grade iNaturalist observations**, retrieved through the [official API](https://api.inaturalist.org/v1/docs/). Each selected image has its own observation/photo ID, attribution and license in the manifest. One selected photo per observation; same observer/date/class records are grouped conservatively.

Photo bytes were retained without editing. Sources are from multiple countries and include catches, handling, market/tank and clear live-fish views. This is **not a market-background-only dataset**. Photo licenses were checked individually, independently of observation-data licensing. Preserve the credited photographers and the relevant license when sharing copies.

Mixed selected licenses: **213 CC BY 4.0**, **25 CC BY-SA 4.0**, **12 CC0 1.0**. Do not relabel the whole collection as CC0. Follow per-image terms, including applicable ShareAlike requirements for adaptations.

## Philippine market relevance

Names/species were checked against Philippine references, not inferred from Indonesian or English common names alone:

- Bangus and Nile tilapia: [BFAR NCR market report](https://www.bfar.da.gov.ph/wp-content/uploads/2025/05/April-Monthly-NCR-RCPMR-for-2025-Retail.pdf).
- Hito / Clarias gariepinus: [BFAR national price monitoring](https://www.bfar.da.gov.ph/wp-content/uploads/2025/04/Revised_NCPMR-for-the-of-March-2025.pdf).
- Dalag / Channa striata: [DOST-FNRI food record](https://i.fnri.dost.gov.ph/fct/library/report/4061) and [DA-NFRDI species discussion](https://nfrdi.da.gov.ph/2026/03/02/feature-unlocking-aquaculture-potential-of-mudfish-in-muslim-region/).
- Karpa / Cyprinus carpio: [BFAR market commodity record](https://www.bfar.da.gov.ph/wp-content/uploads/2023/04/2023-04-5thWeekRetailMarch27-31.pdf).

## Screening and exclusions

All selected files decode successfully and match their recorded SHA-256. No exact byte or decoded-pixel duplicates were found; the current combined difference/perceptual-hash screen flagged no near-duplicate pairs. This does not prove every image shows a different specimen.

Contact sheets were visually inspected. Clearly unsuitable candidates, including bird/predator-dominated photos, skeletons, isolated body parts, tiny/obscured fish and visibly questionable labels, were not intentionally selected. Some retained photos contain several fish of the same source-labeled species; some clear live-fish photos are underwater. Source quality grades and visual screening are not expert taxonomic certification.

Three initially considered Fish-gres classes were excluded after morphology/label concerns: the archives labeled Rastrelliger faughni, Upeneus moluccensis and Eleutheronema tetradactylum. Their derived samples are under `data/quarantine/fishgres-unverified/`, are absent from the final manifest, and must not be trained or silently relabeled. This records a screening concern, not a definitive taxonomic correction to the published dataset.

## Training limits

Fifty photos per class are a starting collection, not evidence of model accuracy. Groups use source filename capture dates for Fish-gres and observer/date for iNaturalist. Exact independent specimen/session identity is not certified; do not random-split repeated specimens across train and test.

The acquisition task initially left `config/species.json` unreviewed and did not train a model. The subsequent explicit training task reviewed all accepted contact sheets and conservatively regrouped repeated Fish-gres setups and iNaturalist observers across dates. The registry now records source/visual review for experimental training, not expert certification. All five IDs are in the bundled app catalog. The real trained model is experimental and failed numeric quality gates: see [model card](reports/0.1.0/MODEL_CARD.md). Independent Philippine-market captures and expert review remain necessary.

Acquisition groups in dataset.csv preserve original metadata. Training uses data/training-reviewed.csv and data/splits.json; portable [split assignments](reports/0.1.0/split-assignment.csv) retain the audited partitions. Four coarse average-hash collisions were individually checked as visibly different taxa/scenes and recorded as checksum-bound reviews, without disabling other duplicate checks.

An additional 72 licensed iNaturalist photos were downloaded only for rejection evaluation: unsupported Lates calcarifer and non-fish Passer domesticus, 32 validation and 40 test photos partitioned by observer. They are not part of the 250-photo training collection or additional model classes. Individual source/author/license records are retained in [negative provenance](reports/0.1.0/negative-image-provenance.json). The narrow negative set does not establish general unknown-image robustness.

## Restore the selected photos

From model-side, with Pillow/NumPy and UnRAR or bsdtar available:

```powershell
python tools/restore_selected_dataset.py
```

This uses the frozen manifest rather than new search results. It refuses checksum mismatches instead of silently replacing missing photos. Source availability and unchanged rendering are required; retain the downloaded local copy and archive backups.
