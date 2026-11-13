"""Package a measured local model and preserve its evidence; never publish online."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def package(run, experimental=False):
    manifest=json.loads((run/'manifest.json').read_text())
    report=json.loads((run/'export-report.json').read_text())
    if not report['valid'] or hashlib.sha256((run/'model.tflite').read_bytes()).hexdigest()!=report['sha256']:
        raise ValueError('Missing validation or artifact changed')
    if not manifest['evaluation'].get('numericGatesPassed') and not experimental:
        raise ValueError('Quality gates failed: use --experimental for a clearly labeled development bundle only')
    catalog=json.loads((ROOT.parent/'app-side/assets/data/catalog.json').read_text())
    if not set(manifest['labels']).issubset({f['id'] for f in catalog['fish']}):
        raise ValueError('Flutter catalog is missing model class IDs')
    target=ROOT.parent/'app-side/assets/models';target.mkdir(parents=True,exist_ok=True)
    for name in ['model.tflite','manifest.json']:
        shutil.copyfile(run/name,target/name)
    (target/'labels.txt').write_text('\n'.join(manifest['labels'])+'\n')
    evidence=ROOT/'reports'/manifest['version'];evidence.mkdir(parents=True,exist_ok=True)
    for name in ['training.json','selection.json','calibration.json','evaluation.json','export-report.json','tflite-measurement.json']:
        if (run/name).exists():shutil.copyfile(run/name,evidence/name)
    splits=json.loads((ROOT/'data/splits.json').read_text())
    with (evidence/'split-assignment.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['split','label','file','sha256','group','group_basis']);writer.writeheader()
        for part,rows in splits['splits'].items():
            for row in rows:writer.writerow(dict(split=part,label=row['label'],file=Path(row['path']).name,sha256=row['sha256'],group=row['group'],group_basis=row['group_basis']))
    # Keep individually licensed photo credits with the artifact without bundling images.
    with (ROOT/'dataset.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    notice=['# Training image credits','', 'Photos are not included in this app bundle. These credits identify the sources used for this experimental model. No new blanket license is assigned to the trained weights. See model-side/DATASET_SOURCES.md for collection and reuse limits.','', 'Fish-gres V2: Eko Prasetyo, Nanik Suciati, Chastine Fatichah. DOI 10.17632/76cr3wfhff.2. CC BY 4.0. TIFF originals converted losslessly to RGB PNG; training additionally resized/augmented images.','']
    for row in rows:
        notice.append(f"- {Path(row['path']).name}: {row.get('attribution') or row.get('author') or 'Fish-gres V2 creators'}. {row['license']}. Source: {row.get('source_url') or row['source']}; license: {row.get('license_url','')}")
    (target/'TRAINING_DATA_NOTICE.md').write_text('\n'.join(notice)+'\n',encoding='utf-8')
    negatives=json.loads((ROOT/'data/provenance/negative-images.json').read_text(encoding='utf-8'))
    for row in negatives:row['path']=Path(row['path']).name
    (evidence/'negative-image-provenance.json').write_text(json.dumps(negatives,indent=2),encoding='utf-8')
    print(f"Experimental Flutter bundle: {target}; evidence: {evidence}")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/tuned-v0.1.0')
    parser.add_argument('--experimental',action='store_true')
    args=parser.parse_args();package(args.run,args.experimental)
