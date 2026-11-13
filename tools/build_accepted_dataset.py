"""Build the 5 × 50 selected dataset after provenance and contact-sheet screening.

Uses explicit observation/photo IDs, not an automatically accepted search result.
Preserves photo credits and separates uncertain Fish-gres classes in quarantine.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from download_market_dataset import ROOT
from select_market_images import contact_sheets, hashes

DATA=ROOT/'data'


def build():
    fishgres_selection=DATA/'provenance/fishgres-selected.csv'
    if not fishgres_selection.exists():fishgres_selection=DATA/'provenance/fishgres-initial-selection.csv'
    with fishgres_selection.open(encoding='utf-8') as source:
        rows=[row for row in csv.DictReader(source) if row['label'] in ['bangus','tilapia']]
    for row in rows:
        row.update(author='Eko Prasetyo; Nanik Suciati; Chastine Fatichah',
                   attribution='Prasetyo, E.; Suciati, N.; Fatichah, C. (2021). Fish-gres V2. doi:10.17632/76cr3wfhff.2. CC BY 4.0.',
                   photo_id='',observation_id='',label_status='research-source label; visually screened; not expert certified')
    candidates=json.loads((DATA/'provenance/inaturalist-candidates.json').read_text(encoding='utf-8'))
    selected=json.loads((ROOT/'config/selected-observations.json').read_text(encoding='utf-8'))
    if set(selected)!={'hito','dalag','karpa'}:
        raise ValueError('All three visually screened observation selections are required.')
    for label,choices in selected.items():
        if len(choices)!=50:
            raise ValueError(f'{label}: exactly 50 selected photos required')
        for index,choice in enumerate(choices,1):
            matches=[r for r in candidates if r['id']==label and r['observationId']==choice['observationId'] and r['photoId']==choice['photoId'] and 'error' not in r]
            if len(matches)!=1:
                raise ValueError(f'Missing or ambiguous licensed source: {choice}')
            item=matches[0]
            if item['qualityGrade']!='research' or item['license'] not in ['CC0 1.0','CC BY 4.0','CC BY-SA 4.0']:
                raise ValueError('Unsupported source grade/license')
            original=DATA/item['path']
            if hashlib.sha256(original.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Cached photo changed')
            with Image.open(original) as image:
                extension='.jpg' if image.format=='JPEG' else '.png' if image.format=='PNG' else None
                if extension is None:
                    raise ValueError('Unsupported display format')
                rgb=ImageOps.exif_transpose(image).convert('RGB')
                pixel_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
            directory=DATA/'raw'/label
            directory.mkdir(parents=True,exist_ok=True)
            destination=directory/f'{label}-{index:03}{extension}'
            shutil.copyfile(original,destination)
            day=item['observedOn'] or f"observation-{item['observationId']}"
            rows.append({'path':destination.relative_to(DATA).as_posix(),'label':label,
                         'group':f"inat-{label}-{item['observer']}-{day}",
                         'source':f"iNaturalist research-grade observation {item['observationId']}, photo {item['photoId']}",
                         'license':item['license'],'scientific_name':item['scientificName'],'local_name':item['localName'],
                         'original_member':item['path'],'source_url':item['sourceUrl'],'license_url':item['licenseUrl'],
                         'sha256':item['sha256'],'original_sha256':item['sha256'],'pixel_sha256':pixel_hash,
                         'width':item['width'],'height':item['height'],'specimen_id_known':'false',
                         'group_basis':'same observer, date and class grouped together; specimen independence not certified',
                         'label_status':'exact research-grade taxon; visually screened; not expert certified',
                         'author':item['observer'],'attribution':item['attribution'],
                         'photo_id':item['photoId'],'observation_id':item['observationId'],'download_url':item['url']})
    counts=Counter(row['label'] for row in rows)
    if len(rows)!=250 or set(counts.values())!={50} or len(counts)!=5:
        raise ValueError(f'Invalid class balance: {counts}')
    seen_sha,seen_pixels=set(),set()
    total_bytes=0
    near_pairs=[]
    fingerprints=[]
    for row in rows:
        path=DATA/row['path']
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError(f'File checksum mismatch: {path}')
        if row['sha256'] in seen_sha or row['pixel_sha256'] in seen_pixels:
            raise ValueError('Duplicate source photos are not counted as separate dataset samples')
        seen_sha.add(row['sha256']);seen_pixels.add(row['pixel_sha256'])
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if min(image.size)<128:
                raise ValueError('Undersized image')
            d,p=hashes(image)
        for previous,od,op in fingerprints:
            if (d^od).bit_count()<=3 and (p^op).bit_count()<=5:
                near_pairs.append([previous,row['path']])
        fingerprints.append((row['path'],d,p))
        total_bytes+=len(raw)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    for target,prefix in [(DATA/'images.csv',''),(ROOT/'dataset.csv','data/')]:
        with target.open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            writer.writerows([{**row,'path':prefix+row['path']} for row in rows])
    for label in counts:
        contact_sheets([r for r in rows if r['label']==label],label)
    audit={'images':len(rows),'counts':dict(counts),'bytes':total_bytes,'decodeAndChecksumChecks':'passed',
           'exactFileAndPixelDuplicates':0,'nearDuplicateCandidates':near_pairs,'syntheticOrAugmentedSamples':0,
           'visualScreening':'selected images inspected in contact sheets; visibly unsuitable candidates excluded',
           'expertTaxonomicCertification':False,'allPhotosCapturedInPhilippines':False,
           'licenses':dict(Counter(row['license'] for row in rows)),
           'groupCounts':{label:len({r['group'] for r in rows if r['label']==label}) for label in counts},
           'excludedFishgresClasses':['alumahang-bato','saramulyete','mamali']}
    (DATA/'provenance/accepted-dataset-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    pages=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fish2Dish · 250 selected fish photos</title>',
           '<style>body{font:15px Arial;margin:30px;background:#eef5f6;color:#102a43}main{max-width:1400px;margin:auto}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}figure{background:white;margin:0;padding:8px;border-radius:10px}img{width:100%;height:190px;object-fit:contain}figcaption{font-size:11px;line-height:1.6;padding:8px}h1,h2{margin-top:32px}p{line-height:1.7}a{color:#005bbb}.note{padding:18px;background:white;border-radius:10px}nav{display:flex;gap:20px;flex-wrap:wrap}</style><main>',
           '<h1>Fish2Dish — 250 selected fish photos</h1><p class="note">50 per class · source-labeled and visually screened · no generated or augmented samples. Sources: Fish-gres V2 and licensed iNaturalist research-grade observations. Photos are from multiple countries; this is not a Philippine-market-only photo collection. Source labels are not a guarantee of expert-certified identity or future model accuracy.</p>',
           '<nav>'+''.join(f'<a href="#{label}">{label} (50)</a>' for label in counts)+'</nav>']
    for label in counts:
        subset=[r for r in rows if r['label']==label]
        pages.append(f'<h2 id="{label}">{html.escape(subset[0]["local_name"])} — <i>{html.escape(subset[0]["scientific_name"])}</i></h2><section>')
        for row in subset:
            pages.append(f'<figure><a href="../{row["path"]}"><img loading="lazy" src="../{row["path"]}" alt="{html.escape(row["scientific_name"])}"></a><figcaption><strong>{Path(row["path"]).name}</strong><br>{html.escape(row["attribution"])}<br><a href="{html.escape(row["source_url"])}">Source record</a> · <a href="{row["license_url"]}">{row["license"]}</a></figcaption></figure>')
        pages.append('</section>')
    pages.append('</main></html>')
    (DATA/'review/index.html').write_text('\n'.join(pages),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    build()
