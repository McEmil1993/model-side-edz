"""Restore only the frozen 250 selected photos from tracked dataset.csv.

The same source URLs/member names and checksums are used; no new search results
are silently substituted. Requires Pillow and UnRAR/bsdtar for Fish-gres TIFFs.
"""
import concurrent.futures
import argparse
import csv
import hashlib
import html
import io
import json
import urllib.request

from PIL import Image,ImageOps
from download_market_dataset import ROOT,CONFIG,download
from select_market_images import tool,read_member,contact_sheets

DATA=ROOT/'data'


def restore(verify_only=False):
    with (ROOT/'dataset.csv').open(encoding='utf-8') as file:
        rows=list(csv.DictReader(file))
    sources=json.loads(CONFIG.read_text())
    source_by_id={c['id']:c for c in sources['classes'] if c.get('eligible',True)}
    missing=[]
    for row in rows:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to((DATA/'raw').resolve()):
            raise ValueError('Destination outside dataset raw folder')
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            missing.append(row)
    if verify_only:
        if missing:raise ValueError(f'{len(missing)} missing or changed photos')
        print(f'All {len(rows)} frozen selected photo checksums match.')
        return
    if any(row['label'] in source_by_id for row in missing):
        executable,is_unrar=tool()
        cache=DATA/'source-archives';cache.mkdir(parents=True,exist_ok=True)
        for label in {row['label'] for row in missing if row['label'] in source_by_id}:
            download(source_by_id[label],cache)
    def recover(row):
        path=ROOT/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if row['label'] in source_by_id:
            archive=DATA/'source-archives'/source_by_id[row['label']]['archive']
            raw=read_member(executable,is_unrar,archive,row['original_member'])
            if hashlib.sha256(raw).hexdigest()!=row['original_sha256']:
                raise ValueError('Source archive member differs')
            with Image.open(io.BytesIO(raw)) as image:
                buffer=io.BytesIO();ImageOps.exif_transpose(image).convert('RGB').save(buffer,format='PNG');raw=buffer.getvalue()
        else:
            url=row['download_url']
            if not url.startswith('https://'):
                raise ValueError('HTTPS source photo URL required')
            request=urllib.request.Request(url,headers={'User-Agent':'Fish2Dish educational dataset restoration/0.1'})
            with urllib.request.urlopen(request,timeout=60) as response:
                raw=response.read(12*1024*1024+1)
            if len(raw)>12*1024*1024:
                raise ValueError('Unexpected image size')
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError(f"Source/rendering changed for {row['path']}; do not substitute silently. Check source and Pillow version.")
        path.write_bytes(raw)
        return row['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(recover,missing):print('Restored',path,flush=True)
    local_rows=[{**row,'path':str((ROOT/row['path']).relative_to(DATA)).replace('\\','/')} for row in rows]
    (DATA/'review').mkdir(parents=True,exist_ok=True)
    for label in dict.fromkeys(row['label'] for row in rows):
        contact_sheets([row for row in local_rows if row['label']==label],label)
    cards=[]
    for row in local_rows:
        cards.append(f'<figure><img loading="lazy" src="../{row["path"]}" alt="{html.escape(row["scientific_name"])}"><figcaption>{row["label"]} · {html.escape(row["attribution"])} · <a href="{row["license_url"]}">{row["license"]}</a></figcaption></figure>')
    (DATA/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fish2Dish dataset</title><style>body{font:14px Arial;background:#eef5f6;margin:25px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:15px}figure{background:white;margin:0;padding:10px}img{width:100%;height:180px;object-fit:contain}figcaption{font-size:11px}</style><h1>250 selected source-labeled photos · 50 per class</h1><p>Not expert-certified. Not all photos were captured in the Philippines. Preserve credits and individual licenses.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
    print(f'All {len(rows)} frozen selected photo checksums match.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    restore(parser.parse_args().verify_only)
