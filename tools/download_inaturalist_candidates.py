"""Acquire licensed, research-grade photo candidates, one photo per observation.

The API quality grade and exact taxon are checked. Images still require visual
screening: community labels are stronger evidence than filenames, not infallible.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from download_market_dataset import ROOT

CLASSES = [
    {"id":"dalag", "localName":"Dalag", "scientificName":"Channa striata"},
    {"id":"hito", "localName":"Hito (African sharptooth catfish)", "scientificName":"Clarias gariepinus"},
    {"id":"karpa", "localName":"Karpa (common carp)", "scientificName":"Cyprinus carpio"},
]
LICENSES = {"cc0":"CC0 1.0", "cc-by":"CC BY 4.0", "cc-by-sa":"CC BY-SA 4.0"}
LICENSE_URLS = {"cc0":"https://creativecommons.org/publicdomain/zero/1.0/", "cc-by":"https://creativecommons.org/licenses/by/4.0/", "cc-by-sa":"https://creativecommons.org/licenses/by-sa/4.0/"}
DATA=ROOT/'data'


def observation_candidates(species):
    snapshot=DATA/'provenance'/f"inat-{species['scientificName'].replace(' ','-')}.json"
    if snapshot.exists():
        result=json.loads(snapshot.read_text(encoding='utf-8'))
    else:
        params=urllib.parse.urlencode({'taxon_name':species['scientificName'],'quality_grade':'research','photos':'true','photo_license':','.join(LICENSES),'per_page':200,'order_by':'id','order':'desc'})
        request=urllib.request.Request('https://api.inaturalist.org/v1/observations?'+params,headers={'User-Agent':'Fish2Dish educational research/0.1'})
        result=json.load(urllib.request.urlopen(request,timeout=45))
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=[]
    for obs in result['results']:
        if obs.get('quality_grade')!='research' or obs.get('taxon',{}).get('name')!=species['scientificName']:
            continue
        photos=[p for p in obs.get('photos',[]) if p.get('license_code') in LICENSES]
        if not photos:
            continue
        photo=photos[0]
        url=re.sub(r'/square\.', '/large.',photo['url'])
        rows.append({**species,'observationId':obs['id'],'taxonId':obs['taxon']['id'],'communityTaxonId':obs.get('community_taxon_id'),'photoId':photo['id'],
                     'url':url,'sourceUrl':obs.get('uri',f"https://www.inaturalist.org/observations/{obs['id']}"),
                     'photoPage':f"https://www.inaturalist.org/photos/{photo['id']}",'license':LICENSES[photo['license_code']],
                     'licenseUrl':LICENSE_URLS[photo['license_code']],'attribution':photo.get('attribution',''),
                     'observer':obs['user']['login'],'observedOn':obs.get('observed_on'),'qualityGrade':'research'})
    return rows


def fetch(row):
    directory=DATA/'candidates'/'inaturalist'/row['id']
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{row['observationId']}-{row['photoId']}.jpg"
    for attempt in range(2):
        try:
            if path.exists():
                raw=path.read_bytes()
            else:
                request=urllib.request.Request(row['url'],headers={'User-Agent':'Fish2Dish educational research/0.1'})
                with urllib.request.urlopen(request,timeout=45) as response:
                    raw=response.read(12*1024*1024+1)
                if len(raw)>12*1024*1024:
                    raise ValueError('Image too large')
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                if min(image.size)<160:
                    raise ValueError('Image too small')
                width,height=image.size
            path.write_bytes(raw)
            return {**row,'path':path.relative_to(DATA).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'width':width,'height':height}
        except Exception as error:
            if attempt==1:
                return {**row,'error':f'{type(error).__name__}: {error}'}
            time.sleep(1)


if __name__=='__main__':
    all_rows=[]
    for species in CLASSES:
        candidates=observation_candidates(species)
        count=min(len(candidates),100 if species['id']=='karpa' else 80)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            rows=[]
            for row in executor.map(fetch,candidates[:count]):
                rows.append(row)
                if len(rows)%20==0:print(f"{species['id']}: {len(rows)}/{count} licensed candidates processed",flush=True)
        all_rows.extend(rows)
        print(f"{species['id']}: {sum('error' not in row for row in rows)} usable source-labeled candidates",flush=True)
        (DATA/'provenance/inaturalist-candidates.json').write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
