"""Download licensed unsupported-fish/non-fish checks, never training positives."""
import concurrent.futures
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from download_inaturalist_candidates import observation_candidates, fetch, DATA

def main():
    all_rows=[]
    for species in [dict(id='unsupported-barramundi',localName='Unsupported barramundi',scientificName='Lates calcarifer'), dict(id='nonfish-sparrow',localName='House sparrow (non-fish)',scientificName='Passer domesticus')]:
        candidates=observation_candidates(species)[:36]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows=[row for row in pool.map(fetch,candidates) if 'error' not in row]
        if len(rows)<20:raise ValueError('Insufficient licensed negative checks')
        # Every observer remains in exactly one partition, including dates.
        observers=sorted(set(r['observer'] for r in rows));random.Random(42).shuffle(observers)
        validation=set(observers[:len(observers)//2])
        for row in rows:
            row['partition']='validation' if row['observer'] in validation else 'test'
            row['path']=str((DATA/row['path']).resolve())
        all_rows.extend(rows)
        sheet=Image.new('RGB',(6*180,6*155),'white');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=ImageOps.exif_transpose(Image.open(row['path'])).convert('RGB');im.thumbnail((176,125))
            x,y=(i%6)*180,(i//6)*155;sheet.paste(im,(x,y));draw.text((x,y+126),f"{i+1}: {row['partition']}",fill='black')
        sheet.save(DATA/'review'/f"{species['id']}-checks.jpg")
        print(species['id'],len(rows),flush=True)
    (DATA/'provenance/negative-images.json').write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
    for part in ['validation','test']:
        (DATA/f'{part}-negatives.json').write_text(json.dumps([r['path'] for r in all_rows if r['partition']==part],indent=2))

if __name__=='__main__':main()
