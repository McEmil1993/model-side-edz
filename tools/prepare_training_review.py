"""Freeze conservative, visually reviewed capture groups for the first baseline."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import prepare

# Every photo was inspected in the two contact sheets per class. Keep a complete
# repeated background/setup together, even when original filenames have other dates.
SETUPS = {
    'bangus': {
        'plastic': [1,4,8,9,10,14,16,25,32,35,38,45,49,50],
        'blue-tile': [2,41], 'concrete': [3,23,24,29,31,33,43,46],
        'white-tile': [5,11,13,26,28,30,34,40],
        'red-tray': [6,17,20,22,36,37,39,48],
        'pavers': [7,12,19,21,42,44], 'yellow': [15], 'sack': [18,27,47],
    },
    'tilapia': {
        'pavers': [1,6,18,19,24,33,38,49],
        'white-tile': [2,3,9,16,27,35,36],
        'yellow': [4,7,28,30], 'red-tray': [5,17,34,37,42],
        'concrete': [8,23,26,29,39,46,47], 'pile': [10],
        'blue-tile': [11,12,20,45,48],
        'plastic': [13,15,21,25,31,32,40,43],
        'green-tray': [14,41], 'sack': [22,44,50],
    },
}

def main():
    with (ROOT/'dataset.csv').open(encoding='utf-8', newline='') as handle:
        rows=list(csv.DictReader(handle))
    for label, setups in SETUPS.items():
        indices=[i for numbers in setups.values() for i in numbers]
        assert sorted(indices)==list(range(1,51)), label
    for row in rows:
        row['path']=str((ROOT/row['path']).resolve())
        row['acquisition_group']=row['group']
        if row['label'] in SETUPS:
            index=int(Path(row['path']).stem.rsplit('-',1)[1])
            setup=next(k for k,v in SETUPS[row['label']].items() if index in v)
            row['group']=f"reviewed-{row['label']}-{setup}"
            row['group_basis']='Conservative visual background/setup cluster; specimen identity unverified'
        else:
            # Observers can photograph the same animal across different dates.
            # The original group encodes observer then ISO date; use that exact observer.
            row['group']=row['acquisition_group'].rsplit('-',3)[0]
            row['group_basis']='Same source observer kept together across all dates; specimen identity unverified'
    target=ROOT/'data/training-reviewed.csv'
    with target.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    registry=json.loads((ROOT/'config/species.json').read_text())
    registry['reviewed']=True
    registry['reviewLevel']='Source taxonomy plus visual screening for experimental training; not expert certification'
    registry['reviewNotes']='All 250 accepted photos visually screened. Fish-gres repeated setups grouped together; iNaturalist photos grouped by observer across dates. Related species, geography and market domain remain unverified. Quarantined classes excluded.'
    by_name={Path(r['path']).name:r for r in rows}
    # Coarse 8x8 average-hash collisions were individually checked in contact
    # sheets: these are different taxa and visibly different photographs.
    pairs=[('bangus-009.png','hito-031.jpg'),('bangus-038.png','hito-031.jpg'),('bangus-031.png','karpa-010.jpg'),('dalag-043.jpg','karpa-026.jpg')]
    registry['distinctImageReviews']=[{'sha256':[by_name[a]['sha256'],by_name[b]['sha256']], 'files':[a,b], 'reason':'Visually distinct species, body outlines and scenes; coarse average-hash collision, not a duplicate.'} for a,b in pairs]
    (ROOT/'config/species.json').write_text(json.dumps(registry,indent=2)+'\n')
    split=prepare(target,ROOT/'config/species.json',ROOT/'data/splits.json')
    print(json.dumps({name:{label:sum(r['label']==label for r in part) for label in split['labels']} for name,part in split['splits'].items()},indent=2))

if __name__=='__main__':main()
