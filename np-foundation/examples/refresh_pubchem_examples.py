"""Re-fetch curated examples from PubChem; retain unmodified response JSON."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request

OUTPUT = Path(__file__).resolve().parent
NAMES = ['paclitaxel','artemisinin','quercetin','resveratrol','curcumin','berberine',
         'caffeine','nicotine','morphine','salicylic acid','epigallocatechin gallate',
         'apigenin','kaempferol','luteolin','naringenin','genistein','daidzein','emodin',
         'honokiol','magnolol','thymoquinone','celastrol','ursolic acid','oleanolic acid',
         'glycyrrhizin','baicalein','baicalin','andrographolide','artemisinic acid',
         'shikonin','cantharidin','piperine','capsaicin','caffeic acid','ferulic acid','gallic acid']
LOCK = threading.Lock()
LAST_REQUEST = 0.0

def fetch(name):
    global LAST_REQUEST
    cache = OUTPUT / 'pubchem_cache' / (name.replace(' ', '_') + '.json')
    if cache.exists():
        item = json.loads(cache.read_text())
        if item.get('response', {}).get('PropertyTable', {}).get('Properties'):
            return item
    url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
           + urllib.parse.quote(name, safe='')
           + '/property/SMILES,ConnectivitySMILES,InChIKey/JSON')
    for attempt in range(4):
        try:
            with LOCK:
                delay = max(0, LAST_REQUEST + 0.27 - time.monotonic())
                if delay: time.sleep(delay)
                LAST_REQUEST = time.monotonic()
            request = urllib.request.Request(url, headers={'User-Agent':'NPFM-Example-Recovery/1.0'})
            with urllib.request.urlopen(request, timeout=35) as response:
                raw = response.read()
                status = response.status
            payload = json.loads(raw)
            properties = payload['PropertyTable']['Properties']
            if len(properties) != 1:
                raise ValueError(f'Ambiguous response for {name}: {len(properties)} CIDs')
            prop = properties[0]
            for key in ('CID','SMILES','ConnectivitySMILES','InChIKey'):
                if not prop.get(key): raise ValueError(f'Missing {key} for {name}')
            item = {'name':name,'source_url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                    'http_status':status,'raw_response_sha256':hashlib.sha256(raw).hexdigest(),
                    'response':payload}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(item, ensure_ascii=False, indent=2)+'\n')
            print(f"retrieved {name}: CID {prop['CID']}", flush=True)
            return item
        except Exception:
            if attempt == 3: raise
            time.sleep(2**attempt)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        provenance = list(pool.map(fetch,NAMES))
    rows=[]
    for record in provenance:
        p=record['response']['PropertyTable']['Properties'][0]
        rows.append({'id':str(p['CID']),'compound_id':str(p['CID']),'cid':p['CID'],
                     'name':record['name'],'smiles':p['SMILES'],
                     'canonical_smiles':p['ConnectivitySMILES'],'isomeric_smiles':p['SMILES'],
                     'inchikey':p['InChIKey'],'origin_class':'curated_natural_product_example',
                     'source':'PubChem PUG REST','source_url':record['source_url'],
                     'record_url':f"https://pubchem.ncbi.nlm.nih.gov/compound/{p['CID']}",
                     'retrieved_at':record['retrieved_at'],'label_status':'unlabeled','label':''})
    with (OUTPUT/'compounds.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (OUTPUT/'pubchem_provenance.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in provenance))
    print(json.dumps({'examples':len(rows),'unique_cids':len({r['cid'] for r in rows}),
                      'output':str(OUTPUT/'compounds.csv')}),flush=True)

if __name__ == '__main__': main()
