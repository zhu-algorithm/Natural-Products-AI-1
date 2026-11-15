"""Download ESOL measured labels and append unlabeled natural-product structures."""
import argparse,csv,hashlib,io,json
from pathlib import Path
from urllib.request import urlopen

SOURCE='https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv'

def main():
    p=argparse.ArgumentParser();p.add_argument('--natural-products',required=True)
    p.add_argument('--output',default='data/training/solubility.csv');a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    raw=urlopen(SOURCE,timeout=90).read()
    original=out.parent/'delaney-processed.csv';original.write_bytes(raw)
    data=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    measured='measured log solubility in mols per litre'
    if not data or measured not in data[0]:raise ValueError('Measured ESOL column is absent')
    rows=[]
    for i,r in enumerate(data):
        value=float(r[measured])
        rows.append({'compound_id':f'esol:{i+1}','smiles':r['smiles'],'logS':value,'label_source':SOURCE,'source_id':'esol'})
    with open(a.natural_products,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append({'compound_id':r['compound_id'],'smiles':r['smiles'],'logS':'','label_source':'','source_id':r.get('source_id','natural_product')})
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    report={'url':SOURCE,'source_sha256':hashlib.sha256(raw).hexdigest(),'measured_column':measured,'measured_rows':len(data),'unlabeled_rows':len(rows)-len(data),'output_rows':len(rows),'natural_product_source':str(Path(a.natural_products).resolve()),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scope':'Measured aqueous solubility benchmark; no clinical or disease label'}
    out.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
