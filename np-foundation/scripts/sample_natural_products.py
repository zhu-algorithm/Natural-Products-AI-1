"""Reproducible bounded-length reservoir sample; record exclusions explicitly."""
import argparse,csv,hashlib,json,random
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--n',type=int,default=2048);p.add_argument('--max-characters',type=int,default=380);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();rng=random.Random(a.seed);sample=[];eligible=0;excluded=0;total=0
    with open(a.input,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            total+=1;s=r.get('smiles','')
            if not s or len(s)>a.max_characters:excluded+=1;continue
            eligible+=1
            v={k:r.get(k,'') for k in ['compound_id','smiles','full_inchikey','source_id','source_record_id','license']}
            if len(sample)<a.n:sample.append(v)
            else:
                j=rng.randrange(eligible)
                if j<a.n:sample[j]=v
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if not sample:raise ValueError('No eligible structures')
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(sample[0]),lineterminator='\n');w.writeheader();w.writerows(sample)
    h=hashlib.sha256()
    with open(a.input,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    report={'input':str(Path(a.input).resolve()),'source_sha256':h.hexdigest(),'sample_rows':len(sample),'eligible_rows':eligible,'excluded_missing_or_long_smiles':excluded,'total_rows':total,'max_characters':a.max_characters,'seed':a.seed,'sample_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
