"""Compare measured logS on the exact recorded neural-model split."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from npfm.train import checkpoint_bytes

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);a=p.parse_args()
    d=Path(a.checkpoint);manifest=json.loads((d/'data_manifest.json').read_text())
    if hashlib.sha256(Path(a.input).read_bytes()).hexdigest()!=manifest['source_file_sha256']:raise ValueError('Training input hash mismatch')
    with open(a.input,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))[:manifest['used_records']]
    split_bytes=checkpoint_bytes(d/'splits.json');split=json.loads(split_bytes);gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    def make(indices):
        xs=[];ys=[]
        for i in indices:
            r=rows[i]
            if not r.get('logS','').strip():continue
            m=Chem.MolFromSmiles(r['smiles'])
            if m is None:raise ValueError('Invalid structure')
            x=np.zeros(2048,dtype=np.float32);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),x)
            xs.append(x);ys.append(float(r['logS']))
        return np.array(xs),np.array(ys)
    x,y=make(split['train']);xt,yt=make(split['test'])
    rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=4).fit(x,y)
    def metrics(pred):return {'mae':float(mean_absolute_error(yt,pred)),'rmse':float(mean_squared_error(yt,pred)**.5),'n_labels':len(yt)}
    report={'source_sha256':manifest['source_file_sha256'],'split_sha256':hashlib.sha256(split_bytes).hexdigest(),'n_train_labels':len(y),'n_test_labels':len(yt),'morgan_random_forest':metrics(rf.predict(xt)),'constant_training_mean':metrics(np.full(len(yt),y.mean())),'scope':'One pilot split; no external natural-product validation'}
    (d/'baseline_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
