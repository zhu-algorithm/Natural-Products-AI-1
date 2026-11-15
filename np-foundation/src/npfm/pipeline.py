"""One-command chemistry, trained endpoints and identity-verified evidence."""
import csv,json
from pathlib import Path
from .chemistry import descriptors,describe_csv
from .evidence import EvidenceStore

ROOT=Path(__file__).resolve().parents[2]

def analyze(input_path,output_dir,checkpoint=None,evidence_path=None,identity_reference=None,disease=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with open(input_path,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    refs={}
    with open(identity_reference or ROOT/'examples/compounds.csv',newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=descriptors(row['smiles'])['full_inchikey']
            claimed=row.get('full_inchikey') or row.get('inchikey')
            if claimed and claimed!=key:raise ValueError('Identity reference structure/key mismatch')
            refs[key]={'name':row['name'],'query':'PubChem:'+row['cid'] if row.get('cid') else row['name'],'reference_id':row['compound_id']}
    store=EvidenceStore.from_jsonl(evidence_path or ROOT/'examples/evidence.jsonl')
    describe_csv(input_path,out/'descriptors.csv')
    predictions=[];prediction_metadata=None
    if checkpoint:
        from .predict import predict_file
        prediction_metadata=predict_file(checkpoint,input_path,out/'predictions.csv')
        with (out/'predictions.csv').open(newline='',encoding='utf-8-sig') as f:predictions=list(csv.DictReader(f))
        if len(predictions)!=len(rows):raise ValueError('Prediction row count mismatch')
    result=[]
    for i,row in enumerate(rows):
        item={'compound_id':row.get('compound_id'),'input_name':row.get('name'),'input_smiles':row.get('smiles')}
        try:
            item['calculated_descriptors']=descriptors(row['smiles'])
            identity=refs.get(item['calculated_descriptors']['full_inchikey'])
            item['verified_evidence_identity']=identity
            item['indication_evidence']=store.indication_report(identity['query'],disease) if identity else {
                'status':'unknown_identity_outside_reference','clinical_efficacy_probability':None,'evidence_records':[],
                'interpretation':'This structure is outside the verified reference table; names alone are not evidence.'}
        except (ValueError,KeyError) as exc:
            item['error']=str(exc);item['verified_evidence_identity']=None
        if predictions:
            pred=predictions[i]
            if pred['compound_id']!=row['compound_id'] or pred['smiles']!=row['smiles']:raise ValueError('Prediction identity mismatch')
            item['learned_predictions']={k:v for k,v in pred.items() if k not in {'compound_id','smiles'}}
        result.append(item)
    report={'records':result,'prediction_metadata':prediction_metadata,'scope':'Research predictions and cited evidence; no clinical efficacy probability'}
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return {'output_dir':str(out.resolve()),'record_count':len(result),'with_verified_evidence_identity':sum(bool(r['verified_evidence_identity']) for r in result),'learned_tasks':prediction_metadata['trained_tasks'] if prediction_metadata else [],'report':str((out/'analysis.json').resolve())}
