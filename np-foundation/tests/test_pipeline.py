import csv,json
from pathlib import Path
from npfm.pipeline import analyze

def test_name_cannot_override_structure_identity(tmp_path):
    p=tmp_path/'input.csv'
    p.write_text('compound_id,name,smiles\nfake,paclitaxel,CCO\n')
    result=analyze(p,tmp_path/'output')
    record=json.loads(Path(result['report']).read_text())['records'][0]
    assert record['verified_evidence_identity'] is None
    assert record['indication_evidence']['clinical_efficacy_probability'] is None
    assert record['indication_evidence']['evidence_records']==[]
