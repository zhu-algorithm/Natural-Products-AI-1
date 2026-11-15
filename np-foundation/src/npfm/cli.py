"""Command-line interface for the reproducible research workflow."""
import argparse,importlib.metadata,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def emit(value,path=None):
    text=json.dumps(value,ensure_ascii=False,indent=2,default=str)
    if path:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+'\n')
    print(text)

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {'download','data','upstream'}:
        from importlib import import_module
        return import_module('npfm.'+argv[0]).main(argv[1:])
    p=argparse.ArgumentParser(description='Natural-product molecular learning and traceable evidence')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    c=sub.add_parser('catalog');c.add_argument('kind',choices=['algorithms','databases','endpoints'])
    c=sub.add_parser('descriptors');c.add_argument('--input',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('conformers');c.add_argument('--smiles',required=True);c.add_argument('--output',required=True);c.add_argument('--n',type=int,default=10)
    c=sub.add_parser('similarity');c.add_argument('--a',required=True);c.add_argument('--b',required=True)
    c=sub.add_parser('train');c.add_argument('--config',required=True);c.add_argument('--data');c.add_argument('--output');c.add_argument('--max-records',type=int);c.add_argument('--epochs',type=int)
    c=sub.add_parser('predict');c.add_argument('--checkpoint',required=True);c.add_argument('--input',required=True);c.add_argument('--output',required=True);c.add_argument('--tasks')
    c=sub.add_parser('generate');c.add_argument('--checkpoint',required=True);c.add_argument('--n',type=int,default=10);c.add_argument('--max-tokens',type=int);c.add_argument('--temperature',type=float,default=1.0);c.add_argument('--output',required=True)
    c=sub.add_parser('evidence');c.add_argument('--compound',required=True);c.add_argument('--disease');c.add_argument('--source',default=str(ROOT/'examples/evidence.jsonl'));c.add_argument('--output')
    c=sub.add_parser('analyze');c.add_argument('--input',required=True);c.add_argument('--output',required=True);c.add_argument('--checkpoint');c.add_argument('--evidence');c.add_argument('--identity-reference');c.add_argument('--disease')
    c=sub.add_parser('admet');c.add_argument('--smiles',required=True);c.add_argument('--output')
    c=sub.add_parser('dock');c.add_argument('--receptor',required=True);c.add_argument('--ligand',required=True);c.add_argument('--center',nargs=3,type=float,required=True);c.add_argument('--box-size',nargs=3,type=float,required=True);c.add_argument('--output',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='doctor':
            versions={}
            for name in ['npfm','torch','rdkit','numpy','scikit-learn','pytest','admet-ai','vina']:
                try:versions[name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:versions[name]='not_installed'
            emit({'python':sys.version,'versions':versions,'project_root':str(ROOT),'scope':'Optional adapters require external software and weights'})
        elif a.command=='catalog':
            name={'endpoints':'endpoint_algorithms'}.get(a.kind,a.kind);emit(json.loads((ROOT/'registry'/f'{name}.json').read_text()))
        elif a.command=='descriptors':
            from .chemistry import describe_csv
            emit(describe_csv(a.input,a.output))
        elif a.command=='conformers':
            from .chemistry import conformers
            emit(conformers(a.smiles,a.output,a.n))
        elif a.command=='similarity':
            from .chemistry import similarity
            emit({'morgan_chiral_tanimoto':similarity(a.a,a.b)})
        elif a.command=='train':
            from .train import train_from_config
            o={'data_path':str(Path(a.data).resolve()) if a.data else None,'output_dir':str(Path(a.output).resolve()) if a.output else None,'max_records':a.max_records,'epochs':a.epochs}
            emit({'checkpoint':train_from_config(a.config,o)})
        elif a.command=='predict':
            from .predict import predict_file
            emit(predict_file(a.checkpoint,a.input,a.output,requested_tasks=a.tasks))
        elif a.command=='generate':
            from .predict import sample_smiles
            emit(sample_smiles(a.checkpoint,a.n,a.max_tokens,a.temperature),a.output)
        elif a.command=='evidence':
            from .evidence import EvidenceStore
            emit(EvidenceStore.from_jsonl(a.source).indication_report(a.compound,a.disease),a.output)
        elif a.command=='analyze':
            from .pipeline import analyze
            emit(analyze(a.input,a.output,a.checkpoint,a.evidence,a.identity_reference,a.disease))
        elif a.command=='admet':
            from .adapters import admet_predict
            result=admet_predict(a.smiles)
            emit(result.to_dict() if hasattr(result,'to_dict') else result,a.output)
        elif a.command=='dock':
            from .adapters import dock_vina
            emit(dock_vina(a.receptor,a.ligand,a.center,a.box_size,a.output))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
