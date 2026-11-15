import argparse,json,sys
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description='Original NP-GPT causal molecular Transformer')
    s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('prepare');c.add_argument('--input',required=True);c.add_argument('--output',required=True);c.add_argument('--max-seq-length',type=int,default=1024);c.add_argument('--seed',default='np-gpt-v1')
    c=s.add_parser('train');c.add_argument('--config',required=True);c.add_argument('--corpus');c.add_argument('--output');c.add_argument('--resume');c.add_argument('--max-steps',type=int);c.add_argument('--max-train-records',type=int);c.add_argument('--device');c.add_argument('--epochs',type=int)
    c=s.add_parser('generate');c.add_argument('--checkpoint',required=True);c.add_argument('--n',type=int,default=128);c.add_argument('--max-new-tokens',type=int,default=384);c.add_argument('--temperature',type=float,default=.8);c.add_argument('--top-k',type=int,default=50);c.add_argument('--top-p',type=float,default=.95);c.add_argument('--prefix',default='');c.add_argument('--seed',type=int,default=42);c.add_argument('--batch-size',type=int,default=32);c.add_argument('--device',default='cpu');c.add_argument('--output',required=True)
    c=s.add_parser('evaluate');c.add_argument('--input',required=True);c.add_argument('--corpus',required=True);c.add_argument('--output',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='prepare':
            from .corpus import prepare_corpus
            result=prepare_corpus(a.input,a.output,max_seq_length=a.max_seq_length,seed=a.seed)
        elif a.command=='train':
            from .training import train_from_config
            absolute=lambda x:str(Path(x).resolve()) if x else None
            result=train_from_config(a.config,{'corpus_dir':absolute(a.corpus),'output_dir':absolute(a.output),'resume':absolute(a.resume),'max_steps':a.max_steps,'max_train_records':a.max_train_records,'device':a.device,'epochs':a.epochs})
        elif a.command=='generate':
            from .generation import generate
            result={'attempts':len(generate(a.checkpoint,a.n,max_new_tokens=a.max_new_tokens,temperature=a.temperature,top_k=a.top_k,top_p=a.top_p,seed=a.seed,prefix=a.prefix,batch_size=a.batch_size,device=a.device,output_path=a.output)),'output':a.output}
        else:
            from .evaluation import evaluate_generated
            with open(a.input) as f:records=[json.loads(line) for line in f if line.strip()]
            result=evaluate_generated(records,a.corpus,a.output)
            result={'metrics':result['metrics'],'output':a.output}
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str));return 0
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
