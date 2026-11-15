"""Checksummed, bounded-size checkpoint storage for original NP-GPT weights."""
from __future__ import annotations
import hashlib,io,json,os
from pathlib import Path
import torch
from .model import NPTransformer,NPTransformerConfig
from ..tokenizer import SmilesTokenizer

FORMAT='np-gpt-decoder-v1'
MAX_MODEL_BYTES=512*1024*1024
MAX_MODEL_PARAMETERS=100_000_000

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def safe_path(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Checkpoint paths must remain inside their directory')
    base=Path(root).resolve();target=(base/p).resolve()
    if not target.is_relative_to(base):raise ValueError('Checkpoint paths must remain inside their directory')
    return target

def save_model(model,tokenizer,directory,metadata=None,part_bytes=256*1024):
    """Tensor-only torch serialization split into independently hashed small files."""
    if isinstance(part_bytes,bool) or not isinstance(part_bytes,int) or part_bytes<1:raise ValueError('part_bytes must be a positive integer')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},buffer)
    raw=buffer.getvalue();parts=[]
    for i,start in enumerate(range(0,len(raw),part_bytes),1):
        name=f'model.part{i:04}.bin';chunk=raw[start:start+part_bytes]
        p=directory/name;temp=p.with_suffix('.tmp');temp.write_bytes(chunk);temp.replace(p)
        parts.append({'path':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    write_json(directory/'model_config.json',model.config.to_dict())
    tokenizer.save(directory/'tokenizer.json')
    index={'format':FORMAT,'serialization':'torch_state_dict_weights_only','model_sha256':hashlib.sha256(raw).hexdigest(),'model_bytes':len(raw),'parts':parts,'config_sha256':sha256(directory/'model_config.json'),'tokenizer_sha256':sha256(directory/'tokenizer.json'),'parameter_count':model.num_parameters(),'metadata':metadata or {}}
    write_json(directory/'model.index.json',index)
    active={p['path'] for p in parts}
    for p in directory.glob('model.part*.bin'):
        if p.name not in active:p.unlink()
    return index

def load_model(directory,device='cpu'):
    directory=Path(directory);index=json.loads((directory/'model.index.json').read_text())
    if index.get('format')!=FORMAT:raise ValueError('Unsupported NP-GPT checkpoint format')
    if not isinstance(index.get('model_bytes'),int) or not 0<index['model_bytes']<=MAX_MODEL_BYTES:raise ValueError('Checkpoint exceeds the supported model byte budget')
    if not isinstance(index.get('parts'),list) or not index['parts'] or len(index['parts'])>10000:raise ValueError('Invalid checkpoint shard count')
    if any(not isinstance(p.get('bytes'),int) or p['bytes']<1 for p in index['parts']) or sum(p['bytes'] for p in index['parts'])!=index['model_bytes']:raise ValueError('Invalid checkpoint shard byte accounting')
    for name,field in [('model_config.json','config_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if sha256(directory/name)!=index[field]:raise ValueError('Checkpoint metadata hash mismatch: '+name)
    raw=io.BytesIO();seen=set()
    for part in index['parts']:
        if part['path'] in seen:raise ValueError('Duplicate checkpoint shard')
        seen.add(part['path']);p=safe_path(directory,part['path'])
        if p.stat().st_size!=part['bytes'] or sha256(p)!=part['sha256']:raise ValueError('Weight shard hash mismatch: '+part['path'])
        raw.write(p.read_bytes())
    if raw.tell()!=index['model_bytes'] or hashlib.sha256(raw.getbuffer()).hexdigest()!=index['model_sha256']:
        raise ValueError('Combined model hash mismatch')
    config=NPTransformerConfig.from_dict(json.loads((directory/'model_config.json').read_text()))
    d=config.d_model;kv=config.n_kv_heads*(d//config.n_heads)
    parameters=config.vocab_size*d*(1 if config.tie_embeddings else 2)+config.n_layers*(2*d*d+2*d*kv+3*d*config.ffn_hidden_dim+2*d)+d
    if parameters>MAX_MODEL_PARAMETERS or parameters!=index.get('parameter_count'):raise ValueError('Model configuration exceeds the parameter budget or disagrees with its index')
    tokenizer=SmilesTokenizer.load(directory/'tokenizer.json')
    if len(tokenizer)!=config.vocab_size:raise ValueError('Tokenizer/model vocabulary mismatch')
    model=NPTransformer(config);raw.seek(0)
    model.load_state_dict(torch.load(raw,map_location='cpu',weights_only=True),strict=True)
    model.to(device);model.eval()
    return model,tokenizer,index
