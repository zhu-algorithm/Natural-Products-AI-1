"""Seeded autoregressive molecular sequence generation with batched KV caches."""
from __future__ import annotations
import json,math
from pathlib import Path
import torch
from .runtime import load_model,write_json,sha256

def filter_logits(logits,top_k=50,top_p=.95):
    if top_k is not None:
        if isinstance(top_k,bool) or int(top_k)<1:raise ValueError('top_k must be positive or null')
        k=min(int(top_k),logits.shape[-1]);threshold=torch.topk(logits,k,dim=-1).values[:,-1:]
        logits=logits.masked_fill(logits<threshold,float('-inf'))
    if not math.isfinite(float(top_p)) or not 0<top_p<=1:raise ValueError('top_p must be in (0,1]')
    if top_p<1:
        values,indices=torch.sort(logits,dim=-1,descending=True)
        remove=values.softmax(-1).cumsum(-1)>top_p
        remove[:,1:]=remove[:,:-1].clone();remove[:,0]=False
        mask=torch.zeros_like(remove).scatter(1,indices,remove)
        logits=logits.masked_fill(mask,float('-inf'))
    return logits

@torch.no_grad()
def generate(checkpoint,n=128,*,max_new_tokens=384,temperature=.8,top_k=50,top_p=.95,seed=42,prefix='',batch_size=32,device='cpu',output_path=None,cpu_threads=4):
    if n<1 or batch_size<1 or max_new_tokens<1:raise ValueError('Counts and batch size must be positive')
    if not math.isfinite(float(temperature)) or temperature<=0:raise ValueError('temperature must be finite and positive')
    torch.set_num_threads(cpu_threads);target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    model,tokenizer,index=load_model(checkpoint,device=target)
    prompt=[tokenizer.bos_id]+(tokenizer.encode(prefix,add_special_tokens=False) if prefix else [])
    if tokenizer.unk_id in prompt:raise ValueError('Prefix contains a token outside the fitted training vocabulary')
    if len(prompt)+max_new_tokens>model.config.max_seq_len:raise ValueError('Requested generation exceeds the model context; reduce max_new_tokens')
    generator=torch.Generator(device=target).manual_seed(seed);records=[]
    for start in range(0,n,batch_size):
        count=min(batch_size,n-start);ids=torch.tensor([prompt]*count,dtype=torch.long,device=target)
        mask=torch.ones_like(ids,dtype=torch.bool);result=model(ids,attention_mask=mask,use_cache=True)
        past=result['past_key_values'];logits=result['logits'][:,-1];finished=torch.zeros(count,dtype=torch.bool,device=target)
        sequences=[list(prompt) for _ in range(count)]
        for _ in range(max_new_tokens):
            scores=logits.float()/temperature
            scores[:,[tokenizer.pad_id,tokenizer.bos_id,tokenizer.unk_id,tokenizer.mask_id]]=float('-inf')
            scores=filter_logits(scores,top_k,top_p)
            following=torch.multinomial(scores.softmax(-1),1,generator=generator).squeeze(1)
            active=~finished;following=torch.where(active,following,tokenizer.pad_id)
            active_cpu=active.cpu().tolist();tokens_cpu=following.cpu().tolist()
            for i,(is_active,token) in enumerate(zip(active_cpu,tokens_cpu)):
                if is_active:sequences[i].append(token)
            finished=finished|(following==tokenizer.eos_id)
            if bool(finished.all()):break
            mask=torch.cat([mask,active[:,None]],dim=1)
            result=model(following[:,None],attention_mask=mask,past_key_values=past,use_cache=True)
            past=result['past_key_values'];logits=result['logits'][:,-1]
        for i,sequence in enumerate(sequences):
            records.append({'sample_id':start+i,'smiles':tokenizer.decode(sequence),'terminated':sequence[-1]==tokenizer.eos_id,'token_count':len(sequence),'new_tokens':len(sequence)-len(prompt),'prefix':prefix,'seed':seed,'temperature':temperature,'top_k':top_k,'top_p':top_p,'scope':'generated_natural_product_like_candidate_not_verified_natural_occurrence'})
        print(json.dumps({'generated_attempts':len(records),'requested':n}),flush=True)
    metadata={'model_index_sha256':sha256(Path(checkpoint)/'model.index.json'),'model_sha256':index['model_sha256'],'model_metadata':index.get('metadata',{}),'attempts':n,'max_new_tokens':max_new_tokens,'temperature':temperature,'top_k':top_k,'top_p':top_p,'seed':seed,'batch_size':batch_size,'device':str(target),'prefix':prefix,'scope':'All attempted sequences retained; no validity-based resampling or hidden selection'}
    if output_path:
        out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        write_json(out.with_suffix(out.suffix+'.metadata.json'),metadata)
    return records
