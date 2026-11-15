"""Memory-mapped causal pretraining with exact exposure accounting and resume."""
from __future__ import annotations
import contextlib,copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from .model import NPTransformer,NPTransformerConfig
from .runtime import save_model,sha256,write_json,safe_path
from ..tokenizer import SmilesTokenizer

class SequenceCorpus:
    def __init__(self,directory,split,verify=True):
        if split not in {'train','val','test'}:raise ValueError('Unknown split')
        self.directory=Path(directory);self.manifest=json.loads((self.directory/'manifest.json').read_text());self.split=split
        if str(self.manifest.get('schema_version'))!='1.0':raise ValueError('Unsupported corpus schema')
        self.spec=self.manifest['splits'][split]
        names=[self.spec[k] for k in ['tokens_path','offsets_path','row_ids_path']]
        if verify:
            for name in names:
                expected=self.manifest['files'][name]
                path=safe_path(self.directory,name)
                if path.stat().st_size!=expected['bytes'] or sha256(path)!=expected['sha256']:raise ValueError('Corpus file hash mismatch: '+name)
        dtype=np.dtype(self.manifest['token_dtype']).newbyteorder('<')
        self.tokens=np.memmap(safe_path(self.directory,names[0]),dtype=dtype,mode='r') if self.spec['tokens'] else np.array([],dtype=dtype)
        self.offsets=np.load(safe_path(self.directory,names[1]),mmap_mode='r',allow_pickle=False)
        self.row_ids=np.load(safe_path(self.directory,names[2]),mmap_mode='r',allow_pickle=False)
        if len(self.offsets)!=len(self.row_ids)+1 or int(self.offsets[-1])!=len(self.tokens) or int(self.offsets[0])!=0:raise ValueError('Invalid corpus offsets')
        self.lengths=np.diff(self.offsets)
        if np.any(self.lengths<2) or np.any(self.lengths>self.manifest['max_seq_length']):raise ValueError('Out-of-budget sequence in token stream')

    def __len__(self):return len(self.row_ids)

    def batch(self,indices,pad_id,device):
        width=int(max(self.lengths[indices]));array=np.full((len(indices),width),pad_id,dtype=np.int64)
        for j,i in enumerate(indices):array[j,:self.lengths[i]]=self.tokens[self.offsets[i]:self.offsets[i+1]]
        ids=torch.as_tensor(array,device=device)
        return ids[:,:-1],ids[:,1:],int(sum(self.lengths[indices]-1))

def select_indices(count,limit,seed):
    if limit is not None and (isinstance(limit,bool) or int(limit)<1):raise ValueError('Record limits must be positive or null')
    if limit is None or limit>=count:return np.arange(count,dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count,int(limit),replace=False)).astype(np.int64)

def batch_indices(lengths,selected,batch_size,max_tokens,seed,shuffle=True):
    """Equal-length batches avoid padding waste; no molecule is dropped."""
    if batch_size<1 or max_tokens<2:raise ValueError('Invalid batch/token budget')
    rng=np.random.default_rng(seed);groups=[]
    order=selected[np.argsort(lengths[selected],kind='stable')]
    boundaries=np.flatnonzero(np.diff(lengths[order]))+1
    for group in np.split(order,boundaries):
        if not len(group):continue
        group=group.copy()
        if shuffle:rng.shuffle(group)
        width=int(lengths[group[0]]);size=min(batch_size,max(1,max_tokens//width))
        groups.extend(group[i:i+size] for i in range(0,len(group),size))
    if shuffle:rng.shuffle(groups)
    return groups

def _precision(device,precision):
    if precision=='fp32':return contextlib.nullcontext
    if device.type!='cuda':raise ValueError('Reduced-precision profiles require CUDA; use fp32 on CPU')
    if precision=='bf16':
        if not torch.cuda.is_bf16_supported():raise ValueError('GPU does not support bfloat16')
        return lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    if precision=='fp16':return lambda:torch.autocast('cuda',dtype=torch.float16)
    raise ValueError('precision must be fp32, bf16 or fp16')

@torch.no_grad()
def evaluate_language(model,corpus,indices,batch_size,max_tokens,device,autocast):
    if not len(indices):raise ValueError('Evaluation partition is empty')
    model.eval();total=0.;tokens=0
    for ids in batch_indices(corpus.lengths,indices,batch_size,max_tokens,0,shuffle=False):
        x,y,n=corpus.batch(ids,model.config.pad_token_id,device)
        with autocast():loss=model(x,targets=y)['loss']
        total+=float(loss)*n;tokens+=n
    nll=total/tokens
    return {'records':len(indices),'prediction_tokens':tokens,'token_nll':nll,'perplexity':math.exp(min(nll,50)),'perplexity_capped_at_exp_50':nll>50}

def _save_resume(path,model,optimizer,scaler,state):
    data={**state,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()}
    if torch.cuda.is_available():data['cuda_rng']=torch.cuda.get_rng_state_all()
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
    write_json(path.with_suffix('.sha256.json'),{'sha256':sha256(path)})

def train_from_config(config_path,overrides=None):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text())
    config.update({k:v for k,v in (overrides or {}).items() if v is not None})
    for key in ['corpus_dir','output_dir','resume']:
        if config.get(key):
            p=Path(config[key]);config[key]=str((p if p.is_absolute() else config_path.parent/p).resolve())
    seed=int(config.get('seed',42));torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    torch.set_num_threads(int(config.get('cpu_threads',4)))
    device=torch.device(config.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; choose the explicit CPU profile')
    precision=config.get('precision','fp32');autocast=_precision(device,precision)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    datasets={s:SequenceCorpus(config['corpus_dir'],s) for s in ['train','val','test']}
    corpus=datasets['train'];manifest=corpus.manifest;corpus_hash=sha256(corpus.directory/'manifest.json')
    tokenpath=safe_path(corpus.directory,manifest['tokenizer']['path'])
    if sha256(tokenpath)!=manifest['tokenizer']['sha256']:raise ValueError('Tokenizer hash mismatch')
    tokenizer=SmilesTokenizer.load(tokenpath)
    args=copy.deepcopy(config.get('model',{}));args.update(vocab_size=len(tokenizer),max_seq_len=int(manifest['max_seq_length']),pad_token_id=tokenizer.pad_id)
    model_config=NPTransformerConfig.from_dict(args);model=NPTransformer(model_config).to(device)
    batch_size=int(config.get('batch_size',64));max_tokens=int(config.get('max_tokens_per_batch',8192));accum=int(config.get('gradient_accumulation_steps',1));epochs=int(config.get('epochs',1))
    if accum<1 or epochs<1:raise ValueError('epochs and gradient_accumulation_steps must be positive')
    lr=float(config.get('learning_rate',.001));decay=float(config.get('weight_decay',.01))
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(decay) or decay<0:raise ValueError('Invalid optimizer parameters')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay,betas=(.9,.95))
    selected={s:select_indices(len(datasets[s]),config.get('max_'+s+'_records'),seed+i) for i,s in enumerate(['train','val','test'])}
    if any(not len(v) for v in selected.values()):raise ValueError('All training/evaluation partitions must be nonempty')
    per_epoch=len(batch_indices(corpus.lengths,selected['train'],batch_size,max_tokens,seed))
    planned_steps=math.ceil(per_epoch/accum)*epochs;max_steps=config.get('max_steps');max_steps=int(max_steps) if max_steps is not None else None
    if max_steps is not None and max_steps<1:raise ValueError('max_steps must be positive')
    schedule_steps=int(config.get('schedule_steps',planned_steps));warmup=int(config.get('warmup_steps',min(200,planned_steps//10)))
    if schedule_steps<1 or warmup<0:raise ValueError('Invalid learning-rate schedule')
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    max_grad_norm=float(config.get('max_grad_norm',1.))
    if not math.isfinite(max_grad_norm) or max_grad_norm<=0:raise ValueError('max_grad_norm must be positive and finite')
    fingerprint={'corpus_manifest_sha256':corpus_hash,'model_config':model_config.to_dict(),'seed':seed,'batch_size':batch_size,'max_tokens_per_batch':max_tokens,'gradient_accumulation_steps':accum,'selected_counts':{s:len(v) for s,v in selected.items()},'epochs':epochs,'learning_rate':lr,'weight_decay':decay,'schedule_steps':schedule_steps,'warmup_steps':warmup,'precision':precision,'max_grad_norm':max_grad_norm,'cpu_threads':torch.get_num_threads(),'device':str(device),'torch_version':str(torch.__version__)}
    step=0;start_epoch=0;next_batch=0;best=float('inf');seen=np.zeros(len(corpus),dtype=np.bool_);train_tokens=0;history=[]
    if config.get('resume'):
        path=Path(config['resume']);expected=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
        if path.resolve().parent!=output.resolve():raise ValueError('Exact resume requires the same output directory as resume.pt')
        if sha256(path)!=expected:raise ValueError('Resume checkpoint hash mismatch')
        state=torch.load(path,map_location=device,weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('Resume data/model/batch/schedule settings changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step=state['step'];start_epoch=state['epoch'];next_batch=state['next_batch'];best=state['best'];history=state['history'];train_tokens=state['train_tokens']
        seen[state['seen_indices'].cpu().numpy()]=True;torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng'])
        if device.type=='cuda' and 'cuda_rng' in state:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    elif (output/'resume.pt').exists():raise ValueError('Output already has training state; explicitly resume or choose a new output directory')
    write_json(output/'training_config.json',config);write_json(output/'corpus_reference.json',{'manifest_sha256':corpus_hash,'manifest':manifest})
    start=time.monotonic();initial_step=step;log_every=int(config.get('log_every_steps',100));save_every=int(config.get('save_every_steps',1000));stop=False
    if log_every<1 or save_every<1:raise ValueError('Log/checkpoint intervals must be positive')
    def state_for(epoch,batch):
        return {'fingerprint':fingerprint,'step':step,'epoch':epoch,'next_batch':batch,'best':best,'history':history,'train_tokens':train_tokens,'seen_indices':torch.from_numpy(np.flatnonzero(seen))}
    for epoch in range(start_epoch,epochs):
        batches=batch_indices(corpus.lengths,selected['train'],batch_size,max_tokens,seed+epoch)
        offset=next_batch if epoch==start_epoch else 0
        if offset>len(batches):raise ValueError('Resume batch cursor exceeds this epoch')
        for begin in range(offset,len(batches),accum):
            if max_steps is not None and step>=max_steps:stop=True;break
            group=batches[begin:begin+accum];group_tokens=sum(int(sum(corpus.lengths[g]-1)) for g in group)
            factor=min(1.,(step+1)/max(warmup,1)) if step<warmup else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,schedule_steps-warmup))))
            for param_group in optimizer.param_groups:param_group['lr']=lr*factor
            model.train();optimizer.zero_grad(set_to_none=True);loss_value=0.
            for ids in group:
                x,y,n=corpus.batch(ids,tokenizer.pad_id,device)
                with autocast():loss=model(x,targets=y)['loss']
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss*(n/group_tokens)).backward();loss_value+=float(loss.detach())*n/group_tokens
                seen[ids]=True;train_tokens+=n
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            scaler.step(optimizer);scaler.update();step+=1;next_batch=begin+len(group)
            if step%log_every==0 or step==1:
                progress={'epoch':epoch+1,'step':step,'planned_steps':planned_steps,'loss':loss_value,'lr':lr*factor,'unique_train_records_seen':int(seen.sum()),'prediction_tokens_seen':train_tokens,'elapsed_seconds':time.monotonic()-start}
                print(json.dumps(progress),flush=True);write_json(output/'progress.json',progress)
            if step%save_every==0:_save_resume(output/'resume.pt',model,optimizer,scaler,state_for(epoch,next_batch))
        completed=next_batch>=len(batches)
        val=evaluate_language(model,datasets['val'],selected['val'],batch_size,max_tokens,device,autocast)
        record={'epoch':epoch+1,'step':step,'epoch_complete':completed,'validation':val,'eligible_for_completed_epoch_selection':completed};history.append(record)
        # A user-requested mid-epoch stop must not change the completed-epoch
        # selection protocol after exact resume. Expose a provisional smoke
        # checkpoint only when no completed-epoch checkpoint exists yet.
        if (completed and val['token_nll']<best) or (not completed and not math.isfinite(best)):
            if completed:best=val['token_nll']
            metadata={'scope':'structural_language_model_only','selected_epoch':epoch+1,'selected_step':step,'selection':'validation_token_nll' if completed else 'provisional_partial_epoch_snapshot','provisional':not completed,'corpus_manifest_sha256':corpus_hash,'unique_train_records_seen':int(seen.sum()),'prediction_tokens_seen':train_tokens,'trained_property_tasks':[]}
            save_model(model,tokenizer,output/'best',metadata)
            np.save(output/'best/seen_train_row_ids.npy',corpus.row_ids[np.flatnonzero(seen)],allow_pickle=False)
        _save_resume(output/'resume.pt',model,optimizer,scaler,state_for(epoch+1 if completed else epoch,0 if completed else next_batch))
        print(json.dumps(record),flush=True)
        next_batch=0
        if stop or (max_steps is not None and step>=max_steps):break
    from .runtime import load_model
    chosen,_,index=load_model(output/'best',device=device)
    test=evaluate_language(chosen,datasets['test'],selected['test'],batch_size,max_tokens,device,autocast)
    report={'format':'np-gpt-training-v1','parameter_count':chosen.num_parameters(),'corpus_manifest_sha256':corpus_hash,'eligible_records':{s:len(datasets[s]) for s in datasets},'selected_records':{s:len(v) for s,v in selected.items()},'unique_train_records_seen_in_run':int(seen.sum()),'training_prediction_tokens_seen':train_tokens,'optimizer_steps':step,'history':history,'selected_checkpoint':index['metadata'],'test':test,'elapsed_seconds_this_invocation':time.monotonic()-start,'device':str(device),'torch':torch.__version__,'limitations':['Molecular language modeling does not establish natural occurrence, synthesis, ADMET or clinical efficacy.','Identity/connectivity split is not scaffold-disjoint evaluation.','Selected checkpoint used validation only; test evaluated afterwards.']}
    write_json(output/'training_report.json',report);write_json(output/'best/training_report.json',report)
    print(json.dumps({'complete':True,'checkpoint':str(output/'best'),'report':str(output/'training_report.json'),'test':test}),flush=True)
    return report
