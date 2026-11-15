"""End-to-end invariants for corpus boundaries, resume, shard integrity and KV generation."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from npfm.tokenizer import SmilesTokenizer
from npfm.np_gpt import NPTransformer, NPTransformerConfig
from npfm.np_gpt.runtime import load_model, save_model, sha256, safe_path
from npfm.np_gpt.training import SequenceCorpus, train_from_config
from npfm.np_gpt.generation import generate


def make_corpus(path):
    path.mkdir()
    strings={"train":["CC","CO","CN","CCC","CCO","CCN","C=O","C#N","O","N"],
             "val":["CCCO","CCCN","COC"],"test":["CCCC","CNCO","NCO"]}
    tokenizer=SmilesTokenizer.fit(strings["train"])
    tokenizer.save(path/'tokenizer.json')
    manifest={"schema_version":"1.0","token_dtype":"uint16","max_seq_length":16,
              "tokenizer":{"path":"tokenizer.json","sha256":sha256(path/'tokenizer.json')},
              "splits":{},"files":{}}
    for split,smiles in strings.items():
        sequences=[tokenizer.encode(s) for s in smiles]
        offsets=np.array([0]+list(np.cumsum([len(s) for s in sequences])),dtype=np.uint64)
        tokens=np.array([t for sequence in sequences for t in sequence],dtype='<u2')
        names={"tokens_path":f'{split}.tokens.bin',"offsets_path":f'{split}.offsets.npy',"row_ids_path":f'{split}.rows.npy'}
        tokens.tofile(path/names['tokens_path'])
        np.save(path/names['offsets_path'],offsets,allow_pickle=False)
        np.save(path/names['row_ids_path'],np.arange(len(smiles),dtype=np.int64),allow_pickle=False)
        for name in names.values():manifest['files'][name]={"bytes":(path/name).stat().st_size,"sha256":sha256(path/name)}
        manifest['splits'][split]={**names,"tokens":len(tokens),"records":len(smiles)}
    (path/'manifest.json').write_text(json.dumps(manifest))
    return tokenizer,strings


def config_file(tmp_path,corpus,output):
    config={"corpus_dir":str(corpus),"output_dir":str(output),"seed":17,"cpu_threads":1,
            "device":"cpu","precision":"fp32","epochs":2,"batch_size":2,
            "max_tokens_per_batch":32,"gradient_accumulation_steps":2,"learning_rate":.002,
            "warmup_steps":0,"log_every_steps":1000,"save_every_steps":1,
            "model":{"d_model":16,"n_layers":1,"n_heads":2,"n_kv_heads":1,"ffn_hidden_dim":32,"dropout":.15}}
    path=tmp_path/(output.name+'.json');path.write_text(json.dumps(config));return path


def test_sequence_batches_never_predict_across_molecule_boundaries(tmp_path):
    path=tmp_path/'corpus';tokenizer,strings=make_corpus(path)
    data=SequenceCorpus(path,'train')
    indices=np.array([0,3,8])  # Deliberately unequal molecule lengths.
    x,y,count=data.batch(indices,tokenizer.pad_id,'cpu')
    expected_count=0
    for row,index in enumerate(indices):
        sequence=tokenizer.encode(strings['train'][index]);n=len(sequence)-1;expected_count+=n
        assert x[row,:n].tolist()==sequence[:-1]
        assert y[row,:n].tolist()==sequence[1:]
        assert y[row,n-1].item()==tokenizer.eos_id
        assert y[row,n:].eq(tokenizer.pad_id).all()
    assert count==expected_count


def test_exact_resume_matches_uninterrupted_optimizer_and_selected_weights(tmp_path):
    corpus=tmp_path/'corpus';make_corpus(corpus)
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    full_report=train_from_config(config_file(tmp_path,corpus,full))
    resumed_config=config_file(tmp_path,corpus,resumed)
    partial=train_from_config(resumed_config,{"max_steps":1})
    assert partial['selected_checkpoint']['provisional'] is True
    report=train_from_config(resumed_config,{"resume":str(resumed/'resume.pt')})
    a=torch.load(full/'resume.pt',weights_only=True)
    b=torch.load(resumed/'resume.pt',weights_only=True)
    assert a['step']==b['step'] and a['train_tokens']==b['train_tokens']
    assert torch.equal(a['seen_indices'],b['seen_indices'])
    for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    for key,values in a['optimizer']['state'].items():
        for name,value in values.items():
            other=b['optimizer']['state'][key][name]
            if torch.is_tensor(value):torch.testing.assert_close(value,other,rtol=0,atol=0)
            else:assert value==other
    full_model,_,_=load_model(full/'best');resumed_model,_,_=load_model(resumed/'best')
    for name,value in full_model.state_dict().items():torch.testing.assert_close(value,resumed_model.state_dict()[name],rtol=0,atol=0)
    assert full_report['test']==report['test']
    assert report['selected_checkpoint']['provisional'] is False
    with pytest.raises(ValueError,match='same output directory'):
        train_from_config(resumed_config,{"resume":str(resumed/'resume.pt'),"output_dir":str(tmp_path/'other')})
    with pytest.raises(ValueError,match='settings changed'):
        train_from_config(resumed_config,{"resume":str(resumed/'resume.pt'),"max_grad_norm":.5})
    resume_file=resumed/'resume.pt';resume_file.write_bytes(resume_file.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='Resume checkpoint hash mismatch'):
        train_from_config(resumed_config,{"resume":str(resume_file)})


def test_weight_and_tokenizer_tampering_and_symlink_escape_are_rejected(tmp_path):
    tokenizer=SmilesTokenizer.fit(['CC','CO'])
    model=NPTransformer(NPTransformerConfig(vocab_size=len(tokenizer),d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden_dim=32,max_seq_len=16))
    checkpoint=tmp_path/'model';index=save_model(model,tokenizer,checkpoint,part_bytes=4096)
    loaded,_,_=load_model(checkpoint)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[name])
    first=checkpoint/index['parts'][0]['path'];raw=first.read_bytes();first.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='shard hash'):load_model(checkpoint)
    first.write_bytes(raw)
    tok=checkpoint/'tokenizer.json';tok.write_bytes(tok.read_bytes()+b' ')
    with pytest.raises(ValueError,match='metadata hash'):load_model(checkpoint)
    outside=tmp_path/'outside.bin';outside.write_bytes(b'outside')
    (checkpoint/'linked.bin').symlink_to(outside)
    with pytest.raises(ValueError,match='inside'):safe_path(checkpoint,'linked.bin')


def test_generation_uses_incremental_kv_cache_and_records_unterminated_attempts(tmp_path,monkeypatch):
    tokenizer=SmilesTokenizer.fit(['C','O'])
    model=NPTransformer(NPTransformerConfig(vocab_size=len(tokenizer),d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden_dim=32,max_seq_len=16))
    # Deterministic test-only weights always choose C after banned special tokens.
    with torch.no_grad():
        for layer in model.layers:
            for module in layer.modules():
                if isinstance(module,torch.nn.Linear):module.weight.zero_()
        model.token_embedding.weight.zero_()
        model.token_embedding.weight[tokenizer.bos_id].fill_(2.)
        model.token_embedding.weight[tokenizer.token_to_id['C']].fill_(1.)
    checkpoint=tmp_path/'model';save_model(model,tokenizer,checkpoint)
    calls=[];original=NPTransformer.forward
    def observe(self,input_ids,*args,**kwargs):
        calls.append((input_ids.shape[1],kwargs.get('past_key_values') is not None))
        return original(self,input_ids,*args,**kwargs)
    monkeypatch.setattr(NPTransformer,'forward',observe)
    records=generate(checkpoint,n=3,max_new_tokens=4,top_k=1,top_p=1.,batch_size=3,cpu_threads=1)
    assert calls[0]==(1,False) and all(length==1 and cached for length,cached in calls[1:])
    assert len(calls)>1 and len(records)==3
    assert all(r['smiles']=='CCCC' and not r['terminated'] and r['new_tokens']==4 for r in records)
