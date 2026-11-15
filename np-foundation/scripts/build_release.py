"""Package verified code, trained checkpoint, source snapshots and database exports."""
from __future__ import annotations
import argparse,hashlib,json,xml.etree.ElementTree as ET,zipfile
from collections import Counter
from pathlib import Path
from npfm.train import checkpoint_bytes

VERSION='0.1.1'
DATE='2026-11-15'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def portable(value,roots):
    if isinstance(value,dict):return {k:portable(v,roots) for k,v in value.items()}
    if isinstance(value,list):return [portable(v,roots) for v in value]
    if isinstance(value,str):
        for root,name in roots:
            if value==str(root):return name
            if value.startswith(str(root)+'/'):return name+'/'+value[len(str(root))+1:]
    return value

def pack(destination,entries,note):
    names=set()
    temp=destination.with_suffix('.zip.part')
    with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
        for path,name in entries:
            path=Path(path)
            if name in names:raise ValueError('Duplicate ZIP entry: '+name)
            names.add(name)
            if not path.is_file() or path.name.endswith('.part'):raise ValueError('Incomplete input: '+str(path))
            with path.open('rb') as f:magic=f.read(4)
            compressed=magic[:2] in {b'PK',b'\x1f\x8b'} or path.suffix=='.dump'
            z.write(path,name,compress_type=zipfile.ZIP_STORED if compressed else zipfile.ZIP_DEFLATED,compresslevel=None if compressed else 3)
        z.writestr('READ_ME_FIRST.txt',note,compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temp) as z:
        bad=z.testzip()
        if bad:raise ValueError('Archive CRC failure: '+bad)
    temp.replace(destination)
    r={'filename':destination.name,'bytes':destination.stat().st_size,'sha256':digest(destination),'files':len(entries)}
    print(json.dumps(r),flush=True);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--data-root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    project=Path(a.project).resolve();data=Path(a.data_root).resolve();sources=Path(a.source_root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    dm=json.loads((data/'raw/cumulative_manifest.json').read_text());sm=json.loads((sources/'SOURCE_MANIFEST.json').read_text())
    if dm['integrity_problems'] or dm.get('incomplete_files'):raise ValueError('Data integrity gate failed')
    acquired=[r for r in sm['sources'] if r['status'] in {'downloaded_source_snapshot','cached_verified'}]
    for r in acquired:
        if digest(r['path'])!=r['sha256']:raise ValueError('Source snapshot hash mismatch: '+r['model'])
    roots=[(project,'np-foundation'),(data,'data'),(sources,'upstream_sources')]
    reports=project/'reports';reports.mkdir(exist_ok=True)
    for name,obj in [('data_acquisition.json',dm),('source_acquisition.json',sm)]:
        (reports/name).write_text(json.dumps(portable(obj,roots),ensure_ascii=False,indent=2)+'\n')
    tests={k:sum(int(s.get(k,0)) for s in ET.parse(reports/'tests.xml').getroot().iter('testsuite')) for k in ['tests','failures','errors','skipped']}
    if tests['failures'] or tests['errors']:raise ValueError('Tests did not pass')
    ck=project/'checkpoints/joint_pilot'
    manifest=json.loads((ck/'data_manifest.json').read_text());metrics=json.loads((ck/'metrics.json').read_text());baseline=json.loads((ck/'baseline_metrics.json').read_text());states=json.loads((ck/'task_states.json').read_text())
    hashes=json.loads((ck/'checkpoint_hashes.json').read_text())
    for name,expected in hashes.items():
        if hashlib.sha256(checkpoint_bytes(ck/name)).hexdigest()!=expected:raise ValueError('Model integrity gate failed: '+name)
    db=json.loads((project/'registry/databases.json').read_text());alg=json.loads((project/'registry/algorithms.json').read_text());union=json.loads((data/'metadata/coconut_lotus_union.json').read_text())
    summary={'release':VERSION,'date':DATE,'database_registry_entries':len(db['resources']),'main_algorithm_rows':len(alg['rows']),'external_unique_entries':len(sm['sources']),'distinct_algorithms_and_tools_including_project':len(sm['sources'])+1,'upstream_source_snapshots':len(acquired),'source_snapshot_status_counts':dict(Counter(r['status'] for r in sm['sources'])),'downloaded_source_files':dm['total_files'],'downloaded_source_bytes':dm['total_bytes'],'database_status_counts':dm['resource_status_counts'],'coconut_lotus_union':union,'tests':tests,'joint_training':manifest,'joint_test':metrics['test'],'same_split_baselines':baseline,'trained_endpoints':[s['name'] for s in states if s.get('trained')],'disease_model_status':'Evidence retrieval only; no trained or calibrated disease efficacy predictor','release_note':'Recovered code was independently retrained as model format 2; all reported metrics derive from the current checkpoint.'}
    (reports/'release_summary.json').write_text(json.dumps(portable(summary,roots),ensure_ascii=False,indent=2)+'\n')
    m=metrics['test']['tasks']['logS'];rf=baseline['morgan_random_forest'];mean=baseline['constant_training_mean']
    report=f'''# 交付状态：NP-FM {VERSION}（{DATE}）

## 实际资源

- {len(db['resources'])}个数据库/辅助资源注册条目，另有COCONUT的71个上游collection目录。
- {len(sm['sources'])}个去重外部方法/工具条目，加本项目共{len(sm['sources'])+1}项。
- {len(acquired)}个固定Git commit的第三方源码快照；保留原许可证，不含其权重与数据。
- 重新取得{dm['total_files']}个原始文件，共{dm['total_bytes']:,}字节；完整性复核通过。
- COCONUT与LOTUS联合表{union['union_full_keys']:,}个完整InChIKey结构条目；不代表全球全部天然化合物。
- {tests['tests']}项测试通过；对36个真实PubChem示例完成统一分析。

“选定导出齐全”只涵盖注册表指定文件，不代表某数据库全部API内容。原始文件数、来源发生记录数、精确结构数分别统计。多个库不能直接累加。

## 当前实际训练权重

联合语料为2,048条COCONUT结构和1,128条ESOL实测溶解度记录，共{manifest['used_records']:,}行；{manifest['parameter_count']:,}个参数。序列、图编码器与融合层均接受真实优化。唯一监督端点为logS，单位log10 mol/L。

采用身份/骨架分组；词表和监督标准化仅用训练部分；验证集选模后才评估测试集。测试集有{m['n_labels']}条实测标签。

| 方法 | MAE | RMSE |
|---|---:|---:|
| 联合Transformer与分子图模型 | {m['mae']:.4f} | {m['rmse']:.4f} |
| 同拆分Morgan指纹+随机森林 | {rf['mae']:.4f} | {rf['rmse']:.4f} |
| 训练集均值 | {mean['mae']:.4f} | {mean['rmse']:.4f} |

这是单次小规模工程基准，不是天然产物专属外部验证，不证明临床有效性或最先进性能。详细拆分、配置、哈希、来源与指标随检查点保存。一般小分子溶解度不能代表完整成药性。

## 尚需继续的研究

完整联合结构包已取得，但本权重没有在全部结构上完成大规模预训练。疾病适应症当前为17条分层示例证据的可追溯检索；没有可靠训练和校准的疗效预测头。外部目录项目并未全部安装或复现。可选ADMET-AI/Vina适配器未在本次环境进行完整集成验证；Dockerfile未构建验证。

## 文件与许可

主数据库来源在 `reports/data_acquisition.json`，源码来源在 `reports/source_acquisition.json`，综合统计在 `reports/release_summary.json`。数据包另含本次ESOL原始测量文件、联合试验训练表与采样记录，供复核。原创代码MIT不改变第三方代码、数据、权重各自条款。
'''
    (project/'docs/DELIVERY_STATUS.md').write_text(report)
    result=[];ignored={'.git','.venv','__pycache__','.pytest_cache','data','runs','third_party','build','dist'}
    code=[]
    for file in sorted(project.rglob('*')):
        rel=file.relative_to(project)
        if not file.is_file() or any(x in ignored or x.endswith('.egg-info') for x in rel.parts) or file.name.endswith(('.pyc','.part')):continue
        code.append((file,'np-foundation/'+rel.as_posix()))
    result.append(pack(out/f'Natural_Product_AI_Model_v{VERSION}.zip',code,'Original research code, examples, tests, provenance and an actually trained pilot checkpoint. Read np-foundation/README.md and docs/DELIVERY_STATUS.md. The current trained endpoint is logS. Disease output is cited evidence retrieval. Dependencies must be installed.'))
    source_entries=[(Path(r['path']),'sources/'+Path(r['path']).name) for r in acquired]
    source_entries +=[(reports/'source_acquisition.json','SOURCE_MANIFEST.json'),(project/'registry/algorithms.json','algorithms.json'),(project/'registry/endpoint_algorithms.json','endpoint_algorithms.json')]
    result.append(pack(out/'Natural_Product_AI_Algorithm_Sources.zip',source_entries,'Fixed-commit filtered source and documentation snapshots, each with original licenses and notices. No datasets, upstream weights, notebooks or general binary assets. GPL/LGPL/AGPL are not relicensed as MIT. Inclusion is not a claim of installation, successful reproduction or integration.'))
    open_ids={'coconut','lotus','mibig','chebi','drugcentral'}
    for label,is_open in [('Open',True),('Research_Restricted',False)]:
        entries=[]
        for row in dm['files']:
            if (row['resource_id'] in open_ids)!=is_open:continue
            f=Path(row['path']);name='raw/'+row['relative_path'];entries.append((f,name))
            side=f.with_name(f.name+'.provenance.json')
            if side.exists():entries.append((side,name+'.provenance.json'))
        for f in sorted((data/'normalized').glob('*')):
            if f.is_file() and f.name.startswith(('coconut','lotus','chebi','drugcentral'))==is_open:entries.append((f,'normalized/'+f.name))
        for f in sorted((data/'metadata').glob('*')):
            if not f.is_file() or f.suffix=='.html' or f.name.endswith('.part'):continue
            if f.name.startswith('npass') and is_open:continue
            if f.name.startswith(('coconut','chebi')) and not is_open:continue
            entries.append((f,'metadata/'+f.name))
        if is_open:
            for f in sorted((project/'data/training').glob('*')):
                if f.is_file():entries.append((f,'benchmarks/'+f.name))
        entries +=[(reports/'data_acquisition.json','DATA_MANIFEST.json'),(project/'registry/databases.json','databases.json'),(project/'docs/DATABASES.md','DATABASES.md'),(project/'docs/DATA_AND_MODEL_BOUNDARIES.md','DATA_AND_MODEL_BOUNDARIES.md')]
        note=('Open exports under their named source terms; attribution/share-alike obligations still apply. ' if is_open else 'RESEARCH RESTRICTED / MIXED TERMS: includes noncommercial, no-derivatives and unverified-redistribution datasets. Do not infer commercial or public redistribution permission. ')
        note+='Exact source URLs, source licenses, raw releases and SHA256 sidecars are retained. File/occurrence/structure counts differ. Missing evidence is not negative evidence; calculated descriptors are not experimental labels.'
        result.append(pack(out/f'Natural_Product_Databases_{label}.zip',entries,note))
    extra=[]
    for f in sorted(out.glob('*.xlsx')):extra.append({'filename':f.name,'bytes':f.stat().st_size,'sha256':digest(f)})
    (out/'PACKAGE_MANIFEST.json').write_text(json.dumps({'release':VERSION,'date':DATE,'packages':result,'companions':extra},indent=2)+'\n')
    (out/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['filename']}\n" for r in result+extra))
    print(json.dumps({'packages':result,'companions':extra},indent=2),flush=True)

if __name__=='__main__':main()
