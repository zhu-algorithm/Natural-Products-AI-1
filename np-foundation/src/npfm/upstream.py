"""Pin and package permitted upstream source snapshots without running them."""
from __future__ import annotations
import argparse,concurrent.futures,datetime,hashlib,io,json,re,subprocess,zipfile
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
FILTER_VERSION='3'
EXTENSIONS={'.py','.pyi','.c','.cc','.cpp','.cxx','.h','.hpp','.cu','.cuh','.f','.f90','.f95','.f03','.f08','.r','.jl','.rs','.go','.java','.js','.ts','.sh','.bash','.md','.rst','.txt','.toml','.yaml','.yml','.cfg','.ini','.cmake','.in','.xml','.m','.sbt','.gradle','.mk'}
EXCLUDED_DIRS={'data','dataset','datasets','checkpoints','weights','model_weights','pretrained','pretrained_models','node_modules','.git','__pycache__','dist','build','notebooks'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def github_repository(url):
    m=re.match(r'https://github\.com/([^/]+)/([^/#?]+)',str(url))
    return '/'.join(m.groups()).removesuffix('.git') if m else None

def permitted_license(value):
    s=str(value).lower()
    if any(x in s for x in ['non-commercial','noncommercial','by-nc','未见','未核实','待核验','定制许可','非商业','学术免费','research only','unknown']):return False
    return bool(re.search(r'(?<![a-z])(?:mit|apache|bsd|gplv?|lgplv?|agplv?|mpl|isc|cc0)(?![a-z])',s))

def records():
    raw=[]
    for name in ['algorithms.json','endpoint_algorithms.json']:
        p=ROOT/'registry'/name
        if not p.exists():continue
        obj=json.loads(p.read_text())
        values=obj if isinstance(obj,list) else next((obj[k] for k in ['rows','algorithms','methods','entries'] if k in obj),[])
        raw.extend(values)
    merged={}
    for r in raw:
        name=r.get('model') or r.get('name') or r.get('algorithm') or r.get('id')
        url=r.get('code_url') or r.get('code') or r.get('repository') or r.get('official_code_or_method_url') or r.get('url') or ''
        if isinstance(url,list):url=next((u for u in url if github_repository(u)),'')
        repo=github_repository(url)
        if not repo:
            for u in r.get('source_urls',[]) or []:
                if github_repository(u):repo=github_repository(u);url=u;break
        key=repo.lower() if repo else str(name).casefold()
        lic=r.get('code_license') or r.get('code_license_review') or r.get('license') or 'unknown'
        if key in merged:
            merged[key]['catalog_aliases'].append(name)
            if permitted_license(lic):merged[key]['license']=lic
            continue
        merged[key]={'model':name,'repository':repo,'code_url':url,'license':lic,'license_evidence_url':r.get('license_evidence_url') or url,'catalog_aliases':[name]}
    return list(merged.values())

def keep_source(name,size):
    p=Path(name);base=p.name.lower();parts=[x.lower() for x in p.parts[1:]]
    legal=any(base.startswith(x) for x in ['license','licence','copying','copyright','notice','authors'])
    if legal:return size<=10*1024**2
    if any(x in EXCLUDED_DIRS for x in parts[:-1]):return False
    if size>10*1024**2:return False
    return p.suffix.lower() in EXTENSIONS or base in {'dockerfile','makefile','cmakelists.txt','requirements','readme','pyproject.toml'}

def snapshot(row,output,max_archive_bytes=160*1024**2):
    result=dict(row)
    if not row['repository']:return {**result,'status':'no_github_source_url'}
    if not permitted_license(row['license']):return {**result,'status':'license_unverified_or_requires_permission'}
    repo=row['repository'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    try:
        rev=subprocess.check_output(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],text=True,timeout=45).split()[0]
        if not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Invalid commit SHA')
        target=out/(repo.replace('/','__')+'__'+rev[:12]+'.source.zip')
        result.update({'commit':rev,'path':str(target.resolve()),'filter_version':FILTER_VERSION})
        side=target.with_suffix('.json')
        if target.exists() and side.exists():
            prior=json.loads(side.read_text())
            if prior.get('filter_version')==FILTER_VERSION and prior.get('sha256')==sha256(target):return {**prior,'status':'cached_verified'}
        url='https://codeload.github.com/'+repo+'/zip/'+rev
        req=Request(url,headers={'User-Agent':'NPFM-source-audit/0.1.1'})
        with urlopen(req,timeout=90) as response:
            raw=response.read(max_archive_bytes+1)
        if len(raw)>max_archive_bytes:raise ValueError('Archive exceeds configured limit')
        if not raw.startswith(b'PK'):raise ValueError('Source endpoint did not return ZIP')
        selected=[];licenses=[];dest=target.with_suffix('.part')
        with zipfile.ZipFile(io.BytesIO(raw)) as source,zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for info in source.infolist():
                if info.is_dir() or '..' in Path(info.filename).parts or not keep_source(info.filename,info.file_size):continue
                content=source.read(info);name=info.filename
                z.writestr(name,content);selected.append(name)
                if any(Path(name).name.lower().startswith(k) for k in ['license','licence','copying','copyright','notice']):licenses.append(name)
        if not licenses:
            dest.unlink(missing_ok=True)
            return {**result,'status':'license_unverified_or_requires_permission','reason':'No license/notice file was present in the retrieved source archive'}
        dest.replace(target)
        result.update({'status':'downloaded_source_snapshot','url':url,'bytes':target.stat().st_size,'sha256':sha256(target),'source_files':len(selected),'license_files':licenses,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Filtered source/documentation snapshot; excludes datasets, weights, notebooks and binary assets. Not an installation or reproduction claim.'})
        side.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return result
    except Exception as exc:
        return {**result,'status':'failed','reason':str(exc)}

def fetch_sources(rows,output,workers=4):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);results=[None]*len(rows)
    def save():
        value={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'filter_version':FILTER_VERSION,'sources':[r for r in results if r is not None]}
        temp=out/'SOURCE_MANIFEST.json.part';temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(out/'SOURCE_MANIFEST.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(snapshot,row,out):i for i,row in enumerate(rows)}
        for future in concurrent.futures.as_completed(futures):
            i=futures[future];results[i]=future.result();save()
            print(json.dumps({'model':results[i]['model'],'status':results[i]['status']},ensure_ascii=False),flush=True)
    return results

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['list','fetch']);p.add_argument('--all-permitted',action='store_true');p.add_argument('--models');p.add_argument('--output',default='third_party/source_snapshots');p.add_argument('--workers',type=int,default=4);a=p.parse_args(argv)
    rows=records()
    if a.models:
        names={s.strip().casefold() for s in a.models.split(',')};rows=[r for r in rows if r['model'].casefold() in names]
    if a.command=='list':print(json.dumps(rows,ensure_ascii=False,indent=2));return 0
    if not a.all_permitted and not a.models:p.error('fetch requires --all-permitted or --models')
    if not 1<=a.workers<=8:p.error('workers must be 1..8')
    results=fetch_sources(rows,a.output,a.workers)
    return int(any(r['status']=='failed' for r in results))

if __name__=='__main__':raise SystemExit(main())
