"""Verify a sha256sum-format file in its own directory."""
from pathlib import Path
import argparse,hashlib

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();m=Path(a.manifest)
    for line in m.read_text().splitlines():
        if not line.strip():continue
        expected,name=line.split(None,1);path=m.parent/name.strip().lstrip('*');h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=expected:raise SystemExit('FAILED: '+str(path))
        print('OK:',path.name)

if __name__=='__main__':main()
