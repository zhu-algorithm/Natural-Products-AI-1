"""Restore and verify public training row IDs; not needed for model inference."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / 'seen_train_rows_provenance.json').read_text())
    encoded = (root / provenance['encoded_file']).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != provenance['encoded_sha256']:
        raise ValueError('Encoded exposure audit hash mismatch')
    with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode='rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('Exposure audit decompression budget exceeded')
    deltas = np.load(io.BytesIO(raw), allow_pickle=False)
    if deltas.dtype != np.dtype('<u4') or deltas.shape != (provenance['count'],) or not (deltas > 0).all():
        raise ValueError('Exposure audit schema mismatch')
    rows = np.cumsum(deltas, dtype=np.int64)
    output = io.BytesIO()
    np.save(output, rows, allow_pickle=False)
    restored = output.getvalue()
    if hashlib.sha256(restored).hexdigest() != provenance['raw_npy_sha256']:
        raise ValueError('Restored exposure audit hash mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(restored)
    print(json.dumps({'rows': len(rows), 'output': str(args.output), 'sha256': provenance['raw_npy_sha256']}))


if __name__ == '__main__':
    main()
