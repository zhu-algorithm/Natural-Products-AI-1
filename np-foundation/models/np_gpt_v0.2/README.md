# NP-GPT v0.2 published checkpoint

This directory contains the one-epoch, 331,680-parameter structural language model described in [the model card](../../docs/NP_GPT_MODEL_CARD.md). It was trained from random initialization on 688,166 eligible sequences from the documented public COCONUT/LOTUS corpus. There are no trained ADMET, efficacy or disease heads.

## Load and generate

Run from the `np-foundation` module root after installing `.[train]`:

```bash
npfm np-gpt generate --checkpoint models/np_gpt_v0.2 --n 16 --seed 42 --cpu-threads 2 --output runs/generated.jsonl
npfm np-gpt score --checkpoint models/np_gpt_v0.2 --input examples/compounds.csv --output runs/representations --cpu-threads 2
```

`model.index.json` verifies six weight shards plus the model configuration and tokenizer. The concatenated weight SHA256 is `b6818cbe086401a22617072ab8d605500cfa64862ce59a561ad19aa5a47c5592`. The loader concatenates in memory and uses weights-only loading; manual concatenation is unnecessary. There is no duplicate complete `model.pt` in this directory.

## Evidence included

- `training_report.json`: actual one-epoch training, validation-selected checkpoint and 4,096-record test evaluation.
- `token_baselines.json` and `evaluation_subsets.json`: train-only unigram/bigram baselines and exact public validation/test row IDs.
- `corpus_manifest.json`: exact preparation manifest; full token streams and SQLite remain in the separately prepared public corpus.
- `training_config.json`: original hyperparameters with only input/output paths made relative and portable.
- `generated_512.jsonl`, its metadata and `generation_evaluation.json`: every attempt in the fixed reported batch and the evaluation with explicit denominators.
- `seen_train_row_id_deltas.npy.gz`: losslessly compressed public input row IDs actually consumed by training; not model parameters and not required for inference.
- `release_manifest.json`: file sizes, hashes and source provenance.

## Restore the exposure audit

The original 5,505,456-byte int64 NPY is represented by a 78,136-byte gzip-compressed array of successive positive uint32 differences. The first preceding value is zero; an int64 cumulative sum restores the original row IDs. These IDs identify one-based data rows in the **public** source CSV, not chemical identifiers from a user upload.

```bash
python models/np_gpt_v0.2/restore_seen_rows.py --output runs/seen_train_row_ids.npy
```

The script validates the encoded hash, bounded decompression, dimensions, increasing row IDs, and the reconstructed original NPY hash. It does not overwrite an existing output. This audit file cannot replace optimizer/RNG/cursor state for exact training resume; `resume.pt` is not distributed here.

NLL, perplexity and hidden-state proximity are properties of the supplied SMILES serialization. Generation validity and absence from this snapshot do not prove natural occurrence, synthesis, druggability or clinical benefit. See [attribution and source terms](ATTRIBUTION.md).
