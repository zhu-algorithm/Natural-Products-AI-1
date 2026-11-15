"""Train-only unigram and first-order token baselines for language evaluation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .training import SequenceCorpus, select_indices
from .runtime import write_json, sha256


def evaluate_baselines(corpus_dir, *, split="test", limit=4096, seed=44,
                       smoothing=0.1, output_path=None):
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Smoothing must be finite and positive")
    train = SequenceCorpus(corpus_dir, "train")
    evaluation = SequenceCorpus(corpus_dir, split)
    vocab = train.manifest["tokenizer"]["vocab_size"]
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    # Corpus streams contain complete BOS...EOS strings. Exclude the only
    # inter-molecule adjacency EOS -> BOS instead of treating it as chemistry.
    tokens = train.tokens
    for start in range(0, max(0, len(tokens)-1), 1_000_000):
        end = min(start + 1_000_000, len(tokens)-1)
        x = np.asarray(tokens[start:end], dtype=np.int64)
        y = np.asarray(tokens[start+1:end+1], dtype=np.int64)
        keep = (x != 2) & (y != 1) & (y != 0)
        counts += np.bincount(x[keep]*vocab+y[keep], minlength=vocab*vocab).reshape(vocab, vocab)
    expected = int(train.spec["tokens"] - train.spec["records"])
    if int(counts.sum()) != expected:
        raise ValueError("Special-token boundaries differ from the corpus contract")
    unigram = counts.sum(0).astype(float) + smoothing
    unigram /= unigram.sum()
    bigram = counts.astype(float) + smoothing
    bigram /= bigram.sum(1, keepdims=True)
    selected = select_indices(len(evaluation), limit, seed)
    if not len(selected):
        raise ValueError("Evaluation partition is empty")
    total_uni = total_bi = 0.0
    total_tokens = 0
    for i in selected:
        seq = np.asarray(evaluation.tokens[evaluation.offsets[i]:evaluation.offsets[i+1]],dtype=np.int64)
        total_uni -= float(np.log(unigram[seq[1:]]).sum())
        total_bi -= float(np.log(bigram[seq[:-1], seq[1:]]).sum())
        total_tokens += len(seq)-1
    result = {"format":"np-gpt-token-baselines-v1",
              "training_records":len(train),"training_prediction_tokens":expected,
              "corpus_manifest_sha256":sha256(Path(corpus_dir)/"manifest.json"),
              "evaluation_split":split,"selected_records":len(selected),
              "selection_seed":seed,"prediction_tokens":total_tokens,"smoothing":smoothing,
              "unigram":{"token_nll":total_uni/total_tokens,"perplexity":math.exp(total_uni/total_tokens)},
              "bigram":{"token_nll":total_bi/total_tokens,"perplexity":math.exp(total_bi/total_tokens)},
              "scope":"Same token likelihood task; not a medicinal chemistry or ADMET baseline"}
    if output_path:write_json(output_path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",required=True);p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=4096);p.add_argument("--seed",type=int,default=44)
    a=p.parse_args()
    print(json.dumps(evaluate_baselines(a.corpus,limit=a.limit,seed=a.seed,output_path=a.output),indent=2))


if __name__ == "__main__":main()
