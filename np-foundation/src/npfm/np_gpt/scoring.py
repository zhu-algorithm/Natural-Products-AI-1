"""Auditable molecular sequence representations, never a drug efficacy score.

The mean/EOS hidden states describe the *given SMILES representation*. Sequence
likelihood is representation dependent and is not calibrated drug-likeness,
toxicity, biological activity, or a probability of natural occurrence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import load_model, sha256, write_json


@torch.no_grad()
def represent_records(records, checkpoint, output_dir, *, smiles_column="smiles",
                      id_column="compound_id", batch_size=32, device="cpu",
                      cpu_threads=4):
    """Write aligned metrics plus EOS/mean representations, preserving failures.

    Every source row gets a metrics row. Nonempty, in-vocabulary complete
    sequences fitting the context get a row in each embedding matrix. Rejected
    records have embedding_row=null, not a silent truncation or zero vector.
    The caller owns chemistry standardization and must retain its provenance.
    """
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch_size and cpu_threads must be positive")
    records = list(records)
    ids = [str(r[id_column]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Record identifiers must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.set_num_threads(cpu_threads)
    model, tokenizer, index = load_model(checkpoint, device=device)
    metrics, eligible = [], []
    for row, record in enumerate(records):
        item = {"compound_id": ids[row], "input_row": row,
                "embedding_row": None, "status": "pending", "token_count": None,
                "unknown_tokens": None, "token_nll": None, "perplexity": None}
        metrics.append(item)
        try:
            tokens = tokenizer.encode(record[smiles_column])
        except (ValueError, TypeError, KeyError) as exc:
            item.update(status="tokenization_failed", reason=str(exc))
            continue
        item.update(token_count=len(tokens), unknown_tokens=tokens.count(tokenizer.unk_id))
        if len(tokens) > model.config.max_seq_len:
            item.update(status="exceeds_context", reason="Complete sequence excluded; not truncated")
        elif tokenizer.unk_id in tokens:
            item.update(status="out_of_vocabulary", reason="Unknown atom/token excluded from likelihood")
        else:
            item.update(status="scored", embedding_row=len(eligible))
            eligible.append((row, tokens))
    eos = np.empty((len(eligible), model.config.d_model), dtype=np.float32)
    mean = np.empty_like(eos)
    # Sort only computation order; embedding rows always retain input order.
    order = sorted(range(len(eligible)), key=lambda i: len(eligible[i][1]))
    for start in range(0, len(order), batch_size):
        chosen = order[start:start + batch_size]
        length = max(len(eligible[i][1]) for i in chosen)
        batch = torch.full((len(chosen), length), tokenizer.pad_id,
                           dtype=torch.long, device=device)
        lengths = []
        for j, i in enumerate(chosen):
            sequence = eligible[i][1]
            lengths.append(len(sequence))
            batch[j, :len(sequence)] = torch.tensor(sequence, device=device)
        mask = batch.ne(tokenizer.pad_id)
        result = model(batch, attention_mask=mask, output_hidden_states=True)
        losses = F.cross_entropy(result["logits"][:, :-1].transpose(1, 2),
                                 batch[:, 1:], ignore_index=tokenizer.pad_id,
                                 reduction="none")
        hidden = result["hidden_states"]
        for j, i in enumerate(chosen):
            row, _ = eligible[i]
            end = lengths[j]
            nll = float(losses[j, :end - 1].mean().cpu())
            metrics[row].update(token_nll=nll, perplexity=float(np.exp(min(50, nll))))
            eos[i] = hidden[j, end - 1].float().cpu().numpy()
            mean[i] = hidden[j, 1:end - 1].mean(0).float().cpu().numpy()
    np.save(output_dir / "eos_embeddings.npy", eos, allow_pickle=False)
    np.save(output_dir / "mean_embeddings.npy", mean, allow_pickle=False)
    with (output_dir / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["compound_id", "input_row", "embedding_row", "status", "token_count",
                   "unknown_tokens", "token_nll", "perplexity", "reason"]
        w = csv.DictWriter(f, columns); w.writeheader(); w.writerows(metrics)
    manifest = {"format": "np-gpt-representations-v1", "input_records": len(records),
                "scored_records": len(eligible), "embedding_dimension": model.config.d_model,
                "checkpoint_model_sha256": index["model_sha256"],
                "checkpoint_index_sha256": sha256(Path(checkpoint) / "model.index.json"),
                "model_training_metadata": index.get("metadata", {}),
                "input_smiles_column": smiles_column,
                "pooling": {"eos": "Final EOS hidden state, with complete sequence visible",
                            "mean": "Mean molecular token hidden states, excluding BOS/EOS/padding"},
                "limitations": ["Not a validated ADMET, food safety, target or efficacy predictor",
                                "Likelihood depends on the supplied SMILES serialization",
                                "Training-corpus overlap must be assessed separately"],
                "files": {}}
    for name in ["eos_embeddings.npy", "mean_embeddings.npy", "sequence_metrics.csv"]:
        p = output_dir / name
        manifest["files"][name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    write_json(output_dir / "manifest.json", manifest)
    return metrics, manifest


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--id-column", default="compound_id")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    with open(a.input, encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    _, manifest = represent_records(records, a.checkpoint, a.output,
                                   smiles_column=a.smiles_column, id_column=a.id_column,
                                   batch_size=a.batch_size, device=a.device)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
