"""Inference gates: only supervised heads with recorded optimizer updates are exposed."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from .features import canonical_smiles, rdkit_available
from .model import NaturalProductModel
from .tokenizer import SmilesTokenizer
from .train import MODEL_DEFAULTS, read_rows, tensor_batch, verify_checkpoint, _write_json


def load_checkpoint(checkpoint_dir):
    directory = Path(checkpoint_dir)
    verify_checkpoint(directory)
    config = json.loads((directory / "resolved_config.json").read_text())
    states = json.loads((directory / "task_states.json").read_text())
    tokenizer = SmilesTokenizer.load(directory / "vocab.json")
    if config.get("model_format_version") != 2:
        raise ValueError("Unsupported checkpoint architecture. Recovered source uses model_format_version=2; do not combine old weights with new metadata.")
    if config["use_graph"] and not rdkit_available():
        raise RuntimeError("This checkpoint requires RDKit; inference cannot substitute a missing molecular graph.")
    torch.set_num_threads(max(1, int(config.get("training", {}).get("cpu_threads", 4))))
    model = NaturalProductModel(vocab_size=len(tokenizer), max_seq_length=config["max_seq_length"],
                                use_graph=config["use_graph"], tasks=[t for t in states if t.get("trainable")],
                                **{**MODEL_DEFAULTS, **config["model"]})
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, tokenizer, config, states


def predict_file(checkpoint_dir, input_csv, output_csv, batch_size=16, requested_tasks=None):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    model, tokenizer, config, states = load_checkpoint(checkpoint_dir)
    available = {t["name"]: t for t in states if t.get("trained") and t.get("optimizer_updates_with_labels", 0) > 0}
    if isinstance(requested_tasks, str):
        requested_tasks = [s.strip() for s in requested_tasks.split(",") if s.strip()]
    selected = list(available) if requested_tasks is None else list(requested_tasks)
    if not selected:
        raise ValueError("No trained supervised endpoints are available; unsupervised representations cannot supply disease or ADMET probabilities.")
    if set(selected) - set(available):
        raise ValueError("Requested tasks are unknown or untrained: " + ", ".join(sorted(set(selected) - set(available))))
    tasks = [available[name] for name in selected]
    rows = read_rows(input_csv)
    output_rows = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start:start + int(batch_size)]
            ids, graph, _, _ = tensor_batch(batch, tokenizer, [], config, "cpu")
            results = model(ids, graph=graph, causal=config["training"]["objective"] == "causal")["tasks"]
            for i, row in enumerate(batch):
                result = {"compound_id": row["compound_id"], "smiles": row["smiles"]}
                for task in tasks:
                    name, kind = task["name"], task["kind"]
                    pred = results[name][i]
                    if kind == "regression":
                        result[name] = float(pred.item()) * task["std"] + task["mean"]
                    elif kind == "binary":
                        result[name + "_probability"] = float(pred.sigmoid().item())
                        result[name + "_label"] = int(pred.item() >= 0)
                    else:
                        probs = pred.softmax(-1)
                        result[name + "_label"] = task["classes"][int(probs.argmax())]
                        for j, label in enumerate(task["classes"]):
                            result[f"{name}_probability_{label}"] = float(probs[j])
                output_rows.append(result)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {"checkpoint": str(Path(checkpoint_dir).resolve()), "rows": len(rows),
                "output": str(destination), "trained_tasks": selected,
                "task_provenance": [{k: t.get(k) for k in ("name", "kind", "domain", "label_source", "label_evidence", "n_train_labels")} for t in tasks],
                "classification_calibration": "not_calibrated", "is_clinical_success_probability": False,
                "status": "research_predictions_not_clinical_recommendations",
                "limitations": "Applicability outside training chemistry is not validated. No unsupported indication probabilities are added."}
    _write_json(destination.with_suffix(destination.suffix + ".metadata.json"), metadata)
    return metadata


def sample_smiles(checkpoint_dir, n=10, max_tokens=None, temperature=1.0, seed=42):
    if int(n) < 1 or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("n and temperature must be positive and temperature finite.")
    model, tokenizer, config, _ = load_checkpoint(checkpoint_dir)
    if config["training"]["objective"] != "causal" or float(config["training"].get("lm_weight", 0)) <= 0:
        raise ValueError("Autoregressive generation requires a checkpoint trained with a nonzero causal-language objective.")
    limit = int(max_tokens) if max_tokens is not None else int(config["max_seq_length"]) - 1
    if not 1 <= limit < config["max_seq_length"]:
        raise ValueError("max_tokens must be positive and smaller than the model sequence limit.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    results = []
    with torch.no_grad():
        for _ in range(int(n)):
            ids, terminated = [tokenizer.bos_id], False
            for _ in range(limit):
                logits = model(torch.tensor([ids]), causal=True, lm_only=True)["lm_logits"][0, -1] / temperature
                logits[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id, tokenizer.mask_id]] = float("-inf")
                token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
                ids.append(token)
                if token == tokenizer.eos_id:
                    terminated = True
                    break
            smiles = tokenizer.decode(ids)
            valid, canonical = None, None
            if rdkit_available():
                try:
                    canonical = canonical_smiles(smiles)
                    valid = True
                except ValueError:
                    valid = False
            results.append({"smiles": smiles, "canonical_smiles": canonical, "valid_smiles": valid,
                            "terminated_with_eos": terminated,
                            "status": "generated_hypothesis_not_a_known_natural_product",
                            "druglikeness": None, "disease_indications": None})
    return results
