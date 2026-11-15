"""Trainable pilot workflow with group isolation, missing labels and honest task gates."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import warnings

import torch
from torch.nn import functional as F

from .features import batch_graphs, canonical_smiles, rdkit_available, rdkit_version, structure_group, molecule_graph
from .model import NaturalProductModel
from .tokenizer import SmilesTokenizer

MISSING = {"", "na", "n/a", "nan", "null", "none"}
MODEL_DEFAULTS = {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128,
                  "dropout": .1, "graph_layers": 2}
MAX_CHECKPOINT_GZIP_BYTES = 16 * 1024 * 1024


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_bytes(path):
    """Read original bytes, with a bounded lossless .gz distribution fallback.

    Hash manifests continue to describe the original uncompressed file. A present
    uncompressed file takes precedence; its failed hash never falls back to gzip.
    """
    path = Path(path)
    if path.suffix != ".gz" and path.is_file():
        return path.read_bytes()
    compressed = path if path.suffix == ".gz" else path.with_name(path.name + ".gz")
    with gzip.open(compressed, "rb") as stream:
        content = stream.read(MAX_CHECKPOINT_GZIP_BYTES + 1)
    if len(content) > MAX_CHECKPOINT_GZIP_BYTES:
        raise ValueError("Compressed checkpoint metadata exceeds the 16 MiB decompression limit.")
    return content


def verify_checkpoint(directory):
    directory = Path(directory)
    checks = json.loads((directory / "checkpoint_hashes.json").read_text())
    required = {"model.pt", "vocab.json", "resolved_config.json", "task_states.json", "splits.json", "data_manifest.json"}
    if not required.issubset(checks):
        raise ValueError("Checkpoint manifest does not cover the required model metadata.")
    for name, expected in checks.items():
        if Path(name).name != name:
            raise ValueError(f"Checkpoint integrity failure: {name}")
        path = directory / name
        actual = sha256_file(path) if path.is_file() and path.suffix != ".gz" else hashlib.sha256(checkpoint_bytes(path)).hexdigest()
        if actual != expected:
            raise ValueError(f"Checkpoint integrity failure: {name}")


def read_rows(path, max_records=None):
    if max_records is not None and int(max_records) < 1:
        raise ValueError("max_records must be positive.")
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"compound_id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires compound_id and smiles columns.")
        for row in reader:
            if not str(row.get("compound_id", "")).strip() or not str(row.get("smiles", "")).strip():
                raise ValueError(f"Missing compound_id or smiles at data row {len(rows) + 1}.")
            row["smiles"] = row["smiles"].strip()
            rows.append(row)
            if max_records is not None and len(rows) >= int(max_records):
                break
    if not rows:
        raise ValueError("No molecular records in input CSV.")
    return rows


def split_rows(rows, seed=42, val_fraction=.15, test_fraction=.15, split="scaffold", group_column=None):
    """Union duplicate IDs/structures and requested groups before assigning partitions."""
    if split not in {"scaffold", "structure", "group"}:
        raise ValueError("split must be scaffold, structure or group; row-random splits are not supported.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    if split == "group" and not group_column:
        raise ValueError("group_column is required for group splitting.")
    requested_split = split
    if split == "scaffold" and not rdkit_available():
        warnings.warn("RDKit unavailable: using exact-string structure grouping, not chemical scaffold grouping.")
        split = "structure"
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    seen = {}
    identities = []
    for i, row in enumerate(rows):
        identity = canonical_smiles(row["smiles"], stereochemistry=False)
        scaffold = structure_group(row["smiles"])
        identities.append({"index": i, "compound_id": row["compound_id"],
                           "smiles_sha256": _hash_text(row["smiles"]),
                           "structure_sha256": _hash_text(identity), "scaffold_sha256": _hash_text(scaffold)})
        keys = [("id", row["compound_id"]), ("structure", identity)]
        if split == "scaffold":
            keys.append(("scaffold", scaffold))
        if split == "group":
            group = str(row.get(group_column, "")).strip()
            if not group:
                raise ValueError(f"Empty group value in {group_column} at row {i}.")
            keys.append(("group", group))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    grouped = list(groups.values())
    if len(grouped) < 3:
        raise ValueError("At least three independent structure/groups are required for train/validation/test.")
    random.Random(seed).shuffle(grouped)
    parts = {"train": [], "validation": [], "test": []}
    for part, fraction, reserve in [("test", test_fraction, 2), ("validation", val_fraction, 1)]:
        target = max(1, round(len(rows) * fraction))
        while len(grouped) > reserve and (not parts[part] or len(parts[part]) < target):
            parts[part].extend(grouped.pop())
        if not parts[part]:
            parts[part].extend(grouped.pop())
    parts["train"] = [i for group in grouped for i in group]
    for part, indices in parts.items():
        indices.sort()
        for i in indices:
            identities[i]["partition"] = part
    metadata = {"requested_strategy": requested_split, "strategy": split,
                "chemical_identity_normalized": rdkit_available(), "group_column": group_column,
                "seed": seed, "independent_groups": len(groups), "rows": identities,
                "note": "IDs, identical connectivity and requested groups stay together; grouped row fractions may differ."}
    return parts, metadata


def numeric_label(value, kind="regression"):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite label: {value}")
    if kind == "binary" and number not in {0., 1.}:
        raise ValueError(f"Binary label must be 0 or 1, got {value}.")
    return number


def build_task_states(tasks, train_rows):
    states, seen = [], set()
    for original in tasks:
        state = copy.deepcopy(original)
        name, kind = state.get("name", ""), state.get("kind", "regression")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in seen:
            raise ValueError("Task names must be unique ASCII identifiers starting with a letter.")
        seen.add(name)
        if kind not in {"regression", "binary", "multiclass"}:
            raise ValueError(f"Unsupported task kind: {kind}")
        weight = float(state.get("weight", 1.))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Task {name} weight must be finite and non-negative.")
        state.update(kind=kind, column=state.get("column", name), weight=weight, output_dim=1, trained=False)
        if state.get("domain") in {"disease", "clinical", "indication"}:
            if not state.get("label_source") or state.get("label_evidence") not in {"curated_association", "clinical_observation", "approved_indication", "experimental"}:
                raise ValueError(f"Task {name} requires a real label_source and explicit label_evidence; no unsupervised disease labels.")
        raw = [r.get(state["column"]) for r in train_rows]
        if kind == "multiclass":
            values = [str(v).strip() for v in raw if v is not None and str(v).strip().lower() not in MISSING]
            state["classes"] = sorted(set(values))
            state["output_dim"] = max(1, len(state["classes"]))
        else:
            values = [v for v in (numeric_label(x, kind) for x in raw) if v is not None]
        state["n_train_labels"] = len(values)
        state["trainable"] = weight > 0 and len(values) >= int(state.get("min_labels", 2)) and len(set(values)) >= 2
        state["untrained_reason"] = None if state["trainable"] else "Insufficient labels or fewer than two distinct training outcomes."
        if weight == 0:
            state["untrained_reason"] = "Supervised loss disabled by this task's weight=0."
        if kind == "regression" and values:
            state["mean"] = sum(values) / len(values)
            state["std"] = max(1e-8, math.sqrt(sum((v - state["mean"]) ** 2 for v in values) / len(values)))
        else:
            state["mean"], state["std"] = 0., 1.
        states.append(state)
    return states


def tensor_batch(rows, tokenizer, tasks, config, device):
    sequences = [tokenizer.encode(row["smiles"], max_length=config["max_seq_length"]) for row in rows]
    ids = torch.full((len(rows), max(map(len, sequences))), tokenizer.pad_id, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    graph = batch_graphs([r["smiles"] for r in rows], max_atoms=config.get("max_atoms"), device=device) if config["use_graph"] else None
    targets, masks = {}, {}
    for task in tasks:
        values, valid = [], []
        for row in rows:
            raw = row.get(task["column"])
            if task["kind"] == "multiclass":
                text = str(raw).strip() if raw is not None else ""
                good = text in task["classes"] and text.lower() not in MISSING
                value = task["classes"].index(text) if good else 0
            else:
                value = numeric_label(raw, task["kind"])
                good = value is not None
                value = (value - task["mean"]) / task["std"] if good and task["kind"] == "regression" else (value or 0.)
            values.append(value)
            valid.append(good)
        dtype = torch.long if task["kind"] == "multiclass" else torch.float32
        targets[task["name"]] = torch.tensor(values, dtype=dtype, device=device)
        masks[task["name"]] = torch.tensor(valid, dtype=torch.bool, device=device)
    return ids, graph, targets, masks


def supervised_loss(outputs, targets, masks, tasks):
    losses = []
    for task in tasks:
        name, kind = task["name"], task["kind"]
        valid = masks[name]
        weight = float(task.get("weight", 1.))
        if not valid.any() or weight <= 0:
            continue
        pred, truth = outputs[name][valid], targets[name][valid]
        if kind == "multiclass":
            loss = F.cross_entropy(pred, truth)
        elif kind == "binary":
            loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), truth)
        else:
            loss = F.mse_loss(pred.squeeze(-1), truth)
        losses.append(weight * loss)
    return torch.stack(losses).mean() if losses else None


def multimodal_contrastive_loss(sequence, graph, fused, identities, temperature=.1):
    if graph is None or len(set(identities)) < 2:
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("contrastive_temperature must be positive and finite.")
    positives = torch.tensor([[a == b for b in identities] for a in identities], dtype=torch.bool, device=sequence.device)

    def pair(a, b):
        scores = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
        forward = torch.logsumexp(scores, dim=1) - torch.logsumexp(scores.masked_fill(~positives, float("-inf")), dim=1)
        reverse = torch.logsumexp(scores.T, dim=1) - torch.logsumexp(scores.T.masked_fill(~positives.T, float("-inf")), dim=1)
        return (forward.mean() + reverse.mean()) / 2

    return (pair(sequence, graph) + pair(fused, sequence) + pair(fused, graph)) / 3


def _language_loss(model, ids, tokenizer, settings, generator=None):
    objective = settings.get("objective", "causal")
    if objective == "causal":
        logits = model(ids[:, :-1], causal=True, lm_only=True)["lm_logits"]
        labels = ids[:, 1:]
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=tokenizer.pad_id, reduction="sum")
        count = int(labels.ne(tokenizer.pad_id).sum())
    elif objective == "mlm":
        probability = float(settings.get("mask_probability", .15))
        if not 0 < probability <= 1:
            raise ValueError("mask_probability must be in (0,1].")
        eligible = ids.ge(len([tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id, tokenizer.mask_id]))
        selected = (torch.rand(ids.shape, device=ids.device, generator=generator) < probability) & eligible
        if not selected.any():
            indices = eligible.nonzero()
            if not len(indices):
                raise ValueError("No known chemical tokens available for masked-language training.")
            selected[tuple(indices[0])] = True
        masked = ids.clone()
        masked[selected] = tokenizer.mask_id
        logits = model(masked, causal=False, lm_only=True)["lm_logits"]
        total = F.cross_entropy(logits[selected], ids[selected], reduction="sum")
        count = int(selected.sum())
    else:
        raise ValueError("Language objective must be causal or mlm.")
    return total / max(count, 1), count


def evaluate(model, rows, tokenizer, tasks, config, device):
    model.eval()
    settings = config["training"]
    collected = {t["name"]: {"pred": [], "truth": []} for t in tasks}
    nll_total, n_tokens, contrastives = 0., 0, []
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 991)
    with torch.no_grad():
        for start in range(0, len(rows), int(settings["batch_size"])):
            batch = rows[start:start + int(settings["batch_size"])]
            ids, graph, targets, masks = tensor_batch(batch, tokenizer, tasks, config, device)
            lm, count = _language_loss(model, ids, tokenizer, settings, generator=generator)
            nll_total += float(lm) * count
            n_tokens += count
            contrastive_weight = float(settings.get("contrastive_weight", 0))
            full = model(ids, graph=graph, causal=settings["objective"] == "causal") if tasks or contrastive_weight > 0 else None
            if contrastive_weight > 0:
                identities = [canonical_smiles(r["smiles"]) for r in batch]
                contrastive = multimodal_contrastive_loss(full["sequence_embedding"], full["graph_embedding"], full["embedding"], identities,
                                                          float(settings.get("contrastive_temperature", .1)))
                if contrastive is not None:
                    contrastives.append(float(contrastive))
            for task in tasks:
                name = task["name"]
                valid = masks[name]
                collected[name]["pred"].extend(full["tasks"][name][valid].cpu().tolist())
                collected[name]["truth"].extend(targets[name][valid].cpu().tolist())
    task_metrics, objectives = {}, []
    for task in tasks:
        name, kind = task["name"], task["kind"]
        pred, truth = collected[name]["pred"], collected[name]["truth"]
        result = {"n_labels": len(truth)}
        if truth:
            p, y = torch.tensor(pred, dtype=torch.float64), torch.tensor(truth, dtype=torch.float64)
            if kind == "regression":
                delta = p.squeeze(-1) - y
                result.update(mae=float(delta.abs().mean()) * task["std"],
                              rmse=float(delta.square().mean().sqrt()) * task["std"], normalized_mse=float(delta.square().mean()))
                objective = result["normalized_mse"]
            elif kind == "binary":
                p = p.squeeze(-1)
                result.update(binary_cross_entropy=float(F.binary_cross_entropy_with_logits(p, y)),
                              accuracy=float(((p >= 0) == y.bool()).double().mean()), calibration="not_calibrated")
                objective = result["binary_cross_entropy"]
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    if len(set(truth)) >= 2:
                        result["auroc"] = float(roc_auc_score(truth, p.sigmoid().tolist()))
                        result["average_precision"] = float(average_precision_score(truth, p.sigmoid().tolist()))
                except ImportError:
                    pass
            else:
                result.update(cross_entropy=float(F.cross_entropy(p, y.long())),
                              accuracy=float((p.argmax(-1) == y.long()).double().mean()), calibration="not_calibrated")
                objective = result["cross_entropy"]
                result["n_unknown_class_labels_excluded"] = sum(str(r.get(task["column"], "")).strip().lower() not in MISSING and
                    str(r.get(task["column"], "")).strip() not in task["classes"] for r in rows)
            objectives.append(float(task.get("weight", 1)) * objective)
        task_metrics[name] = result
    return {"n_rows": len(rows), "language_token_nll": nll_total / n_tokens if n_tokens else None,
            "contrastive_objective": sum(contrastives) / len(contrastives) if contrastives else None,
            "contrastive_evaluable_batches": len(contrastives), "tasks": task_metrics,
            "supervised_objective": sum(objectives) / len(objectives) if objectives else None}


def _initialization_checks(directory, config, split_record):
    verify_checkpoint(directory)
    prior_config = json.loads((directory / "resolved_config.json").read_text())
    if prior_config.get("model_format_version") != config.get("model_format_version"):
        raise ValueError("Initial checkpoint architecture version differs; do not mix old weights with recovered metadata.")
    if prior_config["model"] != config["model"] or prior_config["max_seq_length"] != config["max_seq_length"] or prior_config["use_graph"] != config["use_graph"]:
        raise ValueError("initial_checkpoint must have a compatible model configuration, max_seq_length and graph mode.")
    prior = json.loads(checkpoint_bytes(directory / "splits.json"))
    prior_rows = prior.get("rows", prior.get("metadata", {}).get("rows", []))
    if not prior_rows:
        raise ValueError("Initial checkpoint lacks auditable structure identities for leakage checking.")
    current_rows = split_record["rows"]
    for key in ["structure_sha256", "scaffold_sha256"]:
        trained = {r[key] for r in prior_rows if r["partition"] == "train"}
        heldout = {r[key] for r in current_rows if r["partition"] in {"validation", "test"}}
        if trained & heldout:
            raise ValueError("Historical training structure/scaffold overlaps new validation/test; initialization would leak holdout data.")
    validated = {r["structure_sha256"] for r in prior_rows if r["partition"] == "validation"}
    test = {r["structure_sha256"] for r in current_rows if r["partition"] == "test"}
    if validated & test:
        raise ValueError("Historical validation structures overlap the new test set.")


def train_from_config(config_path, overrides=None):
    path = Path(config_path).resolve()
    config = copy.deepcopy(json.loads(path.read_text()))
    config.setdefault("training", {})
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "epochs":
            config["training"][key] = value
        elif key == "training":
            config["training"].update(value)
        else:
            config[key] = value
    for name in ("data_path", "output_dir", "initial_checkpoint"):
        if config.get(name):
            value = Path(config[name]).expanduser()
            config[name] = str((value if value.is_absolute() else path.parent / value).resolve())
    if not config.get("data_path") or not config.get("output_dir"):
        raise ValueError("Configuration requires data_path and output_dir.")
    settings = {"epochs": 2, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": .01,
                "objective": "causal", "lm_weight": 1., "task_weight": 1., "contrastive_weight": 0.,
                "val_fraction": .15, "test_fraction": .15, "split": "scaffold", "selection_metric": "auto",
                "device": "cpu", "cpu_threads": 4, **config["training"]}
    config["training"] = settings
    config.setdefault("seed", 42)
    config.setdefault("max_seq_length", 512)
    config.setdefault("max_atoms", 256)
    config["model"] = {**MODEL_DEFAULTS, **config.get("model", {})}
    config["model_format_version"] = 2
    if int(settings["epochs"]) < 1 or int(settings["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive.")
    mode = config.get("graph_mode", "auto")
    if mode not in {"auto", "required", "disabled"}:
        raise ValueError("graph_mode must be auto, required or disabled.")
    if mode == "required" and not rdkit_available():
        raise RuntimeError("graph_mode=required needs RDKit.")
    config["use_graph"] = mode != "disabled" and rdkit_available()
    if mode == "auto" and not config["use_graph"]:
        warnings.warn("RDKit unavailable: this run is explicitly sequence-only.")
    if float(settings["contrastive_weight"]) > 0 and not config["use_graph"]:
        warnings.warn("Graph unavailable/disabled: contrastive_weight is explicitly set to zero.")
        settings["contrastive_weight"] = 0.
    weights = [float(settings[k]) for k in ("lm_weight", "task_weight", "contrastive_weight")]
    if any(not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("Loss weights must be finite and non-negative.")
    lm_weight, task_weight, contrastive_weight = weights
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    torch.set_num_threads(max(1, int(settings["cpu_threads"])))
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose device=cpu.")
    rows = read_rows(config["data_path"], config.get("max_records"))
    for row in rows:
        n = len(SmilesTokenizer.tokenize(row["smiles"])) + 2
        if n > config["max_seq_length"]:
            raise ValueError(f"{row['compound_id']} exceeds max_seq_length; no silent truncation.")
        if config["use_graph"]:
            molecule_graph(row["smiles"], max_atoms=config["max_atoms"])
    parts, metadata = split_rows(rows, seed=int(config["seed"]), val_fraction=float(settings["val_fraction"]),
                                  test_fraction=float(settings["test_fraction"]), split=settings["split"],
                                  group_column=settings.get("group_column"))
    split_record = {**parts, "rows": metadata.pop("rows"), "metadata": metadata}
    train_rows = [rows[i] for i in parts["train"]]
    initial_dir = Path(config["initial_checkpoint"]) if config.get("initial_checkpoint") else None
    if initial_dir:
        _initialization_checks(initial_dir, config, split_record)
        tokenizer = SmilesTokenizer.load(initial_dir / "vocab.json")
    else:
        tokenizer = SmilesTokenizer.fit(r["smiles"] for r in train_rows)
    states = build_task_states(config.get("tasks", []), train_rows)
    if task_weight == 0:
        for task in states:
            task["trainable"] = False
            task["untrained_reason"] = "Supervised loss disabled by task_weight=0."
    tasks = [t for t in states if t["trainable"]]
    if not lm_weight and not contrastive_weight and not tasks:
        raise ValueError("No nonzero loss objective with usable labels is configured.")
    model = NaturalProductModel(vocab_size=len(tokenizer), max_seq_length=config["max_seq_length"],
                                use_graph=config["use_graph"], tasks=tasks, **config["model"]).to(device)
    initialization = {"mode": "random_initialization"}
    if initial_dir:
        prior = torch.load(initial_dir / "model.pt", map_location=device, weights_only=True)
        compatible = {k: v for k, v in prior.items() if not k.startswith("heads.") and k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(compatible, strict=False)
        initialization = {"mode": "encoder_initialization_not_optimizer_resume", "checkpoint": str(initial_dir),
                          "model_sha256": sha256_file(initial_dir / "model.pt"), "tensors_copied": len(compatible), "heads_reused": False}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    updated = {t["name"]: 0 for t in tasks}
    contrastive_updates, graph_supervised_updates = 0, 0
    best, history, selection_record = float("inf"), [], None
    best_updated, best_contrastive, best_graph_supervised = {}, 0, 0
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        shuffled = list(train_rows)
        random.Random(int(config["seed"]) + epoch).shuffle(shuffled)
        losses = []
        for start in range(0, len(shuffled), int(settings["batch_size"])):
            batch = shuffled[start:start + int(settings["batch_size"])]
            ids, graph, targets, masks = tensor_batch(batch, tokenizer, tasks, config, device)
            optimizer.zero_grad(set_to_none=True)
            terms = []
            if lm_weight > 0:
                language_loss, _ = _language_loss(model, ids, tokenizer, settings)
                terms.append(lm_weight * language_loss)
            complete = model(ids, graph=graph, causal=settings["objective"] == "causal") if tasks or contrastive_weight else None
            task_loss = supervised_loss(complete["tasks"], targets, masks, tasks) if tasks else None
            if task_loss is not None:
                terms.append(task_weight * task_loss)
            contrastive = None
            if contrastive_weight:
                contrastive = multimodal_contrastive_loss(complete["sequence_embedding"], complete["graph_embedding"], complete["embedding"],
                    [canonical_smiles(r["smiles"]) for r in batch], float(settings.get("contrastive_temperature", .1)))
                if contrastive is not None:
                    terms.append(contrastive_weight * contrastive)
            if not terms:
                continue
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training objective became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("max_grad_norm", 1.)))
            optimizer.step()
            losses.append(float(loss.detach()))
            if task_loss is not None:
                for task in tasks:
                    if masks[task["name"]].any():
                        updated[task["name"]] += 1
                graph_supervised_updates += int(config["use_graph"])
            contrastive_updates += int(contrastive is not None)
        metrics = evaluate(model, [rows[i] for i in parts["validation"]], tokenizer, tasks, config, device)
        selection = settings["selection_metric"]
        if selection == "auto":
            selection = "supervised" if tasks else ("joint" if contrastive_weight else "language")
        if selection == "supervised":
            criterion = metrics["supervised_objective"]
            if criterion is None:
                raise ValueError("Validation has no usable supervised labels; cannot choose a supervised checkpoint.")
            criterion_name = "validation_mean_normalized_MSE_or_cross_entropy"
        elif selection == "language":
            if lm_weight <= 0:
                raise ValueError("Cannot select on an untrained language objective when lm_weight=0.")
            criterion, criterion_name = metrics["language_token_nll"], "validation_language_token_nll"
        elif selection == "joint":
            needed = [(lm_weight, "language_token_nll"), (task_weight if tasks else 0, "supervised_objective"), (contrastive_weight, "contrastive_objective")]
            if any(weight > 0 and metrics[key] is None for weight, key in needed):
                raise ValueError("Validation cannot evaluate every enabled objective; revise split or batch size.")
            criterion = sum(weight * (metrics[key] or 0.) for weight, key in needed)
            criterion_name = "validation_weighted_joint_objective"
        else:
            raise ValueError("selection_metric must be auto, supervised, language or joint.")
        if criterion is None or not math.isfinite(criterion):
            raise FloatingPointError("Validation selection objective is not finite.")
        epoch_record = {"epoch": epoch, "training_objective": sum(losses) / len(losses) if losses else None,
                        "selection_criterion": criterion_name, "selection_value": criterion, "validation": metrics}
        history.append(epoch_record)
        print(json.dumps({k: v for k, v in epoch_record.items() if k != "validation"}), flush=True)
        if criterion < best:
            best = criterion
            torch.save(model.state_dict(), output / "model.pt.tmp")
            (output / "model.pt.tmp").replace(output / "model.pt")
            selection_record = {"epoch": epoch, "criterion": criterion_name, "value": criterion}
            best_updated, best_contrastive, best_graph_supervised = dict(updated), contrastive_updates, graph_supervised_updates
    for task in states:
        task["trained"] = bool(task["trainable"] and best_updated.get(task["name"], 0))
        task["optimizer_updates_with_labels"] = best_updated.get(task["name"], 0)
        task["total_run_updates_with_labels"] = updated.get(task["name"], 0)
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    final_metrics = {"history": history,
                     "validation": evaluate(model, [rows[i] for i in parts["validation"]], tokenizer, tasks, config, device),
                     "test": evaluate(model, [rows[i] for i in parts["test"]], tokenizer, tasks, config, device),
                     "note": "Test set was evaluated only after checkpoint selection. Pilot metrics do not establish generalization or clinical validity."}
    unknown = {}
    for part, indices in parts.items():
        tokens = [token for i in indices for token in tokenizer.encode(rows[i]["smiles"], add_special_tokens=False)]
        unknown[part] = sum(t == tokenizer.unk_id for t in tokens) / max(1, len(tokens))
    manifest = {"source_file": config["data_path"], "source_file_sha256": sha256_file(config["data_path"]),
                "used_records_sha256": _hash_text(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
                "max_records": config.get("max_records"), "used_records": len(rows),
                "unique_smiles": len({r["smiles"] for r in rows}), "python": platform.python_version(),
                "torch": torch.__version__, "rdkit": rdkit_version(), "use_graph": config["use_graph"],
                "graph_encoder_supervised": bool(best_graph_supervised), "contrastive_weight": contrastive_weight,
                "graph_and_fusion_contrastive_objective_enabled": bool(contrastive_weight),
                "contrastive_updates_in_selected_checkpoint": best_contrastive,
                "graph_encoder_trained": bool(best_graph_supervised or best_contrastive),
                "fusion_trained": bool(best_graph_supervised or best_contrastive),
                "tokenizer_fitted_on": "prior_training_partition_only" if initial_dir else "training_partition_only",
                "initialization": initialization, "unseen_token_fraction": unknown,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "limitations": ["Small research baseline, not a clinically validated model.", "2D graph features, not 3D binding.",
                                "Disease tasks require separately supplied evidence and labels.", "Selected CSV records are loaded into memory; this pilot is not a distributed full-corpus trainer."]}
    tokenizer.save(output / "vocab.json")
    for name, value in [("resolved_config.json", config), ("task_states.json", states), ("splits.json", split_record),
                        ("data_manifest.json", manifest), ("metrics.json", final_metrics), ("checkpoint_selection.json", selection_record)]:
        _write_json(output / name, value)
    checked = ["model.pt", "vocab.json", "resolved_config.json", "task_states.json", "splits.json", "data_manifest.json"]
    _write_json(output / "checkpoint_hashes.json", {name: sha256_file(output / name) for name in checked})
    return str(output)


train = train_from_config
