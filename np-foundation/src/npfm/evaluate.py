"""Missing-label-aware endpoint metrics and leakage-conscious data splits."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Sequence
import math
import numpy as np

SPLITS = ('train', 'validation', 'test')


def _float_array(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    result = np.empty(raw.shape, dtype=float)
    for index in np.ndindex(raw.shape):
        value = raw[index]
        if value is None or (isinstance(value, str) and value.strip().casefold() in {'', 'nan', 'na', 'null', 'none', 'unknown'}):
            result[index] = np.nan
        else:
            result[index] = float(value)
    return result


def evaluate_predictions(y_true: Any, y_pred: Any, task: str = 'regression') -> dict[str, Any]:
    """Evaluate one endpoint, excluding missing labels and missing predictions.

    Classification predictions must be probabilities. Ranking metrics for a
    single observed class are undefined and returned as ``None``. PR-AUC is the
    trapezoidal area, reported separately from average precision.
    """
    if task not in {'regression', 'classification'}:
        raise ValueError('task must be regression or classification')
    truth, pred = _float_array(y_true), _float_array(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape:
        raise ValueError('y_true and y_pred must be equally sized one-dimensional arrays')
    observed = np.isfinite(truth)
    valid = observed & np.isfinite(pred)
    n, n_labeled, n_eval = len(truth), int(observed.sum()), int(valid.sum())
    report = {'task': task, 'n_total': n, 'n_labeled': n_labeled, 'n_evaluated': n_eval,
              'label_coverage': n_labeled / n if n else 0.0,
              'prediction_coverage': n_eval / n_labeled if n_labeled else 0.0,
              'coverage': n_eval / n if n else 0.0}
    truth, pred = truth[valid], pred[valid]
    if task == 'regression':
        report.update({'mae': None, 'mse': None, 'rmse': None})
        if n_eval:
            error = pred - truth
            report.update(mae=float(np.mean(np.abs(error))), mse=float(np.mean(error**2)),
                          rmse=float(np.sqrt(np.mean(error**2))))
    else:
        report.update({'roc_auc': None, 'pr_auc': None, 'average_precision': None,
                       'brier': None, 'accuracy': None})
        if n_eval:
            if not np.isin(truth, [0, 1]).all():
                raise ValueError('Binary classification labels must be 0 or 1')
            if (pred < 0).any() or (pred > 1).any():
                raise ValueError('Classification predictions must lie between 0 and 1')
            report['brier'] = float(np.mean((pred - truth)**2))
            report['accuracy'] = float(np.mean((pred >= 0.5) == truth))
            if len(np.unique(truth)) == 2:
                from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
                precision, recall, _ = precision_recall_curve(truth, pred)
                report['roc_auc'] = float(roc_auc_score(truth, pred))
                report['pr_auc'] = float(auc(recall, precision))
                report['average_precision'] = float(average_precision_score(truth, pred))
    return report


def evaluate_multitask(y_true: Any, y_pred: Any, tasks: Sequence[str],
                       names: Sequence[str] | None = None) -> dict[str, Any]:
    """Return per-head metrics; endpoints with different units are not averaged."""
    truth, pred = _float_array(y_true), _float_array(y_pred)
    if truth.ndim != 2 or truth.shape != pred.shape:
        raise ValueError('Multitask inputs must have equal (samples, tasks) shapes')
    if len(tasks) != truth.shape[1]:
        raise ValueError('One task type is needed for each endpoint')
    names = list(names) if names is not None else [f'task_{i}' for i in range(len(tasks))]
    if len(names) != len(tasks) or len(set(names)) != len(names):
        raise ValueError('Endpoint names must be unique and match task count')
    return {'n_samples': int(truth.shape[0]), 'n_tasks': len(tasks),
            'tasks': {name: evaluate_predictions(truth[:, i], pred[:, i], tasks[i])
                      for i, name in enumerate(names)}}


def _fractions(fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError('fractions must be three nonnegative finite values summing to 1')
    return values


def _groups(groups: Sequence[Any]) -> dict[Any, list[int]]:
    grouped: dict[Any, list[int]] = {}
    for i, group in enumerate(groups):
        if group is None or (isinstance(group, (float, np.floating)) and not np.isfinite(group)):
            raise ValueError('Missing group identities cannot be split safely')
        try:
            grouped.setdefault(group, []).append(i)
        except TypeError as exc:
            raise ValueError('Group identities must be hashable') from exc
    return grouped


def group_split(groups: Sequence[Any], fractions: Sequence[float] = (0.8, 0.1, 0.1),
                seed: int = 42) -> dict[str, list[int]]:
    """Assign entire groups to partitions; proportions are approximate.

    Largest groups are allocated first with randomized tie order. When enough
    groups exist, one group is reserved for every requested nonempty partition.
    """
    ratios = _fractions(fractions)
    grouped = _groups(groups)
    chunks = list(grouped.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)
    chunks.sort(key=len, reverse=True)
    result: list[list[int]] = [[], [], []]
    target = ratios * len(groups)
    counts = np.zeros(3, dtype=int)
    active = [i for i, value in enumerate(ratios) if value > 0]
    for i, chunk in enumerate(chunks):
        empty = [j for j in active if not result[j]]
        choices = empty if len(chunks) - i == len(empty) else active
        chosen = max(choices, key=lambda j: target[j] - counts[j])
        result[chosen].extend(chunk)
        counts[chosen] += len(chunk)
    return {name: sorted(rows) for name, rows in zip(SPLITS, result)}


def scaffold_groups(smiles: Sequence[str], include_chirality: bool = False) -> list[str]:
    """Compute true RDKit Bemis-Murcko scaffold identities; no random fallback."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise ImportError('RDKit is required for scaffold splitting') from exc
    result = []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if mol is None:
            raise ValueError(f'Invalid SMILES at row {i}')
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        result.append(scaffold or '__ACYCLIC__')
    return result


def scaffold_split(smiles: Sequence[str], fractions: Sequence[float] = (0.8, 0.1, 0.1),
                   seed: int = 42, include_chirality: bool = False) -> dict[str, list[int]]:
    return group_split(scaffold_groups(smiles, include_chirality), fractions, seed)


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, date):
        stamp = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        try:
            stamp = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'Invalid ISO timestamp: {value}') from exc
    else:
        raise ValueError('Time splits require complete ISO dates or datetime values')
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def time_split(timestamps: Sequence[Any], train_before: Any, validation_before: Any,
               groups: Sequence[Any] | None = None) -> dict[str, list[int]]:
    """Use [past, train_before), [train_before, validation_before), [future).

    If groups are supplied, every group is placed in its latest observation's
    partition. Thus an identity spanning the cutoff is excluded from training.
    """
    train_cut, validation_cut = _time(train_before), _time(validation_before)
    if train_cut >= validation_cut:
        raise ValueError('train_before must precede validation_before')
    stamps = [_time(t) for t in timestamps]
    if groups is None:
        groups = list(range(len(stamps)))
    if len(groups) != len(stamps):
        raise ValueError('groups must match timestamp count')
    result = {name: [] for name in SPLITS}
    for rows in _groups(groups).values():
        latest = max(stamps[i] for i in rows)
        part = 'train' if latest < train_cut else 'validation' if latest < validation_cut else 'test'
        result[part].extend(rows)
    return {name: sorted(rows) for name, rows in result.items()}
