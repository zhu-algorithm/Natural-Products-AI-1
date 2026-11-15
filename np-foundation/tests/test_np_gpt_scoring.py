import csv
import numpy as np
import pytest
import torch

from npfm.np_gpt import NPTransformer, NPTransformerConfig
from npfm.np_gpt.runtime import save_model
from npfm.np_gpt.scoring import represent_records
from npfm.tokenizer import SmilesTokenizer


def test_representations_preserve_failures_and_padding_invariance(tmp_path):
    torch.manual_seed(7)
    tokenizer = SmilesTokenizer.fit(["C", "CCO"])
    model = NPTransformer(NPTransformerConfig(vocab_size=len(tokenizer), d_model=32,
                          n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=8)).eval()
    checkpoint = tmp_path / "checkpoint"
    save_model(model, tokenizer, checkpoint)
    rows = [{"compound_id": "a", "smiles": "C"},
            {"compound_id": "b", "smiles": "CCO"},
            {"compound_id": "oov", "smiles": "N"},
            {"compound_id": "long", "smiles": "C" * 7},
            {"compound_id": "bad", "smiles": ""}]
    metrics, manifest = represent_records(rows, checkpoint, tmp_path / "together", batch_size=5)
    assert [r["status"] for r in metrics] == ["scored", "scored", "out_of_vocabulary",
                                               "exceeds_context", "tokenization_failed"]
    assert [r["embedding_row"] for r in metrics] == [0, 1, None, None, None]
    assert manifest["scored_records"] == 2
    single, _ = represent_records(rows[:2], checkpoint, tmp_path / "single", batch_size=1)
    for i in range(2):
        assert metrics[i]["token_nll"] == pytest.approx(single[i]["token_nll"], abs=1e-6)
        encoded = torch.tensor([tokenizer.encode(rows[i]["smiles"])])
        expected = model(encoded[:, :-1], targets=encoded[:, 1:])["loss"].item()
        assert metrics[i]["token_nll"] == pytest.approx(expected, abs=1e-6)
    for filename in ["mean_embeddings.npy", "eos_embeddings.npy"]:
        np.testing.assert_allclose(np.load(tmp_path / "together" / filename),
                                   np.load(tmp_path / "single" / filename), rtol=1e-5, atol=1e-6)
    with (tmp_path / "together" / "sequence_metrics.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    with pytest.raises(ValueError, match="unique"):
        represent_records([rows[0], rows[0]], checkpoint, tmp_path / "duplicates")
