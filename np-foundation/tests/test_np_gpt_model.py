"""Behavioral invariants for the new causal Transformer, independent of the pilot."""
import json

import pytest
import torch
from torch.nn import functional as F

from npfm.np_gpt import NPTransformer, NPTransformerConfig


def tiny_model(**kwargs):
    torch.manual_seed(123)
    return NPTransformer(NPTransformerConfig(vocab_size=24, d_model=32, n_layers=2,
                         n_heads=4, n_kv_heads=kwargs.pop("n_kv_heads", 2),
                         ffn_hidden_dim=64, max_seq_len=24, dropout=0., **kwargs))


@pytest.mark.parametrize("change", [
    {"vocab_size": 1}, {"d_model": 31}, {"d_model": 12}, {"n_layers": 0},
    {"n_kv_heads": 3}, {"n_kv_heads": 0}, {"max_seq_len": 0}, {"ffn_hidden_dim": 0},
    {"dropout": 1.}, {"rope_theta": 0.}, {"norm_eps": float("nan")},
    {"pad_token_id": 30}, {"tie_embeddings": "yes"},
])
def test_config_rejects_invalid_geometry_and_values(change):
    with pytest.raises(ValueError):
        NPTransformerConfig(**{"vocab_size": 24, **change})


def test_gradients_reach_attention_rope_inputs_and_swiglu_with_tied_weights():
    model = tiny_model()
    ids = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 10]])
    targets = torch.tensor([[5, 6, 7, 2], [8, 9, 10, 2]])
    result = model(ids, targets=targets)
    assert torch.isfinite(result["loss"]) and result["loss"].item() > 0
    result["loss"].backward()
    for param in [model.token_embedding.weight, model.layers[0].attention.q_proj.weight,
                  model.layers[0].attention.k_proj.weight, model.layers[0].attention.v_proj.weight,
                  model.layers[0].ffn.gate_proj.weight, model.layers[0].ffn.up_proj.weight,
                  model.layers[0].ffn.down_proj.weight]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert model.lm_head.weight is model.token_embedding.weight
    assert result["past_key_values"] is None
    assert "hidden_states" not in result


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_future_tokens_do_not_change_prefix_logits(explicit_mask):
    model = tiny_model().eval()
    first = torch.tensor([[1, 5, 6, 7, 8, 2]])
    second = torch.tensor([[1, 5, 6, 12, 13, 14]])
    mask = torch.ones_like(first, dtype=torch.bool) if explicit_mask else None
    with torch.no_grad():
        a = model(first, attention_mask=mask)["logits"]
        b = model(second, attention_mask=mask)["logits"]
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-7)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_cache_single_token_and_offset_chunk_match_full_attention(n_kv_heads):
    model = tiny_model(n_kv_heads=n_kv_heads).eval()
    ids = torch.tensor([[1, 5, 6, 7, 8, 2], [1, 9, 10, 11, 12, 2]])
    with torch.no_grad():
        full = model(ids)["logits"]
        prefill = model(ids[:, :2], use_cache=True)
        old_cache = prefill["past_key_values"]
        single = model(ids[:, 2:3], past_key_values=old_cache, use_cache=True)
        chunk = model(ids[:, 3:], past_key_values=single["past_key_values"], use_cache=True)
    combined = torch.cat([prefill["logits"], single["logits"], chunk["logits"]], dim=1)
    torch.testing.assert_close(full, combined, rtol=1e-5, atol=2e-6)
    assert len(chunk["past_key_values"]) == model.config.n_layers
    assert chunk["past_key_values"][0][0].shape == (2, n_kv_heads, 6, 8)
    assert old_cache[0][0].shape[2] == 2  # Appending must not mutate the caller's cache.


def test_padded_prefix_cache_positions_match_full_and_unpadded_runs():
    model = tiny_model().eval()
    # First row is left-padded; second contains a right-padded prefix, then new tokens.
    ids = torch.tensor([[0, 0, 1, 5, 6, 2], [1, 5, 0, 0, 6, 2]])
    mask = ids.ne(0)
    with torch.no_grad():
        full = model(ids, attention_mask=mask)["logits"]
        prefix = model(ids[:, :4], attention_mask=mask[:, :4], use_cache=True)
        suffix = model(ids[:, 4:], attention_mask=mask, past_key_values=prefix["past_key_values"], use_cache=True)
        unpadded = model(torch.tensor([[1, 5, 6, 2]]))["logits"]
    torch.testing.assert_close(full[:, 4:], suffix["logits"], rtol=1e-5, atol=2e-6)
    for i in range(2):
        torch.testing.assert_close(full[i, mask[i]], unpadded[0], rtol=1e-5, atol=2e-6)
    with pytest.raises(ValueError, match="cached_prefix_length"):
        model(ids[:, 4:], attention_mask=mask[:, 4:], past_key_values=prefix["past_key_values"])


def test_padding_and_ignore_labels_are_excluded_without_nan_or_fake_targets():
    model = tiny_model()
    ids = torch.tensor([[1, 5, 6, 0], [0, 0, 1, 8]])
    targets = torch.tensor([[5, -100, 2, 0], [7, 9, 8, 2]])
    mask = ids.ne(0)
    result = model(ids, targets=targets, attention_mask=mask)
    valid = mask & targets.ne(0) & targets.ne(-100)
    expected = F.cross_entropy(result["logits"][valid].float(), targets[valid])
    torch.testing.assert_close(result["loss"], expected)
    empty = model(torch.zeros_like(ids), targets=torch.zeros_like(ids), attention_mask=torch.zeros_like(mask))
    assert torch.isfinite(empty["logits"]).all()
    assert empty["loss"].requires_grad and empty["loss"].item() == 0
    empty["loss"].backward()
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_checkpoint_roundtrip_and_optional_hidden_states(tmp_path):
    model = tiny_model().eval()
    ids = torch.tensor([[1, 5, 6, 2]])
    with torch.no_grad():
        expected = model(ids, output_hidden_states=True)
        encoded = model.encode(ids)
    assert expected["hidden_states"].shape == (1, 4, 32)
    torch.testing.assert_close(encoded, expected["hidden_states"], rtol=0, atol=0)
    config_path, weights_path = tmp_path / "config.json", tmp_path / "model.pt"
    config_path.write_text(json.dumps(model.config.to_dict()))
    torch.save(model.state_dict(), weights_path)
    config = NPTransformerConfig.from_dict(json.loads(config_path.read_text()))
    restored = NPTransformer(config).eval()
    restored.load_state_dict(torch.load(weights_path, weights_only=True))
    with torch.no_grad():
        actual = restored(ids)["logits"]
    torch.testing.assert_close(actual, expected["logits"], rtol=0, atol=0)
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert restored.num_parameters() == model.num_parameters()
    with pytest.raises(ValueError, match="Unknown"):
        NPTransformerConfig.from_dict({**config.to_dict(), "parameter_count": 1})


def test_context_and_cache_validation_are_explicit():
    model = tiny_model().eval()
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones((1, 25), dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        model(torch.tensor([[1, 24]]))
    with pytest.raises(ValueError, match="binary"):
        model(torch.tensor([[1, 5]]), attention_mask=torch.tensor([[1, -1]]))
    with pytest.raises(ValueError, match="already shifted"):
        model(torch.tensor([[1, 5]]), targets=torch.tensor([[5]]))
    with pytest.raises(ValueError, match="one.*pair"):
        model(torch.tensor([[1]]), past_key_values=())
    with torch.no_grad():
        cache = model(torch.ones((1, 24), dtype=torch.long), use_cache=True)["past_key_values"]
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.tensor([[1]]), past_key_values=cache)


def test_presets_have_documented_capacity_without_claiming_trained_weights():
    ranges = {"smoke": (100_000, 300_000), "small": (1_000_000, 3_000_000), "base": (10_000_000, 30_000_000)}
    for name, (lower, upper) in ranges.items():
        model = NPTransformer(NPTransformerConfig.from_preset(name, vocab_size=64))
        assert lower <= model.num_parameters() <= upper
