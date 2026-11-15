"""Scientific correctness invariants, rather than matching implementation details."""
import pytest
import torch

from npfm.features import molecule_graph, rdkit_available, structure_group
from npfm.model import NaturalProductModel
from npfm.tokenizer import SmilesTokenizer
from npfm.train import build_task_states, split_rows, supervised_loss, multimodal_contrastive_loss


def test_tokenizer_preserves_stereo_and_ring_syntax():
    smiles = ["C[C@H](O)C(=O)O", "ClC1=CC=CC=C1", "C%12CCCCC%12"]
    tok = SmilesTokenizer.fit(smiles)
    for value in smiles:
        assert tok.decode(tok.encode(value)) == value
    with pytest.raises(ValueError, match="No silent|no silent"):
        tok.encode(smiles[0], max_length=3)


def test_missing_binary_label_is_not_negative():
    outputs = {"active": torch.tensor([[0.0], [100.0]], requires_grad=True)}
    targets = {"active": torch.tensor([1., 0.])}
    masks = {"active": torch.tensor([True, False])}
    loss = supervised_loss(outputs, targets, masks, [{"name": "active", "kind": "binary"}])
    assert abs(float(loss.detach()) - .693147) < 1e-5
    loss.backward()
    assert outputs["active"].grad[1].item() == 0


def test_disease_head_needs_provenance_and_labels():
    with pytest.raises(ValueError, match="real label_source"):
        build_task_states([{"name": "cancer", "kind": "binary", "domain": "disease"}], [{"smiles": "CC"}])
    task = {"name": "cancer", "kind": "binary", "domain": "disease", "label_source": "curated_example", "label_evidence": "curated_association"}
    state = build_task_states([task], [{"cancer": ""}, {"cancer": "NA"}])[0]
    assert not state["trainable"] and not state["trained"]


def test_zero_weight_task_never_becomes_a_trained_head():
    rows = [{"active": "0"}, {"active": "1"}]
    task = {"name": "active", "kind": "binary", "weight": 0}
    state = build_task_states([task], rows)[0]
    assert not state["trainable"] and not state["trained"]
    assert "weight=0" in state["untrained_reason"]
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_task_states([{**task, "weight": -1}], rows)


@pytest.mark.skipif(not rdkit_available(), reason="RDKit graph tests require the optional dependency")
def test_graph_encodes_real_connectivity_and_stereochemistry():
    atoms, adj = molecule_graph("CCO")
    assert atoms.shape == (3, 12)
    assert adj[0, 2].item() == 0 and adj[0, 1].item() == 1
    assert molecule_graph("C[C@H](O)C(=O)O")[0].ne(molecule_graph("C[C@@H](O)C(=O)O")[0]).any()


@pytest.mark.skipif(not rdkit_available(), reason="Chemical identity grouping requires RDKit")
def test_split_keeps_duplicate_and_same_scaffold_together():
    smiles = ["c1ccccc1", "Oc1ccccc1", "C1CCCCC1", "C1CCNCC1", "c1ccncc1", "CCO", "OCC", "CCN", "CCC", "CCCC"]
    rows = [{"compound_id": str(i), "smiles": s} for i, s in enumerate(smiles)]
    parts, _ = split_rows(rows, seed=21)
    locations = {i: part for part, indices in parts.items() for i in indices}
    assert locations[0] == locations[1] and locations[5] == locations[6]
    groups = [{structure_group(rows[i]["smiles"]) for i in indices} for indices in parts.values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_causal_transformer_cannot_read_future_tokens():
    torch.manual_seed(1)
    model = NaturalProductModel(vocab_size=12, max_seq_length=16, d_model=16, nhead=2,
                                num_layers=1, dim_feedforward=32, dropout=0).eval()
    a, b = torch.tensor([[1, 5, 6, 7, 2]]), torch.tensor([[1, 5, 6, 9, 2]])
    with torch.no_grad():
        x, y = model(a, causal=True, lm_only=True)["lm_logits"], model(b, causal=True, lm_only=True)["lm_logits"]
    assert torch.allclose(x[:, :3], y[:, :3], atol=1e-6)


def test_graph_checkpoint_refuses_missing_graph():
    model = NaturalProductModel(vocab_size=10, d_model=16, nhead=2, num_layers=1, use_graph=True)
    with pytest.raises(ValueError, match="requires real RDKit graphs"):
        model(torch.tensor([[1, 5, 2]]))


@pytest.mark.skipif(not rdkit_available(), reason="RDKit is required for graph contrastive pretraining")
def test_unsupervised_contrastive_loss_trains_graph_and_fusion():
    from npfm.features import batch_graphs
    tok = SmilesTokenizer.fit(["CCO", "CCN"])
    model = NaturalProductModel(vocab_size=len(tok), d_model=16, nhead=2, num_layers=1, dropout=0, use_graph=True)
    ids = torch.tensor([tok.encode("CCO"), tok.encode("CCN")])
    result = model(ids, graph=batch_graphs(["CCO", "CCN"]), causal=True)
    loss = multimodal_contrastive_loss(result["sequence_embedding"], result["graph_embedding"], result["embedding"], ["CCO", "CCN"])
    loss.backward()
    assert model.graph_encoder.input.weight.grad.abs().sum() > 0
    assert model.fusion[0].weight.grad.abs().sum() > 0
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert multimodal_contrastive_loss(result["sequence_embedding"], result["graph_embedding"], result["embedding"], ["CCO", "CCO"]) is None
