"""Real 2D chemical features, with an explicit optional-RDKit boundary."""
from __future__ import annotations

from functools import lru_cache
import torch

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import rdBase
except ImportError:
    Chem = MurckoScaffold = rdBase = None

ATOM_FEATURE_DIM = 12


def rdkit_available():
    return Chem is not None


def rdkit_version():
    return rdBase.rdkitVersion if rdBase is not None else None


def _mol(smiles):
    if Chem is None:
        raise RuntimeError("RDKit is required for molecular graph features; no synthetic graphs are substituted.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty molecular SMILES: {smiles[:100]}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


@lru_cache(maxsize=65536)
def canonical_smiles(smiles, stereochemistry=True):
    if Chem is None:
        return smiles
    return Chem.MolToSmiles(_mol(smiles), isomericSmiles=stereochemistry, canonical=True)


@lru_cache(maxsize=65536)
def structure_group(smiles):
    """Non-stereo Murcko scaffold, falling back to connectivity for acyclic molecules."""
    if Chem is None:
        return smiles
    mol = _mol(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or Chem.MolToSmiles(mol, isomericSmiles=False)


def molecule_graph(smiles, max_atoms=None):
    mol = _mol(smiles)
    n = mol.GetNumAtoms()
    if max_atoms is not None and n > max_atoms:
        raise ValueError(f"Molecule has {n} atoms, above max_atoms={max_atoms}; no graph truncation.")
    features = []
    for atom in mol.GetAtoms():
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        features.append([
            atom.GetAtomicNum() / 100., atom.GetDegree() / 8., atom.GetFormalCharge() / 5.,
            atom.GetTotalNumHs() / 8., float(atom.GetIsAromatic()), float(atom.IsInRing()),
            atom.GetMass() / 250., int(atom.GetHybridization()) / 8., int(atom.GetChiralTag()) / 4.,
            float(cip == "R"), float(cip == "S"), atom.GetIsotope() / 250.,
        ])
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = float(bond.GetBondTypeAsDouble())
    return torch.tensor(features, dtype=torch.float32), adjacency


def batch_graphs(smiles, max_atoms=None, device=None):
    graphs = [molecule_graph(s, max_atoms=max_atoms) for s in smiles]
    if not graphs:
        raise ValueError("Cannot build an empty graph batch.")
    n = max(len(x) for x, _ in graphs)
    atoms = torch.zeros((len(graphs), n, ATOM_FEATURE_DIM))
    adjacency = torch.zeros((len(graphs), n, n))
    mask = torch.zeros((len(graphs), n), dtype=torch.bool)
    for i, (x, a) in enumerate(graphs):
        atoms[i, :len(x)] = x
        adjacency[i, :len(x), :len(x)] = a
        mask[i, :len(x)] = True
    result = {"atoms": atoms, "adjacency": adjacency, "mask": mask}
    return {k: v.to(device) for k, v in result.items()} if device is not None else result
