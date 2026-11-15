"""RDKit-backed calculations. No calculated value is an experimental label."""
from pathlib import Path
import csv


def _rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors, rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError("Install npfm[chem] to use chemistry functions") from exc
    return Chem, DataStructs, AllChem, Descriptors, Lipinski, QED, rdMolDescriptors, rdFingerprintGenerator


def molecule(smiles):
    Chem = _rdkit()[0]
    m = Chem.MolFromSmiles(str(smiles))
    if m is None or m.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty SMILES")
    return m


def canonicalize(smiles):
    Chem = _rdkit()[0]
    return Chem.MolToSmiles(molecule(smiles), canonical=True, isomericSmiles=True)


def descriptors(smiles):
    Chem, _, _, D, L, Q, R, _ = _rdkit()
    m = molecule(smiles)
    return {
        "smiles": Chem.MolToSmiles(m, canonical=True, isomericSmiles=True),
        "full_inchikey": Chem.MolToInchiKey(m),
        "molecular_weight": D.MolWt(m), "calculated_logP": D.MolLogP(m),
        "tpsa": R.CalcTPSA(m), "hbd": L.NumHDonors(m), "hba": L.NumHAcceptors(m),
        "rotatable_bonds": L.NumRotatableBonds(m), "ring_count": R.CalcNumRings(m),
        "fraction_csp3": R.CalcFractionCSP3(m), "qed": Q.qed(m),
        "heavy_atoms": m.GetNumHeavyAtoms(), "formal_charge": Chem.GetFormalCharge(m),
        "value_type": "calculated_descriptor_not_experimental",
    }


def similarity(a, b, radius=2, nbits=2048):
    _, DS, _, _, _, _, _, F = _rdkit()
    gen = F.GetMorganGenerator(radius=radius, fpSize=nbits, includeChirality=True)
    return float(DS.TanimotoSimilarity(gen.GetFingerprint(molecule(a)), gen.GetFingerprint(molecule(b))))


def conformers(smiles, output, n_conformers=10, seed=42):
    Chem, _, A, *_ = _rdkit()
    m = Chem.AddHs(molecule(smiles))
    params = A.ETKDGv3(); params.randomSeed = seed; params.pruneRmsThresh = 0.5
    ids = list(A.EmbedMultipleConfs(m, numConfs=n_conformers, params=params))
    if not ids:
        raise ValueError("No conformer was embedded")
    method = "MMFF94" if A.MMFFHasAllMoleculeParams(m) else "UFF"
    if method == "MMFF94":
        energies = A.MMFFOptimizeMoleculeConfs(m, numThreads=1)
    else:
        energies = A.UFFOptimizeMoleculeConfs(m, numThreads=1)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(target)) as writer:
        for cid, (status, energy) in zip(ids, energies):
            m.SetProp("force_field", method)
            m.SetProp("minimization_status", str(status))
            m.SetProp("calculated_force_field_energy", str(energy))
            writer.write(m, confId=cid)
    return {"path": str(target), "conformers": len(ids), "force_field": method}


def describe_csv(input_path, output_path):
    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ident = {k: row.get(k, "") for k in ["compound_id", "name"]}
            try:
                rows.append({**ident, **descriptors(row["smiles"]), "error": ""})
            except (ValueError, KeyError) as exc:
                rows.append({**ident, "smiles": row.get("smiles", ""), "error": str(exc)})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return {"records": len(rows), "valid": sum(not r["error"] for r in rows), "path": str(out)}
