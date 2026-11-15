"""Optional third-party API adapters, loaded only when explicitly invoked."""
from pathlib import Path


def admet_predict(smiles):
    try:
        from admet_ai import ADMETModel
    except ImportError as exc:
        raise RuntimeError("Install the separately licensed ADMET-AI package and its weights: pip install npfm[admet]") from exc
    return ADMETModel().predict(smiles=smiles)


def dock_vina(receptor, ligand, center, box_size, output, exhaustiveness=8, seed=42):
    try:
        from vina import Vina
    except ImportError as exc:
        raise RuntimeError("Install AutoDock Vina and supply prepared receptor/ligand PDBQT files") from exc
    for f in [receptor, ligand]:
        if not Path(f).is_file(): raise FileNotFoundError(f)
    v = Vina(sf_name="vina", seed=seed, cpu=1)
    v.set_receptor(str(receptor)); v.set_ligand_from_file(str(ligand))
    v.compute_vina_maps(center=list(center), box_size=list(box_size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=10)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    v.write_poses(str(output), n_poses=10, overwrite=True)
    return {"path": str(output), "calculated_energies": v.energies(n_poses=10).tolist(), "scope": "docking_scores_not_clinical_efficacy"}
