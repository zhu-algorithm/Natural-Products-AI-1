# Attribution and source boundaries

NP-GPT engineering code and this project's parameter artifacts are distributed with the project repository license. This does not relicense any underlying database or third-party component.

The actual pretraining input is the fixed public COCONUT/LOTUS union named by `corpus_manifest.json`, SHA256 `08765cdfebced158b308978f08dba106b878b6f87ea09c6e93da9c553a4c0685`. The union contains 764,701 source rows; the trained partition contains 688,166 eligible complete sequences. This directory contains model parameters and audit metadata, not the source database exports.

- **COCONUT, November 2026 export:** source [download service](https://coconut.naturalproducts.net/download), named upstream file `coconut_csv_lite-11-2026.zip`; registered source terms CC0-1.0. Cite the [COCONUT database paper](https://doi.org/10.1093/nar/gkae1063). Original collection attribution and natural-origin uncertainty remain relevant.
- **LOTUS, frozen 2026-04-13 export:** [Zenodo record 19360665](https://zenodo.org/records/19360665), named metadata file `260413_frozen_metadata.csv.gz`; this frozen channel is registered under CC BY 4.0. Cite the [LOTUS paper](https://doi.org/10.7554/eLife.70780) and the pinned dataset record. The separate Wikidata channel has different terms and is not silently substituted.
- **PyTorch and RDKit:** retain their respective software licenses. Standard Transformer components are attributed to their original papers in the model card. No external pretrained checkpoint was loaded into this model.

Database registration and a model checkpoint do not establish that every source structure occurs naturally. No source-table output is an experimental ADMET label, and no disease supervision was used in this run. Broader registry contents, commercial restrictions on other databases, and separately downloaded external model weights do not become part of this trained checkpoint.
