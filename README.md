# AI Algorithm for Natural Compound Screening and Optimization

An AI-driven framework for reverse target prediction and multi-agent synergy screening of natural compounds, combining protein structure prediction, molecular docking, and network pharmacology to support drug repurposing and combination-therapy design.

## Overview

Natural products commonly face a well-known R&D bottleneck: **multiple targets, unclear mechanisms of action, and modest single-agent activity**. Traditional wet-lab screening is slow and costly. This project builds an end-to-end computational framework organized around two directions — **reverse target prediction** and **synergy-based combination screening** — to help researchers quickly identify plausible mechanisms of action for natural compounds and design combination regimens with stronger efficacy and lower toxicity, providing prioritized, mechanism-backed hypotheses for downstream wet-lab validation and investigator-initiated trials (IITs).

## Core Research Directions

### Direction 1: Reverse AI Target Prediction for Natural Compounds

Aimed at "drug repurposing and mechanism elucidation," this direction fuses protein 3D structure prediction, molecular docking, and transcriptome–proteome protein-protein interaction (PPI) network modeling to reverse-infer the potential targets of natural compounds.

- **Priority compounds**: Artesunate, Resveratrol, β-glucan
- **Anti-fibrosis & NASH**: Predicts novel targets involved in hepatic stellate cell activation, lipid metabolism regulation, and key inflammatory pathways (TGF-β/Smad, Wnt/β-catenin)
- **Tumor microenvironment remodeling**: Characterizes how β-glucan modulates macrophage polarization (M1/M2) and NK cell activating receptors, supporting IIT trials in NASH and hepatocellular carcinoma (HCC)

### Direction 2: AI Screening for Natural Product + Drug Synergy Combinations

Covers three combination-therapy scenarios, using an AI-computed synergy score to guide regimen design and patent strategy.

| Scenario | Combination | Screening Objective |
| --- | --- | --- |
| Multi-component natural product synergy | Artesunate + Resveratrol / β-glucan | Compute a multi-target Synergy Score to enhance overall anti-inflammatory and anti-fibrotic effects |
| Natural product + targeted therapy potentiation | Artesunate + Sorafenib / Lenvatinib | Predict synergy nodes and novel targets that overcome resistance to targeted therapy |
| Natural product + immune checkpoint inhibitor | β-glucan + anti-PD-1/PD-L1 | Assess synergistic anti-tumor response rate and predict reduced immune-related adverse events (irAEs) |

## Algorithm Pipeline

```
Raw Data Input
  ├─ Compound structure (SMILES / 3D conformation)
  ├─ Protein sequence & structure databases
  └─ Transcriptomic / proteomic expression profiles
        │
        ▼
① Protein 3D structure prediction ──▶ ② Molecular docking (compound–target binding score)
        │                                          │
        ▼                                          ▼
③ Transcriptome–proteome PPI network modeling ──▶ ④ Reverse target prediction & ranking (Direction 1 output)
        │
        ▼
⑤ Multi-component / multi-drug synergy modeling (Synergy Score)
        │
        ▼
⑥ Candidate targets & combination regimens output (for wet-lab / IIT validation)
```

## Key Modules

- **Reverse Target Prediction Module**: Takes compound structure as input, outputs a ranked list of candidate targets with confidence scores
- **Molecular Docking Engine**: Computes binding energy and binding modes between compounds and candidate targets
- **PPI Network Analysis Module**: Integrates transcriptomic and proteomic data to build pathway-level mechanistic hypotheses
- **Synergy Scoring Module**: Quantifies synergistic, additive, or antagonistic relationships across multi-component / multi-drug combinations
- **Pathway Enrichment & Visualization**: Surfaces key pathways (e.g., TGF-β/Smad, Wnt/β-catenin) and immune-cell-polarization evidence chains

## Input / Output

**Input**
- Compound structure files (SMILES, SDF, or 3D conformation)
- Target disease / pathway context (optional, narrows prediction scope)
- Candidate drug list for combination screening (Direction 2)

**Output**
- Ranked list of candidate targets (with binding scores, pathway annotations, confidence levels)
- Ranked table of synergistic combinations (with Synergy Score and predicted mechanism)
- Mechanism hypothesis report suitable for wet-lab design or IIT proposal support

## Use Cases

- Mechanism elucidation and target hypothesis generation for natural product drug repurposing
- Computationally prioritized ranking of combination-therapy candidates
- Translational research support in anti-fibrosis, NASH, and tumor immunology
- Early-stage evidence support for research proposals, patent strategy, and clinical trial design

## Limitations & Disclaimer

All outputs from this framework are computational predictions intended to generate mechanistic hypotheses and prioritized candidate lists. **They are not a substitute for wet-lab validation or clinical trial conclusions.** Any candidate target or combination regimen must undergo laboratory validation (e.g., in vitro binding assays, cell / animal model studies) and necessary ethical review before clinical application.

## Roadmap

- Expand the compound library with additional natural product structures and bioactivity data
- Incorporate wet-lab feedback data to continuously calibrate model prediction accuracy
- Extend prediction capability to ASEAN-region natural medicine resources (e.g., Tongkat Ali, cat's whiskers)
