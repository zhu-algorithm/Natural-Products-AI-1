# 天然产物与药物研发数据目录

恢复核查日期：2026-11-15。此前工作目录在跨会话期间丢失，本次代码已重建、数据重新下载；历史数量不冒充现存文件。最终实际取得范围以 `npfm-data/raw/cumulative_manifest.json` 和 `npfm-data/metadata/delivery_data_summary.json` 为准。

本目录登记37个主要天然产物与互补药物研发资源。它不声称世界上所有天然化合物、所有数据库和所有上游来源已经收齐。

## 获取和完整性

下载器仅使用Python标准库，支持明确选择资源、全目录计划、许可门控、原子文件、超时/重试、SHA256缓存校验、发布方校验值、HTML错误页拒绝及压缩文件签名检查。计算得到的本地SHA256是完整性指纹，不是发布方数字签名。

```bash
PYTHONPATH=src python -m npfm.download list
PYTHONPATH=src python -m npfm.download plan --all
PYTHONPATH=src python -m npfm.download fetch --ids coconut,lotus,chebi,mibig,drugcentral --output ../npfm-data/raw
PYTHONPATH=src python -m npfm.download fetch --ids npatlas,cmnpd,sider --allow-noncommercial --output ../npfm-data/raw
PYTHONPATH=src python -m npfm.download manifest --output ../npfm-data/raw --verify-hashes
```

`--all`是审查注册表全部项目，不是绕过许可或声称全部下载成功。没有文件地址的项目显示manual；许可条件未满足显示blocked；失败显示failed。fetch遇到未完成项返回非零状态。`downloaded`仅表示明确注册的选定文件齐全，不能被扩写成数据库所有表/API对象/上游库齐全。

`--allow-unknown-license`允许在核对使用目的后获取公开文件，并不赋予商业、训练或再分发权。NPASS数据库许可未确认；THEOBROMA为逐条混合许可；IMPPAT为NC-ND。这些原件不进入公共GitHub，不能把文章许可或Zenodo顶层许可替代底层数据库条款。付费DNP/MarinLit不自动获取，不绕过登录或订阅。

每个成功原件有 `*.provenance.json`，记录URL、最终URL、版本、字节、SHA256、抓取时间、服务器头、许可和实际记录数。累计manifest遍历所有旁车，避免并行下载最后一个进程覆盖总体状态。历史失败保留，但随后有效成功会更新当前状态。

MIBiG官方 `mibig_json_4.0.tar.gz` 在先前验证中实际为未压缩TAR。注册表仅对该文件指定格式例外，并强制验证官方MD5。不会对任意损坏gzip放宽校验。

## 规范化与模型输入

最小模型输入是 `compound_id,smiles`。规范CSV同时保留完整 `full_inchikey`、`connectivity_id`、来源ID/原记录ID、名称、实体类型、证据类型、来源URL、版本、许可和原始属性JSON。CSV/TSV/GZIP/ZIP/JSONL及顶层JSON数组均按记录读取；SDF需要可选RDKit。

```bash
PYTHONPATH=src python -m npfm.data normalize ../npfm-data/raw/coconut/coconut_csv_lite-11-2026.zip --output ../npfm-data/normalized/coconut_compounds.csv --source-id coconut --source-version 2026-11 --license CC0-1.0 --properties molecular_weight,alogp,qed_drug_likeliness --compact --deduplicate
PYTHONPATH=src python -m npfm.data normalize ../npfm-data/raw/lotus/260413_frozen_metadata.csv.gz --output ../npfm-data/normalized/lotus_compounds.csv --source-id lotus --source-version 2026-04-13 --license CC-BY-4.0 --compact --deduplicate
PYTHONPATH=src python -m npfm.data merge ../npfm-data/normalized/coconut_compounds.csv ../npfm-data/normalized/lotus_compounds.csv --output ../npfm-data/normalized/coconut_lotus_union_compounds.csv
```

`--compact`省略重复嵌入的原始属性，以控制训练文件体积；原件继续保留。`--deduplicate`按完整27字符InChIKey去重，缺失时只按精确SMILES字符串去重；前14字符仅是连通性元数据，绝不用于合并立体异构体。缺失活性不会被标为inactive，部分key不会被升级为完整key。

SDF由RDKit解析并生成isomeric SMILES/完整InChIKey，转换信息保存在原始字段。失败记录进入missing_smiles计数，不生成假结构。保留输入盐型和立体标记，不额外执行去盐或互变异构体标准化。完整InChIKey并不能表达所有构象、同位素之外的微观状态或未指定立体中心；标准InChI也可能把某些互变异构体视作同一标识。完整key并集应解释为标识符层面的去重。跨库实体协调和盐型策略应在任务配置中另行声明。标识层次与标准InChI互变异构体规则参见[InChI Trust技术FAQ](https://www.inchi-trust.org/technical-faq/)。

### 联合语料与发生证据

联合CSV保持 `source_ids`、提供代表SMILES的 `preferred_source` 及来源许可。COCONUT优先，补入LOTUS独有完整key。相同完整key的多个原始记录保留在来源映射中。它是结构语料，不是全部实验关系表。

`metadata/coconut_lotus_occurrence_provenance.jsonl.gz`逐行记录COCONUT原始表与LOTUS基础/扩展表的来源文件、数据行号、ID、完整key及LOTUS物种、分类、DOI和人工核验信息。其JSON说明保存源文件hash和许可。原始文件+行号可定位省略的字段。基础与扩展LOTUS有重叠，映射行数不是独立实验数。

COCONUT lite本身没有物种/文献/collection发生关系，不能通过保留记录ID声称这些上游证据已恢复。COCONUT目录中的collection条目也不是各上游库均下载完成的证明。

本次固定LOTUS Zenodo导出声明CC BY 4.0，区别于Wikidata渠道CC0。联合语料必须保留相应署名。NC、NC-ND、混合和未核实许可源不加入默认开放联合CSV。

### Python接口

```python
from npfm.download import load_registry, plan_downloads, fetch_resources, build_manifest
from npfm.data import iter_records, normalize_record, export_records, merge_normalized_csvs, export_occurrence_map
```

## 数量与证据边界

分子、完整立体结构、连通性家族、发生关系、活性测量、谱图、构象和蛋白序列是不同分母。COCONUT、LOTUS、NPASS、NPAtlas等相互重叠，不能直接相加。一个天然化合物有结构或来源记录，也不能推导它有成药性或获批适应症。

| 输入证据 | 可以学习的任务 | 不能直接推断的结论 |
|---|---|---|
| 天然结构与物种来源 | 化学语言、分类和来源 | 临床有效、可成药 |
| 计算QED/SA/logP/NP-likeness | 描述符蒸馏和基线 | 实测ADME、真实临床成功率 |
| 体外靶点活性 | 指定实验背景的定量活性 | 人体疗效和批准适应症 |
| 提取物、草药、方剂 | 对应实体的关系和响应 | 每个单体都具有同一作用 |
| 细胞或动物扰动 | 特定模型系统的响应 | 人体疗效或长期安全 |
| 药品说明书和临床结果 | 特定产品/途径/人群证据 | 类似结构都具相同适应症 |

NPASS论文的9,713条ADME测量只覆盖744个NP，34,975条毒性覆盖3,662个NP。此前官方下载页只有11个3.0文件，未暴露独立ADME bulk链接；不能把论文ADME计数称为已取得训练标签。THEOBROMA二进制dump若未恢复计数，文章化合物数只作为发布统计。DrugCentral本次只选结构与靶点交互，不包含完整适应症库。

## 注册资源

下表为恢复时的资源目录；统计历史与当前实测状态分开保存。逐文件实际成功/失败请查累计manifest。

| ID | 资源/官方入口 | 内容与版本 | 许可 | 获取策略 | 训练用途 |
|---|---|---|---|---|---|
| coconut | [COCONUT](https://coconut.naturalproducts.net/download) | November 2026 export; Aggregated structures including varying natural-origin certainty, supplier compounds and derivatives | CC0-1.0 | 1个候选文件 | structure_pretraining, chemical_classification, provenance |
| lotus | [LOTUS](https://lotus.nprod.net/lotus-manuscript/) | Frozen 2026-04-13; Zenodo 19360665; Structure-organism-publication occurrence exports | CC-BY-4.0 | 2个候选文件 | structure_pretraining, natural_origin_provenance, taxonomy |
| theobroma | [THEOBROMA](https://theobroma.l3s.uni-hannover.de/) | v1.36 / 2026-09-17 dump; paper 2026-11-02; Aggregated structures with per-compound/source license annotations | MIXED | 2个候选文件 | structure_pretraining, license_provenance, stereoisomer_families |
| npass | [NPASS](https://bidd.group/NPASS/downloadnpass.html) | 3.0 / 2026 paper; NP structures, activities, toxicity, sources and engineered production evidence | UNVERIFIED | 11个候选文件 | activity_supervision, adme_toxicity_supervision, taxonomy |
| npatlas | [Natural Products Atlas](https://www.npatlas.org/download) | v2024_09 / website 3.0.2; Microbial NP structures, origin and biosynthesis annotations | CC-BY-NC-4.0 | 2个候选文件 | structure_pretraining, microbial_np, classification |
| cmnpd | [CMNPD](https://docs.cmnpd.org/downloads) | 1.0 / 2021 paper; Marine NP structures, calculated properties, measured activities | CC-BY-NC-SA-4.0 | 5个候选文件 | marine_np, structure_pretraining, activity_supervision |
| imppat | [IMPPAT](https://cb.imsc.res.in/imppat/statistics) | 3.0; batch files 2026-09-30; Plant parts, phytochemicals, therapeutic use, formulations and target evidence | CC-BY-NC-ND-4.0 | 10个候选文件 | taxonomy, traditional_use_evidence, target_evidence |
| npbs | [NPBS Atlas](https://biochemai.cstspace.cn/npbs/downloads/) | 2025 paper; Biological source and activity annotations for natural products | CC-BY-NC-4.0 | 3个候选文件 | natural_origin_provenance, taxonomy |
| mibig | [MIBiG](https://mibig.secondarymetabolites.org/download) | 4.0 / 2024-11-15; Experimentally characterized biosynthetic gene clusters and products | CC-BY-4.0 | 3个候选文件 | biosynthesis_multimodal, gene_cluster_product_mapping |
| chebi | [ChEBI](https://www.ebi.ac.uk/chebi/downloads) | Monthly FTP snapshot; resolved URL and Last-Modified pinned in sidecar; Biologically relevant chemical ontology, structures and identifiers | CC-BY-4.0 | 3个候选文件 | ontology, identifier_mapping, structure_pretraining |
| drugcentral | [DrugCentral](https://drugcentral.org/download) | Official download page dated 2026-09-25; path includes 2027; Small-molecule drug structures and target interactions | CC-BY-SA-4.0 | 2个候选文件 | drug_mechanisms, target_evidence |
| sider | [SIDER](https://sideeffects.embl.de/download/) | 4.1 / 2015-10-21; Drug labels, adverse events, frequency and indication relations | CC-BY-NC-SA-4.0 | 8个候选文件 | adverse_effect_supervision, indication_evidence |
| supernatural | [SuperNatural 3.0](https://bioinf-applied.charite.de/supernatural_3/) | 3.0 / 2023 paper; NPs and NP-derived supplier structures with calculated annotations | UNVERIFIED | Official website and FAQ returned HTTP 503 on 2026-11-14. No verified full CSV endpoint. | structure_pretraining, supplier_metadata |
| herb | [HERB 2.0](https://herb.ac.cn/) | 2.0 / 2025 paper; Ingredients, herbs, formulations, transcriptomics and clinical evidence | UNVERIFIED | Official service returned HTTP 503 on 2026-11-14. Keep herb/extract/formula/compound entities distinct. | perturbation_supervision, traditional_use_evidence, clinical_evidence |
| knapsack | [KNApSAcK](https://www.knapsackfamily.com/knapsack_core/top.php) | 2026-07-03 statistics; Metabolite occurrence in organisms | PROPRIETARY | Commercial permission required; use documented query/download arrangements. | taxonomy, natural_origin_provenance |
| gnps | [GNPS reference spectral libraries](https://ccms-ucsd.github.io/GNPSDocumentation/downloadlibraries/) | Rolling; pin chosen library; Reference tandem-mass spectra and metadata | MIXED | Some GNPS submissions are CC0; imported libraries retain separate terms. | spectra_structure_pretraining, dereplication |
| npmrd | [NP-MRD](https://np-mrd.org/downloads) | Versioned downloads; Natural product NMR structure and spectral records | CC-BY-NC-4.0 | Select experimental versus predicted spectra and record separate evidence types. | nmr_structure_pretraining, structure_validation |
| pubchem | [PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/downloads) | Rolling FTP/API; Compounds, substances and bioassay records from many depositors | MIXED | PUG REST/FTP subset; source-specific data rights remain relevant. | general_structure_pretraining, identifier_mapping, bioassay_supervision |
| chembl | [ChEMBL](https://www.ebi.ac.uk/chembl/) | Resolve and pin release; Quantitative assays, drugs, mechanisms, indications and ADME | CC-BY-SA-3.0 | Use official API/SQLite subsets. Whole collection intentionally not acquired. | activity_supervision, admet_supervision, indication_evidence |
| bindingdb | [BindingDB](https://bindingdb.org/rwd/bind/info.jsp) | Rolling download; Measured protein-ligand binding affinities | MIXED | Own data CC BY 3.0; imported ChEMBL has distinct share-alike terms. | affinity_supervision, target_prediction |
| opentargets | [Open Targets Platform](https://platform.opentargets.org/downloads) | Quarterly release; Target-disease associations with evidence provenance | CC0-1.0 | GraphQL or release Parquet; retain evidence type and strength. | disease_target_supervision, evidence_graph |
| ttd | [Therapeutic Target Database](https://idrblab.org/ttd/) | 2026 release; Therapeutic targets, drugs, disease and clinical stages | UNVERIFIED | Use official downloads after checking named release permission. | target_disease_graph, clinical_stage_supervision |
| lincs | [LINCS CMap L1000](https://clue.io/connectopedia/lincs_cmap_data) | GSE92742/GSE70138 and subsequent releases; Compound-induced cell transcriptomic profiles | MIXED | GEO/clue metadata and cell/dose/time context required; inferred genes not direct measurements. | cell_response_supervision, mechanism_similarity |
| jump | [JUMP Cell Painting](https://github.com/jump-cellpainting/datasets) | Versioned gallery releases; Chemical perturbation morphology images and profiles | CC0-1.0 | Download metadata/profile subset before large image bulk. | phenotype_pretraining, mechanism_similarity |
| toxcast | [ToxCast invitrodb](https://www.epa.gov/comptox-tools/exploring-toxcast-data) | v4.3 at prior inspection; High-throughput toxicity screening and dose-response | US-GOVERNMENT-OPEN-DATA | Retain assay interference, concentration, cytotoxicity and quality flags. | toxicity_supervision, dose_response |
| tox21 | [Tox21](https://tox21.gov/) | Challenge and subsequent assay releases; Toxicity screening measurements | CC0-1.0 | Select official release; not every assay outcome is a clinical safety label. | toxicity_supervision |
| toxval | [EPA ToxValDB](https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data) | Version-specific; Toxicological reference values from multiple sources | MIXED | Preserve species, route, duration, endpoint and original source restrictions. | toxicity_supervision |
| dailymed | [DailyMed](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) | REST API v2; rolling; Drug labeling, formulation, warnings and indications | US-GOVERNMENT-OPEN-DATA | Use selected natural drugs; labels attach to product/route/population, not arbitrary analogues. | approved_indication_evidence, formulation_context |
| clinicaltrials | [ClinicalTrials.gov](https://clinicaltrials.gov/data-api/how-download-study-records) | API v2; daily updates; Trial registrations, phases, outcomes and results | US-GOVERNMENT-OPEN-DATA | Trial registration or completion does not mean positive efficacy. | clinical_stage_evidence, indication_evidence |
| tcmid | [TCMID 2.0](https://academic.oup.com/nar/article/46/D1/D1117/4584630) | 2018 publication; Herbs, ingredients, prescriptions and targets | UNVERIFIED | Legacy service, permissions and bulk availability require verification. | traditional_use_evidence, hypothesis_graph |
| batman_tcm | [BATMAN-TCM 2.0](https://doi.org/10.1093/nar/gkad926) | 2024 publication; Traditional medicine interaction and target hypotheses | UNVERIFIED | Distinguish predicted from experimentally supported relationships. | target_hypothesis_graph |
| tcmsp | [TCMSP](https://tcmsp-e.com/tcmsp.php) | Legacy/live version; Herb-ingredient-target relationships and computed pharmacology | UNVERIFIED | Computed oral bioavailability/drug-likeness are not clinical labels. | traditional_use_evidence, computed_descriptors |
| uniprot | [UniProt](https://www.uniprot.org/help/downloads) | Release-specific; Protein sequences and functional annotations | CC-BY-4.0 | REST API or bulk subset; reviewed and unreviewed entries separate. | protein_encoder, target_evidence |
| pdb | [wwPDB/RCSB PDB](https://www.rcsb.org/docs/programmatic-access/file-download-services) | Weekly releases; Experimental macromolecular structures and ligands | CC0-1.0 | Retain structural resolution, missing atoms, ligand identities and artifacts. | structure_encoder, binding_geometry |
| pmc | [PubMed and PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Rolling; retain dates; Primary literature, open-access full text and metadata | MIXED | OA availability does not give identical training/commercial rights to every article. | evidence_retrieval, relation_extraction, text_molecule_alignment |
| dnp | [Dictionary of Natural Products](https://dnp.chemnetbase.com/) | Subscription release; Curated licensed natural-product structures and references | PROPRIETARY | Requires subscription and explicit rights; no automated restricted access. | licensed_structure_expansion |
| marinlit | [MarinLit](https://marinlit.rsc.org/) | Subscription release; Licensed marine natural-product chemistry and references | PROPRIETARY | Requires subscription and explicit rights; no automated restricted access. | licensed_marine_expansion |

## 原始论文与授权来源

注册表source_urls逐项保存官方文档、原始论文和下载页。核心来源：

- COCONUT：[官方下载](https://coconut.naturalproducts.net/download)、[论文](https://doi.org/10.1093/nar/gkae1063)。
- LOTUS：[冻结导出及许可](https://zenodo.org/records/19360665)、[原始论文](https://doi.org/10.7554/eLife.70780)。
- NPASS：[官方文件列表](https://bidd.group/NPASS/downloadnpass.html)、[2026论文](https://doi.org/10.1093/nar/gkaf1196)。
- NPAtlas：[发布/许可](https://www.npatlas.org/download)、[论文](https://doi.org/10.1093/nar/gkae1093)。
- CMNPD：[下载](https://docs.cmnpd.org/downloads)、[条款](https://docs.cmnpd.org/terms-and-conditions)。
- THEOBROMA：[官方archive](https://zenodo.org/records/22816330)、[论文](https://doi.org/10.1093/nar/gkag1083)。
- MIBiG：[官方发布与校验](https://zenodo.org/records/14835872)。
- IMPPAT：[官方batch下载](https://cb.imsc.res.in/imppat/download)。
- NPBS：[官方许可](https://biochemai.cstspace.cn/npbs/license/)。

## 发布与复现

公开代码、目录、模型定义、必要的来源manifest和说明，与原始数据包分别管理。原始/派生数据按具体来源与版本许可处理，保留引用。NC、NC-ND、混合、未核实和订阅原件不进入默认公共GitHub。模型权重商业许可也取决于真实训练数据、代码和底层权重。

关键验证命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_data_download.py -v
```

6项测试覆盖HTML错误页、缓存hash及篡改、立体key与跨来源合并、缺失标签、许可门控和累计manifest历史协调。此目录不含用户私人账户数据。

## 本次恢复完成状态

本次实际重新取得 **49个原始文件、2,390,324,546字节**；逐文件SHA256复核完成，完整性异常0项，未完成.part文件0项。状态：{"downloaded": 11, "partial": 0, "failed": 1, "manual": 25, "blocked": 0, "not_attempted": 0}。

| 资源 | 已取得/注册文件 | 字节 | 当前状态 |
|---|---:|---:|---|
| coconut | 1/1 | 199,108,039 | downloaded |
| lotus | 2/2 | 110,893,185 | downloaded |
| theobroma | 2/2 | 596,446,237 | downloaded |
| npass | 11/11 | 347,013,398 | downloaded |
| npatlas | 2/2 | 509,244,851 | downloaded |
| cmnpd | 5/5 | 78,531,033 | downloaded |
| imppat | 10/10 | 222,252,717 | downloaded |
| npbs | 0/3 | 0 | failed |
| mibig | 3/3 | 125,490,405 | downloaded |
| chebi | 3/3 | 146,346,040 | downloaded |
| drugcentral | 2/2 | 1,692,775 | downloaded |
| sider | 8/8 | 53,305,866 | downloaded |

COCONUT/LOTUS重新计数：728,389与227,298个完整key，交集190,986，并集**764,701**。联合CSV为238,371,765字节，SHA256 `08765cdfebced158b308978f08dba106b878b6f87ea09c6e93da9c553a4c0685`。来源映射2,122,987行，不是独立实验总数。

| 规范文件 | 原始读取行 | 导出结构行 | 缺失SMILES |
|---|---:|---:|---:|
| chebi_compounds.csv | 199,902 | 195,043 | 120 |
| cmnpd_compounds_NONCOMMERCIAL.csv | 31,561 | 31,526 | 0 |
| coconut_lotus_union_compounds.csv | 955,687 | 764,701 | 0 |
| coconut_compounds.csv | 738,793 | 728,389 | 0 |
| drugcentral_compounds.csv | 4,310 | 4,240 | 0 |
| lotus_compounds.csv | 709,740 | 227,298 | 0 |
| npass_compounds_RESTRICTED.csv | 203,390 | 203,290 | 0 |
| npatlas_compounds_NONCOMMERCIAL.csv | 36,454 | 36,422 | 0 |

没有把数据库重叠记录加总为所有已知天然化合物。COCONUT 71个collection条目已重新取得为目录metadata，并未单独下载71库。此次gzip映射重新生成，压缩头时间可改变压缩文件hash；以当前metadata为准。
