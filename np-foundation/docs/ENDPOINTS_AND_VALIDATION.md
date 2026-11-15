# 天然产物模型：证据、终点与验证

恢复日期：2026-11-15。本文件对应恢复后实际存在的源码与示例；工作区丢失前的下载数量、训练产物和测试结果不作为本次验证结果。

## 1. 当前实现的范围

`npfm.evidence` 实现带来源的证据检索；`npfm.evaluate` 实现缺失标签感知的基础指标及组别、骨架、时间拆分。它们是可运行工程组件，不是已经验证的天然产物临床疗效模型。

结构语料可供分子表示、自监督学习或结构生成使用。若要训练成药性和疾病适应症相关的预测头，还需要对应监督信号。应将以下三类信息分开保存：

| 信息类别 | 例子 | 可支持的结论 |
|---|---|---|
| 结构计算属性 | 分子量、logP估计、QED、结构警示 | 指定算法计算结果 |
| 实验终点 | 亲和力、溶解度、渗透性、代谢稳定性、细胞活性 | 对应实验条件下的观测 |
| 临床与监管证据 | 说明书适应症、指南、临床探索 | 对应分子、制剂、地区和研究阶段的证据 |

计算属性不充当实测 ADME、毒性或临床疗效标签。未知标签保留缺失；未在数据库出现并不证明没有活性。

## 2. 真实结构示例与来源

`examples/compounds.csv` 包含 36 个经过人工选择的天然产物示例，来自恢复时实际成功的 PubChem PUG REST 请求。每行提供 CID、完整 InChIKey、SMILES 和来源 URL；`examples/pubchem_provenance.jsonl` 保留服务返回的原始 JSON 对象与请求时间、HTTP 状态和响应字节 SHA-256。

这组数据用于安装检查、结构流程和自监督训练示例，**没有实验标签**。`label` 为空，`label_status` 为 `unlabeled`。示例集合也不是全部已知天然产物清单。天然来源分类是人工选择示例的说明，不能把一个 PubChem 结构标识本身当成天然来源确证。

本次 PubChem 响应字段与 CSV 的映射：

| PubChem 字段 | CSV 字段 |
|---|---|
| `SMILES` | `smiles`、`isomeric_smiles` |
| `ConnectivitySMILES` | `canonical_smiles` |
| `CID` | `cid`、`id`、`compound_id` |
| `InChIKey` | `inchikey` |

`canonical_smiles` 在此专指 PubChem 的 connectivity 表示；结构身份验证使用带立体信息的 `smiles` 和完整 InChIKey。不同化学工具可能生成不同但等价的 SMILES，不能仅凭 SMILES 文本不同判定结构不同。

可重复运行 `python examples/refresh_pubchem_examples.py`。该脚本优先读取 `examples/pubchem_cache` 的已成功响应；缺失时才请求 PubChem，控制并发并重试。删除特定缓存可重新取该条记录。固定实验应保留原始响应和时间，而不是静默覆盖发布版。

来源与接口：[PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)。每个示例的精确 URL 也在 CSV 和 JSONL 内。

## 3. 证据结构与身份边界

每条 JSONL 关系包含：

- `subject`、`subject_type`、`predicate`、`object`、`object_type`：有向关系与实体类型。
- `subject_aliases`、`object_aliases`：明确登记的等价名称，采用完整字符串匹配。
- `evidence_level`：`regulatory_label`、`guideline`、`clinical_trial`、`animal`、`cell`、`biochemical`、`mechanistic`、`computational` 或 `unknown`。
- `claim_status`：`positive`、`negative`、`mixed` 或 `unknown`。它描述该关系断言的状态，不是疗效概率。
- `context`：地区、制剂、实验体系、剂量/时间（有来源时）、研究设计等情境。
- `sources`：至少一个原始来源的标题和 HTTP(S) URL。

`approved_for` 只接受肯定的 `regulatory_label` 证据并要求 `context.jurisdiction`。`recommended_for`、`not_recommended_for` 要求指南层级。`hypothesized_for` 要求计算层级。检索记录不能填入数值化的 `clinical_efficacy_probability`。

分子、衍生物、制剂和联合方案是不同实体。别名匹配不会把 `artemisinin`、`artesunate` 和 ACT 合并。`related_records` 可展示直接相关背景，但不会传播适应症。未找到记录时返回 `unknown`，不产生阴性实验标签。

关系图支持化合物、靶点、通路、细胞状态和疾病节点。`trace_paths` 仅串接文件中实际存在的肯定有向边，并保留逐边上下文；它不填补缺边，也不把路径视为因果链或临床证明。跨实验、剂量、物种的上下文是否一致仍需另行核对。

### 使用 API

```python
from npfm.evidence import EvidenceStore

store = EvidenceStore.from_jsonl("examples/evidence.jsonl")
records = store.query("紫杉醇")
report = store.indication_report("artemisinin", "colorectal cancer")
paths = store.trace_paths("paclitaxel", "mitotic progression", max_hops=4)
```

`query` 返回直接关系列表。`indication_report` 把监管适应症、指南推荐、非推荐、临床探索、实验和计算假设分组；所有返回记录保留来源。此示例对青蒿素与结直肠癌返回未知；这只说明本示例没有该直接证据，不宣称完成全世界文献或监管审查。

## 4. 紫杉醇与青蒿素示例

17 条示例记录在恢复时重新检查来源。

**紫杉醇。** 保存美国常规静脉注射剂说明书中的卵巢癌、乳腺癌、非小细胞肺癌和 AIDS 相关卡波西肉瘤条目，并保留制剂及治疗阶段、联用等条件。微管蛋白结合与微管稳定的机制作为独立关系保存。不同制剂的说明书不能自动互换。来源：[DailyMed 说明书](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=59906bd2-8337-4cd8-9288-65b90eac2a7a)、[NCI 机制条目](https://www.cancer.gov/publications/dictionaries/cancer-drug/def/paclitaxel)。

**青蒿素。** 化学蛋白组学记录明确为疟原虫与标记探针的实验情境；该研究支持血红素活化和多蛋白结合，不能还原成唯一已确证的普适单靶点，也不产生肿瘤获批适应症。来源：[原始机制研究](https://pubmed.ncbi.nlm.nih.gov/26694030/)。

**ACT 与青蒿琥酯。** WHO 的联合疗法推荐登记在 ACT 方案实体上，口服青蒿素类单药的非推荐也以方案实体记录。青蒿琥酯结直肠癌研究登记为 23 人随机化的临床探索，不登记为批准，也不转移到青蒿素原体。来源：[WHO 治疗指导](https://www.who.int/teams/global-malaria-programme/case-management/treatment)、[原始随机先导试验](https://pubmed.ncbi.nlm.nih.gov/26137537/)。

## 5. 指标的实际实现

```python
from npfm.evaluate import evaluate_predictions, evaluate_multitask

# y_true / y_pred 来自相同留出样本；缺失标签可用 None 或 NaN。
result = evaluate_predictions(y_true, y_pred, task="regression")
classification = evaluate_predictions(binary_labels, probabilities, task="classification")
```

回归输出 MAE、MSE、RMSE；二分类输出 ROC-AUC、PR 曲线梯形面积、average precision、Brier 和 0.5 阈值准确率。PR-AUC 与 average precision 是分别计算的两个量，不能混用名称。

标签缺失与预测缺失均从指标对中排除，并报告 `n_total`、`n_labeled`、`n_evaluated`、标签覆盖率及有标签样本的预测覆盖率。缺失预测仍会降低覆盖率，不能通过丢弃失败样本掩盖结果。没有有效标签时指标为 `None`；仅一个观测类别时 ROC、PR 排序指标为 `None`。分类值必须位于 0 到 1。

`evaluate_multitask` 对每个终点分别掩码和计算，不把不同单位的回归误差平均成一个成药性指标。输出维度、单位、检测阈值和不等式测量应在训练数据规范中保留。当前组件没有实现删失回归、生存分析、序数终点或临床试验因果效应估计。

## 6. 拆分与泄漏控制

| 接口 | 实际行为 | 需要注意的范围 |
|---|---|---|
| `group_split(groups, fractions, seed)` | 整组分配，近似目标比例，重复调用可复现 | 组可以是化合物、实验室、研究或物种，需根据问题选择 |
| `scaffold_groups(smiles)` | RDKit Bemis–Murcko 骨架 | 默认不区分骨架立体信息；所有无环分子归同一组 |
| `scaffold_split(smiles, ...)` | 对真实骨架组拆分 | 非法结构报错，不退回随机拆分；大组可能造成比例偏差 |
| `time_split(timestamps, train_before, validation_before, groups=None)` | 按明确时间边界拆分 | 缺失时间报错；给定组后，整组分配到该组最新记录对应的分区 |

三个函数返回 `train`、`validation`、`test` 索引。时间区间为：训练 `< train_before`，验证 `>= train_before` 且 `< validation_before`，测试其余。组跨越日期时整组从训练中移出，避免同一身份同时进入训练和未来测试。

选择划分方式是研究设计的一部分。预测新结构需要骨架或结构家族留出；预测新蛋白还需要蛋白侧留出；跨实验室或跨物种应用应留出相应组。随机拆分药物-蛋白配对不能证明对双方都陌生时的泛化。

预处理拟合、目标归一化、特征选择和校准仅使用训练/验证信息。临床适应症任务还应控制关系图、文献和批准日期，防止把未来知识用于过去的测试。模型结果与检索命中须分别评估，不能把检索测试标签视为模型发现新知识。

## 7. 接入外部算法的顺序

`registry/endpoint_algorithms.json` 登记 16 类可复用方法及原始代码或方法来源。它们在当前工程中是外部参考；下载源码不意味着已经下载权重、完成复现或接入训练。

先用带明确单位的少数实验终点，建立指纹模型和 Chemprop 等基线，再比较本项目分子表示。随后加入分子-蛋白任务；有真实扰动表达数据后再比较 CPA、chemCPA 等细胞响应层。最后将经过时间控制的疾病、靶点、监管和文献证据引入候选排序，并保留证据检索的独立输出。

[Chemprop](https://github.com/chemprop/chemprop) 提供分子属性训练框架；[ADMET-AI v2](https://zenodo.org/records/18673143) 有版本明确的终点预测器。它们的任务覆盖和数据版本不同，不能仅凭“成药性评分”这一名称合并。

[ConPLex](https://github.com/samsledje/ConPLex)、[MolTrans](https://github.com/kexinhuang12345/MolTrans)、[DrugBAN](https://github.com/peizhenbai/DrugBAN) 支持药物-靶点方向的研究；[TxGNN](https://github.com/mims-harvard/TxGNN) 支持图谱上的用途排序。靶点相互作用和图谱分数都不是临床疗效概率。

[CMap 原始研究](https://pubmed.ncbi.nlm.nih.gov/29195078/)将药物和疾病表达签名用于连接性分析；[CPA](https://github.com/theislab/cpa)、[chemCPA](https://github.com/theislab/chemCPA)、[scGen](https://github.com/theislab/scgen)、[GEARS](https://github.com/snap-stanford/GEARS)、[State](https://github.com/ArcInstitute/state)、[MAP](https://github.com/MAGIC-AI4Med/MAP)对应不同的细胞扰动建模假设。基因敲除与化合物处理应分开，剂量、细胞类型、时间和批次不能省略。转录签名逆转需排除一般性细胞毒性的混杂，不直接等于疾病改善。

有阳性而缺少确认阴性时，可以研究 [nnPU](https://arxiv.org/abs/1703.00593)，但需要检查其阳性抽样、类别先验等条件。建议报告外部数据表现、适用域以及不确定性；[Deep Ensembles](https://arxiv.org/abs/1612.01474)是候选方法之一，而不是自动保证可靠性的标签。

当前评估模块只实现上述基础指标。温度缩放、保序校准、保形区间、组别 bootstrap 置信区间、OOD 基准及前瞻性外部验证仍需按实际终点另行实现；不能将未执行的验证写成已达到的性能。

## 8. 许可与恢复验证

代码、权重、原始数据和模型输出可能有不同许可。官方仓库可见不代表可以任意再分发。State 的代码与模型许可有非商业限制；scGen 源码遵守 GPL，不能改标成本工程的 MIT。DeepDTA 在恢复时未确认明确源码许可，继续保留待核实状态。具体来源链接与恢复时核查日期在注册表中。

恢复后执行：

```bash
PYTHONPATH=src /workspace/scratch/14a752057ad3/npfm-venv/bin/python -m pytest -q tests/test_evidence.py
```

2026-11-15 的实际运行退出码为 0，17 个测试函数完成。检查覆盖来源校验、适应症身份边界、未知与计算假设分离、路径不作临床推论、缺失标签指标、组别/骨架/时间拆分，以及全部 36 个实际 PubChem 结构的 RDKit 完整 InChIKey 一致性。这不是疾病预测性能评测，也不替代整个项目的集成测试。
