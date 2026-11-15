# 算法与药化软件目录

目录恢复日期：2026-11-15；官方来源与许可审计日期：2026-11-14；SAFE和MegaMolBART源码许可于2026-11-15补核。机器可读记录见 `registry/algorithms.json`。共57项，涵盖模型、分子表示、研究框架与药化软件，不是57个大语言模型，也不是57项已完成集成运行的算法。

## 本仓库实际实现

`src/npfm/` 是独立实现的紧凑SMILES Transformer、真实RDKit二维图消息传递、序列/图融合、因果或掩码语言预训练、可开关的多正例跨模态对比学习，以及带缺失标签掩码的回归、二分类与多分类任务头。它不是NaFM、Uni-Mol、NPGPT等外部算法的复现，也没有把这些方法的权重拼接成新模型。

因果语言输出只依赖序列，完整图结构不会泄露未来token。对比损失对齐序列、图和融合表征；同一立体化学结构的不同SMILES属于多正例，仅有一种结构的batch跳过对比损失。训练清单记录图与融合分支是否实际获得监督或对比训练更新。没有RDKit时，auto模式明确降级为序列模型，关闭图对比目标；化学骨架拆分也明确降级为字符串身份分组，不能宣称做过化学骨架验证。required模式则直接报错。

每个任务必须有实际标签。缺失值不改成阴性；数据不足或损失权重为零的头不会被标成已训练。疾病、临床或适应症任务还必须注明标签来源和证据等级。分类概率默认未校准，不能解读为临床成功概率。原型将选取的CSV记录载入内存，尚不是分布式全量语料训练器。

恢复版使用 `model_format_version=2`。此前丢失的训练权重、指标与精确实现不能根据文字记忆伪造；新权重必须由当前代码实际训练。不得把旧权重文件与新生成的词表、配置或指标混合。

## 优先组合

原型训练与基准：RDKit、Morgan/RF/XGBoost、Chemprop以及本仓库序列/图融合模型。天然产物专用扩展：NaFM、NPGPT、NPCLMs、NP-VAE、FragFM/NPGen。跨模态扩展：Uni-Mol2、BioT5+、MolCA。药化工作流：mmpdb、REINVENT4、AiZynthFinder、Vina/GNINA、PoseBusters、OpenMM/OpenFE与BoTorch。每个扩展都需要独立安装、适配、许可核验与验证。

## 全部目录

P0表示优先建立或作为必要基线；P1表示下一阶段扩展；P2表示研究对照。优先级是工程判断，不是性能排行榜。`source_included=false`是注册表的保守默认值，实际归档内容以源码下载和构建manifest为准。

| ID | 模型/工具 | 主要用途 | 许可核验与限制 | 优先级 |
|---|---|---|---|---|
| ALG-001 | [NaFM](https://github.com/TomAIDD/NaFM-Official) | 天然产物专用表征；天然产物类别/通路、活性回归等下游表征 | 代码MIT；数据与权重独立核验 | P0 |
| ALG-002 | [NPGPT](https://github.com/ohuelab/npgpt) | 天然产物专用生成；天然产物样新结构 | 代码MIT；上游模型与数据独立核验 | P0 |
| ALG-003 | [NPCLMs (Mamba/Mamba-2/GPT)](https://github.com/rozariwang/CLMs-for-NPs) | 天然产物专用语言模型；结构生成、性质和活性任务表征 | 本轮官方页未见明确license；待核验 | P1 |
| ALG-004 | [NP-VAE](https://github.com/toshikiochiai/NPVAE) | 天然产物专用潜空间；潜空间、重构与结构探索 | 定制许可：学术免费；商业使用需联系作者 | P1 |
| ALG-005 | [FragFM / NPGen](https://github.com/lee-jwon/FragFM) | 片段分层生成；新分子图与COCONUT衍生NPGen基准 | 代码MIT；数据独立核验 | P1 |
| ALG-006 | [BGC-MLM](https://github.com/aswalker-lab/BGC-MLM) | 生物合成来源表征；产物类别、活性、性质和官能团/指纹预测 | 代码MIT；权重与数据独立核验 | P1 |
| ALG-007 | [SELFIES](https://github.com/the-matter-lab/selfies) | 分子字符串表示；可解码分子字符串 | 代码Apache-2.0；依赖另核验 | P0 |
| ALG-008 | [SAFE / SAFE-GPT](https://github.com/datamol-io/safe) | 片段字符串表示与生成；片段组织、骨架装饰、连接和生成 | 代码Apache-2.0（仅源码）；数据与权重许可见独立字段；数据CC BY4.0；SAFE-GPT权重CC BY-NC4.0 | P1 |
| ALG-009 | [NPClassifier](https://github.com/mwang87/NP-Classifier) | 天然产物分层本体分类；Pathway/Superclass/Class | 代码MIT；分类器权重和数据独立核验 | P0 |
| ALG-010 | [ChemBERTa / ChemBERTa-2](https://github.com/seyonechithrananda/bert-loves-chemistry) | 通用分子语言编码器；嵌入与性质分类/回归 | 代码MIT；具体权重独立核验 | P1 |
| ALG-011 | [ChemBERTa-3](https://github.com/deepforestsci/chemberta3) | 分子预训练与基准框架；模型训练与比较评测 | 代码MIT；底层数据与权重独立核验 | P0 |
| ALG-012 | [MoLFormer / MoLFormer-XL](https://github.com/IBM/molformer) | 通用分子语言编码器；分子嵌入与性质预测 | 代码Apache-2.0；具体权重独立核验 | P0 |
| ALG-013 | [GP-MoLFormer](https://github.com/IBM/gp-molformer) | 通用自回归生成；新结构与局部优化 | 具体代码和权重许可本轮未逐项核验 | P1 |
| ALG-014 | [Chemformer / MolBART](https://github.com/MolecularAI/Chemformer) | 分子序列编码解码；重构、反应、逆合成与分子优化 | 代码MIT；模型与数据独立核验 | P1 |
| ALG-015 | [MegaMolBART](https://github.com/NVIDIA/MegaMolBART) | 分子序列编码解码；潜空间、生成和插值 | 该仓库固定commit源码Apache-2.0；NGC权重/容器与训练数据独立核验 | P2 |
| ALG-016 | [ChemGPT](https://github.com/ncfrey/litmatter) | 通用分子自回归生成；新分子序列 | 具体代码与模型许可本轮未逐项核验 | P2 |
| ALG-017 | [Chemprop / D-MPNN](https://github.com/chemprop/chemprop) | 监督性质与活性模型；ADMET/活性等端点分类回归 | 代码MIT | P0 |
| ALG-018 | [GROVER](https://github.com/tencent-ailab/grover) | 通用图预训练；图嵌入与性质 | 代码MIT；保留Chemprop MIT版权声明；权重/数据独立核验 | P2 |
| ALG-019 | [GraphMVP](https://github.com/chao1224/GraphMVP) | 2D/3D跨模态预训练；几何感知表征 | 代码MIT | P1 |
| ALG-020 | [MolCLR](https://github.com/yuyangw/MolCLR) | 图对比预训练；分子性质表征 | 代码MIT | P2 |
| ALG-021 | [Uni-Mol / Uni-Mol2](https://github.com/deepmodeling/Uni-Mol) | 三维分子基础模型；性质、分子与口袋表征 | 代码MIT；数据与具体权重独立核验 | P0 |
| ALG-022 | [MolT5](https://github.com/blender-nlp/MolT5) | 结构与文本对齐；分子描述和文本条件结构 | 代码BSD-3-Clause；底层语料独立核验 | P1 |
| ALG-023 | [BioT5 / BioT5+](https://github.com/QizhiPei/BioT5) | 分子-蛋白-文本多模态；描述、结构生成与生物任务 | 代码MIT；各语料与具体模型条款独立核验 | P0 |
| ALG-024 | [MolCA](https://github.com/acharkq/MolCA) | 分子图与语言模型连接；文本检索与分子描述 | 本轮官方页未见明确license；待核验 | P1 |
| ALG-025 | [MoleculeSTM](https://github.com/chao1224/MoleculeSTM) | 结构-文本对比对齐；检索和文本引导分子编辑 | NVIDIA自定义许可；PubChem关联文本权属须独立核验 | P1 |
| ALG-026 | [LlaSMol / SMolInstruct](https://github.com/OSU-NLP-Group/LLM4Chem) | 化学指令微调；化学问答、性质、反应与生成 | 代码MIT；底座模型及数据独立核验 | P1 |
| ALG-027 | [MolGen](https://github.com/zjunlp/MolGen) | 反馈约束分子生成；生成/优化候选 | 代码MIT；模型和数据独立核验 | P2 |
| ALG-028 | [DiffDock / DiffDock-L](https://github.com/gcorso/DiffDock) | 结合姿态预测；结合姿态与置信排序 | 代码和模型权重MIT（官方README） | P1 |
| ALG-029 | [TargetDiff](https://github.com/guanjq/targetdiff) | 口袋条件三维生成；原子类型/坐标与配体候选 | 本轮官方页未见明确license；待核验 | P2 |
| ALG-030 | [DiffSBDD](https://github.com/arneschneuing/DiffSBDD) | 口袋条件三维生成；新配体与局部补全 | 代码MIT；数据与权重独立核验 | P2 |
| ALG-031 | [FlowMol3](https://github.com/Dunni3/FlowMol) | 三维分子flow matching；三维新小分子 | 具体许可本轮未逐项核验 | P2 |
| ALG-032 | [REINVENT4](https://github.com/MolecularAI/REINVENT4) | 多目标强化学习优化；装饰、连接、替换与优化候选 | 代码Apache-2.0；各模型/打分器独立核验 | P0 |
| ALG-033 | [RxnFlow](https://github.com/SeonghwanSeo/RxnFlow) | 合成感知GFlowNet；多样高分候选及合成轨迹 | 代码MIT；砌块、反应库及权重独立核验 | P1 |
| ALG-034 | [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) | 逆合成规划；多步逆合成候选路线 | 代码MIT；反应模型和库存独立核验 | P0 |
| ALG-035 | [SynFormer](https://github.com/wenhao-gao/synformer) | 可合成分子生成；模板可构建类似物和路线 | 代码Apache-2.0；Enamine预处理砌块数据仅研究，商业需权限 | P1 |
| ALG-036 | [RetroBioCat](https://github.com/willfinnigan/retrobiocat) | 酶催化逆合成；反应级联与候选酶 | 原始论文明确代码MIT | P1 |
| ALG-037 | [RDKit / ETKDGv3 / BRICS](https://github.com/rdkit/rdkit) | 结构标准化、构象与片段；标准结构、构象集、片段和描述符 | RDKit主体BSD-3-Clause；Contrib及依赖独立核验 | P0 |
| ALG-038 | [Morgan fingerprints + XGBoost](https://github.com/dmlc/xgboost) | 传统QSAR基线；性质和活性分类回归 | XGBoost代码Apache-2.0；RDKit指纹BSD-3-Clause | P0 |
| ALG-039 | [antiSMASH](https://github.com/antismash/antismash) | 天然产物BGC识别；BGC类型和生物合成注释 | AGPLv3+ | P1 |
| ALG-040 | [NP-likeness score](https://github.com/rdkit/rdkit) | 天然产物相似性辅助评分；天然产物样程度 | 具体Contrib代码与评分模型许可需独立核验 | P0 |
| ALG-041 | [AutoDock Vina](https://github.com/forlilab/AutoDock-Vina) | 分子对接与虚拟筛选；候选结合姿态及对接分数 | 代码Apache-2.0 | P1 |
| ALG-042 | [GNINA](https://github.com/gnina/gnina) | 深度学习对接重评分；姿态和CNN分数 | Apache与GPL；默认OpenBabel依赖涉及GPL，不能标Apache-only | P1 |
| ALG-043 | [OpenMM](https://github.com/openmm/openmm) | 分子动力学；轨迹、能量及动力学统计 | 分组件MIT/LGPL；保留发行版许可证 | P1 |
| ALG-044 | [GROMACS](https://github.com/gromacs/gromacs) | 分子动力学；轨迹、能量与分析 | LGPL-2.1-or-later | P1 |
| ALG-045 | [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit) | 小分子力场参数化；力场参数与系统信息 | 代码MIT；可选商业依赖独立核验 | P1 |
| ALG-046 | [PySCF](https://github.com/pyscf/pyscf) | 量子化学；电子能量、轨道和性质 | 代码Apache-2.0；外部库独立核验 | P2 |
| ALG-047 | [xTB](https://github.com/grimme-lab/xtb) | 半经验量子化学；几何、能量及电子性质 | LGPL-3.0；发行版亦含GPL许可文本，保留组件说明 | P1 |
| ALG-048 | [mmpdb](https://github.com/rdkit/mmpdb) | 配对分子构效关系；局部化学变换与性质变化统计 | BSD-3-Clause；部分源文件保留MIT来源 | P0 |
| ALG-049 | [BoTorch](https://github.com/meta-pytorch/botorch) | 主动学习与多目标贝叶斯优化；下一批实验/候选优先级 | 代码MIT | P1 |
| ALG-050 | [OpenFE](https://github.com/OpenFreeEnergy/openfe) | 炼金自由能计算；自由能差及不确定性 | 代码MIT；底层MD和力场组件各自许可 | P1 |
| ALG-051 | [alchemlyb](https://github.com/alchemistry/alchemlyb) | 自由能统计分析；自由能估计、误差和重叠诊断 | 代码BSD-3-Clause；PyMBAR等依赖独立保留许可 | P1 |
| ALG-052 | [RDKit Pharm2D / Pharm3D](https://github.com/rdkit/rdkit) | 药效团表示与筛选；药效团指纹与匹配候选 | RDKit主体BSD-3-Clause；模块依赖独立核验 | P1 |
| ALG-053 | [RDKit shape alignment](https://github.com/rdkit/rdkit) | 三维形状与相似性；对齐姿态与形状/特征相似度 | RDKit主体BSD-3-Clause；具体依赖独立核验 | P1 |
| ALG-054 | [PoseBusters](https://github.com/maabuu/posebusters) | 三维几何质量验证；逐项质量检查结果 | 代码MIT | P0 |
| ALG-055 | [ADMET-AI](https://github.com/swansonk14/admet_ai) | 多端点ADMET预测；吸收、分布、代谢、排泄、毒性预测 | 代码MIT；发布权重与数据独立核验 | P1 |
| ALG-056 | [PyMBAR](https://github.com/choderalab/pymbar) | 自由能统计估计；自由能差、误差估计与重叠诊断 | 代码MIT | P1 |
| ALG-057 | [Morgan fingerprints + Random Forest (scikit-learn)](https://github.com/scikit-learn/scikit-learn) | 传统QSAR基线；性质与活性分类回归 | scikit-learn代码BSD-3-Clause；RDKit指纹BSD-3-Clause | P0 |

## 授权边界

- 代码、模型权重、训练数据分别授权。代码MIT或Apache不能自动授权数据和权重。
- NP-VAE采用定制学术许可，商业使用需作者许可。
- SAFE代码为Apache-2.0，训练数据为CC BY4.0，SAFE-GPT权重为CC BY-NC4.0。
- SynFormer代码为Apache-2.0，Enamine预处理砌块数据有单独研究用途条件。
- MoleculeSTM含NVIDIA定制条款，关联文本还需要检查来源授权。
- 未声明明确许可证的仓库不因可公开浏览而获得再分发许可。
- GROMACS、xTB、GNINA、OpenMM及antiSMASH涉及LGPL、GPL、AGPL或分组件许可。第三方源码快照保持原有许可证，不能统一改为本项目MIT。

## 训练与评价边界

天然产物相似度、QED、可解析SMILES、对接分数、结构生成率和量子化学能量都不是临床成功标签。论文公布的结果需在本项目固定数据、拆分和计算预算下另行验证。NaFM官方说明其test.py是简化演示模板。

训练按骨架、结构或来源组分区；重复ID、相同连接结构和相关分组在拆分前合并。词表只由训练分区拟合，回归标准化和分类类别也只来自训练分区。未知类别不会伪造为已知类别的阴性。有效监督任务默认使用验证集归一化MSE或交叉熵选checkpoint；只有无监督目标时才按语言或加权语言/对比目标选择。测试集在选模后评测，不用于自动挑选checkpoint。

`initial_checkpoint` 复用兼容的编码器、语言、图与融合权重以及原词表，任务头重新初始化，优化器不恢复。历史训练结构或骨架进入新验证/测试集，或历史验证结构进入新测试集时拒绝初始化，避免继续训练泄漏。

模型保存配置、词表、数据和权重哈希、拆分、任务来源和训练状态。预测验证哈希并仅暴露真正获得监督更新的任务。无监督模型拒绝输出成药性或疾病概率；采样输出标记为生成假说，并报告SMILES有效性与是否正常终止。

## Python接口

```python
from npfm.train import train_from_config
from npfm.predict import predict_file, sample_smiles

# 配置内的相对路径以配置文件所在目录为基准；覆盖参数建议使用绝对路径。
run_dir = train_from_config("configs/supervised_template.json", overrides={"epochs": 5})
# 只有该检查点实际训练的任务才能预测。
report = predict_file(run_dir, "examples/compounds.csv", "runs/predictions.csv")
# 仅支持训练过因果语言目标的检查点；生成结构需进一步验证。
candidates = sample_smiles(run_dir, n=10, max_tokens=256)
```

`configs/supervised_template.json`中的任务来源是占位说明，使用者必须换成实际数据来源。没有相应实验或整理标签时，应使用纯结构预训练配置，而不是填补假的端点值。
