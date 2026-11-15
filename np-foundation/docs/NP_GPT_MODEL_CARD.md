# NP-GPT 天然产物结构语言模型卡

更新日期：2026-11-15。模型族：`NP-GPT`。检查点格式：`np-gpt-decoder-v1`。

**NP-GPT 是本项目直接用 PyTorch 实现的因果 Transformer，用 SMILES 学习下一 token 分布。工程代码由本项目实现；Transformer、RoPE、RMSNorm、SwiGLU、GQA 与权重绑定都是已有方法，本文逐项列出来源，不将它们声称为新的数学架构。** 本模型未加载外部预训练模型权重，也不把旧版结构/图联合 pilot 的权重或 logS 结果称作 NP-GPT 的训练结果。

## 1. 用途与当前结果状态

该模型用于研究天然产物结构序列的概率建模、SMILES 生成，以及可另行监督微调的 token 表示。输入是化学结构 token；它没有自然语言指令微调，也没有蛋白、细胞、实验条件或疾病预测头。`encode` 返回最终层的因果 token 表示；它本身不提供成药性或适应症。

本轮训练结构来源为固定版本的 COCONUT 与 LOTUS 联合结构文件。结构文件没有疾病、药效或 ADMET 监督标签。原始语料、实际可纳入序列、训练器选择的序列以及检查点真正读取的序列分别计数；不能因为全量数据完成准备，就声称所有记录已经参与参数更新。

### 真实运行记录

下表为本轮真实训练/生成结果的登记位置。在对应报告产生前保持未填，不引用旧 pilot 指标，也不把测试中的小分子夹具当作生成表现。

| 项目 | 状态/结果 | 依据 |
|---|---|---|
| 原始结构文件、版本与 SHA-256 | `coconut_lotus_union_compounds.csv`；238,371,765 bytes；完整 SHA 见下文 | 本轮 corpus `manifest.json` 的 `source` |
| 原始行数、合格 train/val/test 数 | 764,701；合格分区分别为 688,166 / 37,720 / 38,810 | 本轮 corpus manifest 与 SQLite |
| 分词器规模、上下文和排除数 | 376 tokens；上下文 1,024；排除 5 条：1 条超长、4 条 RDKit 失败 | 本轮 corpus manifest、tokenizer |
| 实际参数数、配置与权重 SHA-256 | 等待本轮检查点 | `model_config.json`、`model.index.json` |
| 优化器更新数、实际看过的分子/token | 尚无可登记的本轮训练报告 | `training_report.json` 与实际 seen row IDs |
| 验证集选择与最终语言测试 | 尚无可登记的本轮训练报告 | 选定检查点与真实测试记录 |
| 真实生成的 EOS、有效率、唯一率、新颖性 | 尚未登记本轮真实生成评测 | 原始生成 JSONL、生成参数与评测 JSON |
| 疾病/ADMET/临床评测 | 未训练、未评测 | 结构语言目标不提供这些标签 |

数据准备清单于 `2026-11-15T06:24:45.631834+00:00` 生成。以上是已经核对的准备结果；训练/生成数值将在各自实际报告生成后更新。训练报告必须同时说明是否完整遍历所选训练集合；中途停止、步数上限或子样本训练不能写为完成了整个语料的一个 epoch。

数据核验定位：`npfm-data/np_gpt_corpus/manifest.json`。

| 文件 | SHA-256 |
|---|---|
| 联合原始结构 CSV | `08765cdfebced158b308978f08dba106b878b6f87ea09c6e93da9c553a4c0685` |
| 本轮 corpus manifest | `a93c49e68e29a891ea73fe28ca78a030ca97914d41b63702f534f03e0ef0fcf5` |
| `metadata.sqlite` | `6b13f289b8ce85cd7cca86342bddebe9ee230877ca6ca827e446d0418e3c64c0` |
| `tokenizer.json` | `f4d63d611bd8a571ea3fd2ee8904e86a15fbb0525f58171bbb6c08f5b629bd7a` |

## 2. 已实现架构与方法来源

实现入口：[model.py](../src/npfm/np_gpt/model.py)。因果目标为

\[
p_\theta(t_1,\ldots,t_n)=\prod_{i=1}^{n}p_\theta(t_i\mid t_{<i}).
\]

训练调用以 `input_ids = sequence[:-1]`、`targets = sequence[1:]` 对齐下一 token；padding 与忽略标签不计入损失。一个结构的 BOS/EOS 属于序列边界。模型不会使用整分子的图特征向语言解码器泄露后续 token。

| 组件 | 本实现 | 原始/官方依据 |
|---|---|---|
| Decoder-only 因果 Transformer | 仅因果自注意力与前馈残差块，不含编码器交叉注意力 | [Transformer 原始论文](https://arxiv.org/abs/1706.03762)；[LLaMA 架构背景](https://arxiv.org/abs/2302.13971) |
| 注意力计算 | 调用 PyTorch scaled dot-product attention；带缓存的多 token 情况使用前缀偏移后的明确因果 mask | [PyTorch SDPA 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| RoPE | 对 query/key 的相邻通道对施加旋转位置编码；位置由配置与有效 token 推导 | [Su 等，RoFormer](https://arxiv.org/abs/2104.09864) |
| RMSNorm | 残差子层之前归一化，平方均值累积使用 float32 | [Zhang 与 Sennrich，RMSNorm](https://arxiv.org/abs/1910.07467) |
| SwiGLU | SiLU 门控支路与线性支路逐元素相乘，再投影回隐藏维度 | [Shazeer，GLU Variants](https://arxiv.org/abs/2002.05202) |
| 可配置 GQA | 多个 query heads 共享较少的 key/value heads；缓存保存紧凑 K/V，计算注意力时扩展 | [Ainslie 等，GQA](https://arxiv.org/abs/2305.13245) |
| 输入/输出权重绑定 | 默认 token embedding 与语言输出矩阵使用同一权重 | [Press 与 Wolf，权重绑定](https://arxiv.org/abs/1608.05859) |
| KV cache | 生成时复用各层前缀 key/value；前缀和新增 token 总长必须在上下文限额内 | 本项目实现与测试；SDPA mask 语义见官方接口 |
| SMILES 分词 | 复用项目的显式化学 token 词法规则，保留方括号原子、环闭合、键和手性记号；词法覆盖不代表化学有效 | [Molecular Transformer 官方分词示例](https://github.com/pschwllr/MolecularTransformer#pre-processing)；[项目 tokenizer](../src/npfm/tokenizer.py) |

参考 LLaMA 等架构文献仅用于说明标准组件组合；NP-GPT 不是这些模型的官方变体，不使用其 checkpoint，也未获得其论文中报告的效果。RoPE 的存在不证明能够超出已训练上下文长度可靠外推。

代码提供 `smoke`、`small`、`base` 容量配置，分别使用 2/4/8 个 decoder 层、64/192/512 的隐藏维度。它们是可实例化的架构配置，不表示三个规模都完成训练。实际参数数取决于词表与最终配置，以本轮 checkpoint 的实算值为准。

## 3. 数据、身份与分词边界

### 3.1 结构语料的实际范围

数据准备由 [corpus.py](../src/npfm/np_gpt/corpus.py) 执行。原始 SMILES 不被规范 SMILES 覆盖：训练 token 对应原始序列；RDKit 生成的 `canonical_isomeric_smiles` 另存，供结构审计和生成参考查询。所有输入行，包括重复、失败、超长或不纳入的记录，都保存在 `metadata.sqlite`；只有满足纳入规则的首次完整 key 记录进入 token 文件。

`full_inchikey` 在该语料中沿用原库提供的完整键。格式检查不是 RDKit 重算一致性核验。来源 key 与原始结构的可能不一致不能被“完整 27 位”这一名称掩盖。独立的 canonical SMILES 索引可以提供另一个精确结构比对渠道；其计算版本、覆盖率与跨分区重叠审计随 manifest 保存。

天然产物数据库收录与生成结构的天然来源真实性是不同问题。来源物种、文献和可能的半合成/类似物标签应通过原始来源记录审查；本语言模型没有学习或验证每个生成物是否在自然界存在。完整数据库联合也不等于全部已知天然化合物。

### 3.2 分区政策

新 corpus 按来源 InChIKey 的前 14 字符连接层键，结合固定 seed 的 SHA-256 做稳定 train/val/test 分配。相同连接层组不因立体形式不同而进入不同分区。实际比例、组数和不交叠检查以 manifest 为准。

这属于身份/连接层分组，不是 Bemis–Murcko 骨架拆分，也不是时间、靶点或物种留出。相似骨架可以同时出现在不同分区。来源 key 未重新核验时，还需要检查 RDKit canonical SMILES 是否跨分区；该审计未完成或发现重叠时必须披露，不能声称完成了严格的化学身份隔离。

本次实算的来源连接层跨分区组数为 0；RDKit canonical SMILES 的跨分区组数在全部可解析记录和合格记录两种口径下均为 0。SQLite 保留全部 764,701 行，其中 764,697 行具有可用 canonical SMILES。4 条解析失败的来源记录保留原 key 与原文，未伪造 canonical 表示。

### 3.3 词表、长度与排除

词表仅从合格的训练分区建立；验证/测试的新 token 映射为 `<unk>` 并单独计数。训练集、验证集和测试集中的 `<unk>` 比率，以及含未知 token 的分子比例，均需随语言指标提供。特别是方括号原子 token 可包含元素、同位素、电荷和手性组合；未知 token 不能当作对这些化学信息的准确表示。

上下文限额包含 BOS/EOS。超长分子保留在审计索引，不能被静默截断后当作完整结构训练。原子数过滤、化学解析过滤和词法过滤分别记录；某来源被纳入数据索引不代表其全部记录进入训练 token 流。代码的容量配置也不改变实际数据准备采用的上下文限额。

本轮数据使用 `max_seq_length=1024`、`max_atoms=null`，未启用原子数过滤。一个 1,136-token 原始序列超过限额并被排除，合格训练序列最大长度为 910。下表是准备的序列流，不是已经消耗的训练量：

| 分区 | 合格序列 | 序列 token 数，含 BOS/EOS | 预测目标 token 数 | OOV token / 词法 token | 含 OOV 的序列 |
|---|---:|---:|---:|---:|---:|
| train | 688,166 | 46,145,451 | 45,457,285 | 0 / 44,769,119 | 0 |
| val | 37,720 | 2,521,588 | 2,483,868 | 9 / 2,446,148 | 9 |
| test | 38,810 | 2,611,716 | 2,572,906 | 8 / 2,534,096 | 6 |

数字来自上述固定 hash 的 manifest；OOV 分母不含 BOS/EOS。超长和解析失败记录仍保留在全语料参考中，但不进入上表的训练流。

## 4. 训练与检查点报告规则

本模型使用结构语言损失学习下一 token，没有实测药理或疾病监督。训练与生成的配置、corpus manifest、tokenizer、权重分片均需要 hash 记录。每次实验单独登记：设备与依赖版本、随机种子、模型配置、训练/验证/测试实际选择数、已见训练 row IDs、预测 token 数、优化器更新数、完整/部分 epoch、模型选择标准与耗时。

当前训练工作流使用预先准备的分区和序列边界。验证集用于选择检查点，最终测试应在选择后计算。不同 tokenization、OOV 比率、过滤规则、长度分布或测试集合的语言 NLL/perplexity不能无条件直接比较。较低语言损失不等于更多药效或更好的临床安全性。

复现需要区分推理用的模型分片与恢复训练用的完整状态。推理 checkpoint 的模型 hash 证明加载了指定参数；继续训练还需要优化器、调度进度、随机数和实际数据游标的一致状态。只有真实完成的训练步骤才计入训练量，代码测试和随机初始化模型容量不计入已训练成果。

## 5. 生成评测定义与实际接口

评测实现：[evaluation.py](../src/npfm/np_gpt/evaluation.py)。接口：

```python
from npfm.np_gpt.evaluation import evaluate_generated

report = evaluate_generated(
    records=generated_records,
    corpus_dir="/absolute/path/to/prepared_corpus",
    output_path="/absolute/path/to/generation_evaluation.json",
)
```

输入通常为 `{smiles, terminated, token_count, seed, temperature, top_k, top_p}` 的记录序列；`terminated_with_eos` 是兼容别名。终止状态必须为布尔值，缺失保持未知。可接收 SMILES 字符串列表，但此时不能推断是否由 EOS 结束。

原始生成记录应另存 JSONL，并记录指定 checkpoint 的 hash、生成提示、所有抽样参数、最大长度、attempts 数、是否使用任何筛选/拒绝采样。若只有“直到获得 N 个有效分子”的结果，而没有全部失败 attempts，就不能按本协议报告生成有效率。

### 5.1 指标分母

令 `A` 为全部生成 attempts，`V` 为其中 RDKit 有效样本，`U` 为按明确身份政策去重的有效结构。

| 指标 | 定义 | 分母及未知处理 |
|---|---|---|
| EOS 结束率 | 生成器明确报告 `terminated=True` 的数量 | 全部 `A`；若有缺失终止标志，整体率为 `null`，另报已报告部分与覆盖率 |
| RDKit 有效率 | 通过解析、sanitization、非空原子与规范化检查 | `V/A`；全部失败 attempts 保留，空串无效 |
| 有效且 EOS 结束率 | 同时满足以上两条件 | 全部 `A`；未知终止状态不被当成 False |
| 唯一率 | 去重有效结构数 | `U/V`；同时给 canonical SMILES 与 full key 各自唯一数 |
| 相对训练参考的新颖性 | 未在合格 train 分区命中的独特有效结构 | `U`；只要某独特结构无法核对，整体率为 `null` 并报告已评估覆盖率 |
| 相对全语料的新颖性 | 未在完整索引命中的独特有效结构 | 同一个 `U`；全语料包括超长、重复和未纳入训练的索引行 |

无有效样本或零 attempts 时，分母为零的率保持 `null`，不填 0 或 1。未正常结束但碰巧构成合法 SMILES 的样本仍计入化学有效率，同时保留 `terminated=False`；这样不会将两项指标混同。

RDKit 默认可把 SMILES 后的文本当作名称。本评测禁用名称与 CXSMILES 扩展并拒绝空白，因此不会把带额外文本的输出偷偷修复成有效结构。[RDKit 官方解析说明](https://rdkit.org/docs/RDKit_Book.html#reading-molecule-names)

### 5.2 去重与精确参考查询

去重优先使用从生成结构计算的标准 InChI 的完整 key；转换失败时回退到 RDKit canonical isomeric SMILES。两种表示的唯一数量单独报告，因为标准 InChI 的规范化可能使不同 canonical SMILES 得到同一个 key。没有人为去盐、去电荷、去同位素或移除手性。[RDKit InChI 接口](https://rdkit.org/docs/source/rdkit.Chem.inchi.html)

读取 `corpus_dir/manifest.json` 中 `metadata.path` 指向的 SQLite，默认名称为 `metadata.sqlite`。使用只读查询，验证 manifest 给出的数据库 hash，记录本次实际数据库与 manifest 的 SHA-256。索引表为 `molecules`。

- `in_training_corpus`：任一可用精确键命中 `split='train' AND included=1`。
- `in_full_corpus`：任一可用精确键命中完整 `molecules` 表，不限制 `included`。
- canonical 比对只查询 `canonical_isomeric_smiles` 列；没有该列或整列为空时，不能把原始 `smiles` 当作规范结构索引。
- full InChIKey 或 canonical SMILES 任一命中即保守判定为已知；每种匹配结果均保留，避免掩盖来源 key 的不一致。
- 无参考语料时，新颖性为未知。缺数据库、缺必要 schema 或 hash 不一致时明确报错，不静默返回“全部新颖”。

这里的训练参考是**准备好的合格训练分区**。它不证明选定检查点已经逐条看过所有记录。若实际训练只消耗该分区的一部分，报告必须同时给出检查点实际 seen row IDs 和数量；本指标本身不是记忆性或实际曝光审计。

精确新颖性只表示在指定快照与身份规则下没有命中。它没有计算邻近结构、骨架新颖性、文献查重或专利查重，不等于世界首次发现。

### 5.3 计算性质

每条有效结构可输出 RDKit 分子量、Crippen logP、TPSA、氢键供体/受体数、可旋转键数、环数、重原子数和 QED。字段统一标为 `computed_not_experimental`；聚合分布使用每个独特有效结构的一个代表，避免大量重复样本放大权重。

QED 是计算描述符，不是药物成功概率。RDKit 官方文档也说明其 QED 取决于具体性质计算实现，不能把它当作临床或实测终点。[RDKit QED 文档](https://rdkit.org/docs/source/rdkit.Chem.QED.html)

通配原子、自由基和不连通片段单独标记。它们可能满足 RDKit 的语法/结构解析条件，但未由本协议排除其稳定性、合成和实验风险。有效、唯一或精确新颖不推出天然来源真实、可合成、可成药或临床有效。

## 6. 实际验证与尚未完成的验证

评测测试使用明确的小型合成测试夹具，验证分母、重复、立体异构体与同位素、名字/空白拒绝、未知终止、新颖性参考、源 key 与 canonical 匹配差异、哈希变化拒绝和 JSON 输出。测试通过证明相应代码行为；测试中的小分子结果不是模型生成性能。

2026-11-15 本模块 12 项测试通过。另从已冻结的真实 SQLite 各取一条 train、val、test 与超长排除结构完成只读集成检查：四条均命中全语料，仅合格 train 记录命中训练参考。这是索引范围与接口检查，不是生成有效率测试。

可重跑本模块测试：

```bash
python -m pytest -q tests/test_np_gpt_evaluation.py
```

架构测试、语料测试、运行时测试与真实训练/生成结果分别报告。尚未完成的分布匹配、近邻相似度、骨架多样性、化学稳定性、合成可行性、靶点活性、ADMET、动物或临床验证不能由语言指标代替。本版本没有与外部大型模型在同一信息与计算预算下完成性能比较，因此不声称达到 SOTA。

## 7. 许可与结果解释

本项目工程代码遵循仓库许可证；PyTorch、RDKit 和语料来源保留各自许可。语料准备与引用论文不改变源数据的再分发条件。模型参数、数据、算法源码和计算结果分别记录来源，不将下载过的外部模型目录视为已经接入或训练。

本模型提供结构生成假设与可复查的工程指标。若未来加入药理或疾病监督，需要独立的实验标签、上下文、留出验证和校准，并更新模型版本与模型卡；不能将旧模型的性质头输出拼接进新生成模型后，称为 NP-GPT 已学会成药性或适应症。
