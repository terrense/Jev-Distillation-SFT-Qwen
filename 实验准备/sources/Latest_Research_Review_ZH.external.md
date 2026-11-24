# 2025-11—2026-11 蒸馏 × 医疗意图识别研究梳理

> 研究范围：GPT/JeV/开源强 Teacher → Qwen3-8B 或更轻模型；主任务：导诊、预问诊、院内流程信息咨询、其他。以下均对应可以检索的真实论文；“适用性”是针对本项目的判断，不代表原论文验证过四类中文医院任务。

## 一、研究结论（按重要性排序）

1. **别把四分类视为必须使用 8B 的问题。** 先做 MacBERT、ModernBERT（需要实测中文性能）、BGE 向量+线性头等强基线。对单轮四分类，判别式 Encoder 的成本和稳定性可能胜过 8B；真正让 8B 有优势的通常是上下文、复合意图、澄清问题与多任务联动。
2. **JeV/Jev 是 System One / Typed Decision Model，不是传统生成式大模型。** 可以直接做零样本四选一，也可以在合规条件下做标注教师；是否提供全类别归一化概率，需要按实际 API 校验。不要把“最高类别概率”硬说成完整的 Soft Target。
3. **GPT 提供 JSON 标签与自然语言解释，未必提供完整词表 Logits。** 因此适合 Label Distillation、Rationale Distillation、Teacher Consensus、Active Relabeling；闭源教师场景不能未经证实就称作 Token-level KL OPD。
4. **最值得研究的是边界难例**：导诊 vs 预问诊、导诊 vs 挂号流程、症状陈述本身上下文不足、多意图并存、ASR 错误、跨医院术语变化。
5. **仅有 Accuracy 不够。** 主指标 Macro-F1，另评估置信校准、澄清/人工升级、风险漏检、跨院表现、随机种子方差、端到端延迟和标注成本。
6. **论文创新应该体现在“数据—监督—决策”闭环**：多教师异构标签 → 争议样本人工复核 → Student 主动发现高价值难例 → 校准后的 ROUTE/CLARIFY/ESCALATE，而不是仅仅 SFT 四个类别。

## 二、最近 12 个月最值得看的 15 篇

| 优先级 | 论文（发表/预印本年份） | 主要方法 / 已报告贡献 | 对我们的启发 |
|---|---|---|---|
| S | [Scaling Intent Understanding (EACL Industry 2026)](https://aclanthology.org/2026.eacl-industry.14/) | 小模型同时分类和歧义澄清；报告约 8× 推理成本下降、生产路由误差下降 | ROUTE vs CLARIFY 的行动设计；评测澄清的时机与质量 |
| S | [Below the Noise Floor (Findings EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.2063/) | 740 例医疗 API 路由、Qwen 1.5B 学生；多种 KD 发生种子坍缩与输出截断 | 必须 n≥5 seeds、报告失败率、输出合法率；简单 CE 是强基线 |
| S | [How DDAIR you? (EACL 2026)](https://aclanthology.org/2026.eacl-short.20/) | 检测 LLM 合成意图样本与非目标类别的语义重叠，过滤/重生成 | 防止合成“导诊”数据实际属于“预约流程”；真歧义应保留为澄清样本 |
| S | [Preference-Aligned Data Synthesis for Fine-Grained Intent Detection (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.163/) | 原型驱动合成、难负样本、DPO 优化生成器 | 不只是 Prompt 批量造数据；要与此方法对比而非简单冒充创新 |
| S | [Building Multi-turn Intent Classification with LLM-based Labeling (CustomNLP4U 2026)](https://aclanthology.org/2026.customnlp4u-1.8/) | 通过解释+自审查生成标签，模拟多轮意图切换；微调 RoBERTa 与混合路由 | 对比含 Context 与不含 Context；真实医疗会话按 session 划分 |
| S | [Evaluating and Benchmarking the System One Model Jev (arXiv 2026)](https://arxiv.org/abs/2609.37647) | Jev 在分类/路由等 37 数据集上的系统基准 | Jev 本身应是强零样本基线，而非默认“更强的老师” |
| A | [From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev (arXiv 2026)](https://arxiv.org/abs/2610.09188) | Jev + Qwen3-32B 等多教师判断；验证预算内组合教师的价值 | 做强模型 + 类型化决策模型的异构教师消融，评估各自错误相关性 |
| A | [Student-Guided Teacher Distillation for Efficient LLM Task Routing (arXiv 2026)](https://arxiv.org/abs/2610.02516) | Student 先筛候选，Teacher 再判别；强调截断 Top-K 概率不能冒充全分布软标签 | 可以从学生难例与候选类开始，不可虚构 KL 分布 |
| A | [Bridging Generation and Parsing (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.689/) | 合成+一致性+争议仲裁，训练 7B/8B 多意图模型 | Multi-label/Secondary Intent 以及分层标签和争议仲裁 |
| A | [Cost-Efficient Intent Classification via Semantic-Candidate Adaptation and Selective Capability Routing (EMNLP Industry 2026)](https://aclanthology.org/2026.emnlp-industry.257/) | CLINC150/BANKING77/HWU64 上检索辅助、小模型重排序、大模型选择性升级 | 这可能比直接蒸馏 8B 更经济；应纳入部署 Pareto 比较 |
| A | [ActiveLLM (TACL 2026)](https://aclanthology.org/2026.tacl-1.1/) | 大模型引导小样本 Active Learning，缓解 cold start | 标注预算每轮 250/500/1000/2000 条的学习曲线 |
| A | [Reducing Annotation Costs Using KD for Radiology Negation Detection (JMIR 2026)](https://medinform.jmir.org/2026/1/e103912/) | 三医院去标识报告，GPT/Gemma/Qwen 教师 → BERT；人工标签通常仍更优 | 医疗领域“Teacher 伪标签有噪声”；预算应分给专家金标准 |
| A | [PCP: Clinical Pragmatic Intention Recognition (LUHME 2026)](https://aclanthology.org/2026.luhme-1.1/) | 间接话语、临床语用；Hard/Real-Clean 比平均准确率难得多 | 测试要专门构造近邻对抗样本，而不只随机切分 |
| B | [Entropy-Aware On-Policy Distillation (ICML 2026)](https://proceedings.mlr.press/v306/jin26e.html) | Teacher 高熵位置混合 Forward/Reverse KL，改善 OPD 的模式多样性 | 对多轮 Agent 可以研究；对单步四分类并非首要方法 |
| B | [Student-Centered Distillation Narrows the Agentic Gap (ICML 2026)](https://proceedings.mlr.press/v306/lyu26j.html) | Student 自己运行，Teacher 只纠正首个错误，再进行短程优化 | 若把路由与后续工具链联训，可借鉴局部错误纠正 |

另有：[Better as Generators Than Classifiers (2026)](https://arxiv.org/abs/2601.16278)支持“大 Teacher 更多用于造数据、低成本 Student 执行”的方向；[Guided OPD (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1424/)适合未来把四类路由扩展成多轮 Agent Policy 的研究。[ADMEDTAGGER (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-54167-y)是医疗文本“大模型教师→BERT 类学生”的近邻应用。

## 三、研究问题与发表定位

**核心假设 H1**：在固定人工标注预算与推理成本下，异构教师产生的置信可审计监督，加上 Student 驱动的高价值边界样本复核，会比简单强教师伪标签 SFT 在 Hard/OOD 样本上更稳定。

**核心假设 H2**：将 CLARIFY 和 ESCALATE 从四类 Primary Intent 中分离，而不是粗暴增加第五、第六个类，可以降低在模糊、多意图或风险相关句子上的错误自动路由。

**核心假设 H3**：仅靠 8B 模型容量无法保证部署占优；如果 Encoder + LLM Fallback 在风险、延迟和成本方面占优，应报告这一结论，而非预设 8B 一定是最佳架构。

**适合论文的潜在贡献**：医疗意图路由 Benchmark / Annotation Guideline / Data-efficient Heterogeneous Teacher Distillation / Robustness & Calibration / Deployment cost-quality study。单纯“GPT 标注 + Qwen SFT 四分类”比较适合工程应用报告，达不到严谨算法创新研究的强度。

## 四、最短可执行实验流程

**Phase 0（先证明任务真实存在）**：标注指南、四分类 + 副标签 + 处理动作；200 个难例盲审试标；确认哪些边界靠上下文可解，哪些必须 CLARIFY。

**Phase 1（强基线）**：人工 Gold 训练 2,000、验证 600；MacBERT、BGE+LogReg、Qwen3-1.7B/4B/8B 的 Zero-shot + SFT；选定测试 1,200 IID + 800 Hard + 800 新医院 + 400 Temporal + 200 Safety。

**Phase 2（教师质量）**：在仅用于校准的 Gold 子集上测 GPT / Jev / Qwen3-32B 的标签质量、歧义敏感性和错误相关性，分层统计教师一致与分歧；再给训练侧未标注池做伪标签。

**Phase 3（蒸馏对照）**：Gold-only、Teacher Hard-label SFT、教师解释/证据蒸馏、多教师一致性标注、可用时 Class-level Soft KL、难例重生、主动学习、校准/拒判；同等教师调用预算下公平比较。

**Phase 4（上线价值）**：以 ROUTE / CLARIFY / ESCALATE 的端到端正确工作流激活率为核心指标；验证 Error/Latency/Cost；医疗高风险必须有独立规则或临床复核。

**Phase 5（论文化）**：五种随机种子、逐 Seed Raw Scores、95% CI、成对 bootstrap、跨医院时间外测、模型/教师费用核算、负结果分析；填写论文所有占位结果与图表。

## 五、三个必须避免的研究陷阱

- **概念替换**：闭源 GPT 标签蒸馏不能直接称为原生 Logit KD / Token-level OPD。除非真实拿到 Teacher 的对齐 Logprobs 并在 Student Rollout 上优化。
- **伪创新**：2026 年已有歧义澄清、多意图蒸馏、主动学习、偏好驱动合成等研究。新论文必须通过实际实验显示“多教师 + 难例选择 + 校准路由”的独立新价值。
- **伪结果**：未跑出的准确率、Macro-F1、P95 和安全指标不写具体值。论文 Methods 可以完整，Results 必须保留空白，统计流程预注册。

**参考资料与数据治理提示**：上述论文是公开的研究证据；用户医疗系统涉及的既有患者文本不得因“已经脱敏”就自动上传到 GPT、Jev 等外部服务。先确认医院授权、数据处理协议和安全审查。可先采用公开或全合成数据搭建离线原型。
