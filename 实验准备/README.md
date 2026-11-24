# CARE-Distill 实验准备

补齐日期：2026-11-19。原研究包与综述日期：2026-11-18。

**这次完成的是可执行的实验准备，尚未运行论文实验。** 已对照共享对话、Downloads综述、研究包和现有数据管线；没有训练、部署或新增教师付费调用。

先看 [缺项与处理结果](GAP_AUDIT.md)，再按 [你和娜样的研究任务单](TASK_MATRIX.md) 执行。实际状态见 [readiness.json](reports/readiness.json)。

验证记录：35项离线软件检查通过，见[verification_summary.json](reports/verification_summary.json)。它们验证代码和数据约束，不代表模型或临床指标达标。

## 已落地

| 内容 | 文件 | 当前状态 |
|---|---|---|
| 统一研究协议、数据配额、随机种子、比较预算 | config/protocol.v1.yaml | 工作草案，未正式预注册 |
| 可验证的数据结构与语义约束 | config/annotation.schema.json、care_contract.py | 156条现有候选迁移并通过检查 |
| 主标签/副意图/澄清/升级标注指南 | ANNOTATION_GUIDE.md | 待双方试标冻结，临床部分待专业复核 |
| 两份不显示教师和候选答案的标注包 | data/annotation_you.v1.jsonl、annotation_nayang.v1.jsonl | 156条/人，均待标 |
| 双人结果合并与分歧保留 | merge_annotations.py | 空白包验证为0通过、156未决 |
| 23条文献元数据审计与修正版BibTeX | reports/REFERENCE_AUDIT.md | 两处作者信息修正，原件保留 |
| 校准、分类、行动、风险—覆盖率等指标 | care_metrics.py、prepare.py | 软件检查通过；没有正式金标结果 |
| 五seed成对分组bootstrap、置换检验、Holm | care_statistics.py | 有软件验证，不代替实际五次训练 |
| 训练池主动选择与随机对照 | care_acquisition.py | 只允许训练池，需学生概率后才能实跑 |
| 可执行TF-IDF+LogReg基线入口 | baseline_tfidf.py | 未训练，正式金标齐备后由你执行 |
| 缩减后的主实验矩阵 | plans/run_matrix.json | 5个主要条件×5 seeds=25个计划训练任务 |
| 教师接口现状与安全样本量审查 | reports/teacher_interface_characterization.json、safety_sample_size.json | 已离线计算 |

## 当前数据实况

- 156条合成候选，全部unassigned、pending，**人工金标为0**。
- 153个Jev返回的完整四类概率向量通过结构检查；97个为API所返回的精确one-hot，约75.2%的最大概率≥0.99。
- 结构齐全不等于校准正确。当前**可直接用于已校准软蒸馏的目标为0**；旧政策教师结果还需迁移审核。
- 没有真实医院、外院、未来时间切片或临床审核过的安全测试，不能声称这些外测已具备。

## 快速验证

```powershell
Set-Location 'C:\Users\Administrator\hgdoctor-intent-lab\实验准备'
$carePython='C:\Users\Administrator\miniconda3\python.exe'
& $carePython .\prepare.py validate --input data/candidates.canonical.jsonl --output reports/data_validation.json
& $carePython -m unittest -v test_preparation.py
```

`prepare.py build`可从上一轮交接包重新生成研究格式和计划；已有人工标注包不会覆盖。标注前请另存副本。没有上一轮目录时直接使用本包已提供的数据与标注包，不必运行build。

正式数据齐备后的命令见 [RUNBOOK.md](RUNBOOK.md)。命令示例不表示对应模型训练和实验已经执行。

## 来源与阅读边界

- 共享对话：[模型蒸馏演进解析](https://chatgpt.com/share/6afd8654-bbd8-83ea-bfa1-f14f969ea1df)。已保存可见文字；最早和最后的助手回复未显示正文，未推断它们的内容。
- `F:\Downloads\Latest_Research_Review_ZH.md`与zip内同名文件字节完全相同，不是两份不同综述。
- zip六个文件全部保留在sources/package，外部综述另留副本；没有把材料中的操作建议当作额外授权。
- 文献审计核对官方元数据；没有声称逐篇复现论文实验。修正和方法范围说明见 [论文改稿补丁](MANUSCRIPT_DELTAS.md)。
