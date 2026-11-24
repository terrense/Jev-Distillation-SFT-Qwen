# Jev-Distillation-SFT-Qwen · Nayang 交接仓库

中文医院四类意图路由的数据蒸馏与实验准备。交接日期：2026-11-24。

**Nayang现在接手这个项目。** 先推进数据质量与可复现对比，再按实际收益决定训练方案。当前没有完成8B SFT、临床验证或生产部署。

## 从哪里开始

| 读者 | 入口 |
|---|---|
| Nayang本人 | [从零理解与接手](docs/NAYANG_START_HERE_ZH.md) |
| Nayang的Claude/Codex | [AGENTS.md](AGENTS.md) → [技术交接](docs/AGENT_HANDOFF.md) |
| 做标注/实验的人 | [标注指南](实验准备/ANNOTATION_GUIDE.md)、[任务表](实验准备/TASK_MATRIX.md)、[命令手册](实验准备/RUNBOOK.md) |
| 核对进度的人 | [交接状态](docs/HANDOFF_STATUS_20261124.json)、[研究就绪状态](实验准备/reports/readiness.json) |
| 接着调用Jev | [凭据和手动批次说明](docs/CREDENTIALS.md) |

## 目标

| 标签 | 负责的业务 |
|---|---|
| triage | 按症状或就医目的选择科室 |
| preconsult | 见医生前收集、补充和整理病史 |
| info | 医院位置、流程、挂号、收费、停车等行政服务 |
| other | 范围外，进入固定说明/拒答处理器 |

CLARIFY和ESCALATE是独立动作，不是第五、第六个主标签。信息不足时可以先澄清，医疗风险由独立审核过的规则处理。

## 数据实况

- 156条当前合成候选，研究记录全部pending/unassigned；**人工金标0条**。
- 155次历史Jev标注+1次连通性测试；历史消耗156点，最后记录余额11844，时间2026-11-18。没有把余额当实时状态。
- 153个关联完整四类概率向量，97个API返回值为精确one-hot；尚未独立校准。
- 根管线8项、研究准备35项离线检查曾通过；交接复验见[audit](audit/)。软件通过不等于模型性能通过。
- v0.1的78条teacher-consensus只是弱标签初筛。60条本地模型生成候选被质量隔离。均保留来源与历史。

## 离线启动：不调用API，不训练模型

建议使用Python3.13，与原验证环境一致：

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-prep.txt
python -m unittest -v test_pipeline.py
cd 实验准备
python -m unittest -v test_preparation.py
python prepare.py validate --input data/candidates.canonical.jsonl --output ../reports/current_data_validation.json
```

不需要本地hgdoctor容器、不需要GPU或模型权重。普通测试不使用密钥。

## 材料结构

- 根目录代码/config/data/reports：已跑通的工程管线、候选、历史教师响应与开发报告。
- `实验准备/`：统一研究协议、最新标注包、校准/统计/主动选样工具、原始研究稿与文献审计。
- `business_context/hgdoctor_intent_snapshot/`：原业务意图模块的只读源码快照，供理解现状。
- `archive/originals/`：用户提供的原始研究ZIP与综述。
- `archive/prior_deliveries/`：旧交付包，保留历史，不作为最新执行入口。
- `audit/UPLOAD_MANIFEST.json`：本次上传范围、逐文件指纹和恢复说明。

旧机器的C:/F:路径只是取证位置，不是接手环境的依赖。原根README保留为[README_LAB_LEGACY.md](README_LAB_LEGACY.md)。

## 你们接下来真正要做的

先由Nayang和另一位人类独立完成156条新版标注，补到200条共同试标并冻结边界，再扩到MVP。agent协助实现和找分歧，不能冒充第二位人类或临床审查者。

MVP目标为4k训练/500开发/1k独立测试；工程目标为20k/2k/3k。论文研究计划的6k金标采用另一套划分，不与工程目标重复相加。所有数字都是目标，实际统计以manifest为准。

没有明确授权的真实患者文本不进入外部教师。训练、开发、测试按会话/患者/场景族及外院、时间规则隔离；测试内容不用于合成和调prompt。真实外院/时间数据、临床安全金标和正式模型效果不能由合成记录替代。

## 凭据

Jev密钥使用仓库Actions Secret `JEVSTATION_API_KEY`，不放Git历史。通过手动工作流提供有限预算调用；本地调用时自行配置环境变量。详情见[说明](docs/CREDENTIALS.md)。

根目录没有自动发布或自动消耗Jev额度的脚本入口。训练脚本是后续工具，安装准备依赖不会下载模型或启动训练。
