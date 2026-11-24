# hgdoctor-intent-lab

按最新要求，本轮只执行蒸馏数据与任务规划，训练和部署留给你。先读 [你和娜样的数据任务单](TASKS_你和娜样.md)，领取 `data/handoff_v0.2/reviewer_you.jsonl` 和 `reviewer_nayang.jsonl` 开始独立试标，再看 [实测结果](reports/RESULTS.md)。[完整流程](WORKFLOW_你和娜样.md) 保留后续SFT与评测标准供参考。

本目录独立于运行中的hgdoctor，不会自动改业务服务、导出患者日志、下载8B模型或启动训练。当前默认政策是v0.2草案，历史v0.1试验已冻结。所有种子都是待审核候选，不是人工金标。

## 文件

| 文件 | 用途 |
|---|---|
| config/policy.json | 四类定义、上下文与多意图规则 |
| make_seed.py | 生成120条待审候选和独立澄清案例 |
| generate_candidates.py | 经配置的生成式教师批量写候选；默认每次20条 |
| intent_lab.py | 预测、共识队列、分组切分、导出SFT、泄漏检查、评测 |
| train_qlora.py | 文本Qwen3响应标签SFT，包含token mask验证模式 |
| test_pipeline.py | 离线检查依赖失败、数据泄漏和审核状态约束 |
| reports/ | API与对比实测，不含密钥 |

## 本机启动

在PowerShell中：

```powershell
Set-Location C:\Users\Administrator\hgdoctor-intent-lab
$intentPython='C:\Users\Administrator\miniconda3\python.exe'
& $intentPython -m unittest -v test_pipeline.py
& $intentPython .\intent_lab.py --help
```

重新生成v0.2候选会覆盖两个候选JSONL文件，不覆盖历史试验文件：

```powershell
& $intentPython .\make_seed.py
```

## Jev标注

API固定为官方文档中的 `POST https://jevstation.com/api/v1/systemone`。密钥通过环境变量读取。Windows下已验证PowerShell HTTP客户端；Python urllib在此次机器/站点组合被返回403/1010，脚本不会无限重试。

如需在本机新会话配置密钥，使用隐藏输入，不要直接写进脚本：

```powershell
$intentSecureKey=Read-Host 'Jev API key' -AsSecureString
$env:JEVSTATION_API_KEY=[Net.NetworkCredential]::new('', $intentSecureKey).Password
try {
  & $intentPython .\intent_lab.py predict --provider jev --model jev `
    --api-key-env JEVSTATION_API_KEY --transport powershell `
    --input data/seed_candidates.jsonl --output reports/jev_v0.2.jsonl --max-requests 10
} finally {
  Remove-Item Env:JEVSTATION_API_KEY -ErrorAction SilentlyContinue
}
```

默认预算10次；放量必须显式设置 `--max-requests`。输出缓存包含模型、政策、输入指纹；修改模型或政策后选择新输出文件。402/401/403立即停止，其他远程错误连续3次停止，失败不当作other。没有自动重试；如遇429/529，按服务商建议退避后重新执行，已成功的样本从缓存跳过。超时可能已计费，应先查账，不能把本地失败一概当作未扣点。

历史Jev试验使用的是v0.1政策与原始候选。复算时显式选择冻结版本：

```powershell
& $intentPython .\intent_lab.py --policy reports/pilot_policy_v0.1.json score `
  --gold data/pilot_v0.1_smoke.jsonl --predictions reports/jev_seed.jsonl `
  --output reports/jev_smoke_metrics.json --allow-provisional
```

`--allow-provisional`只能用于开发自测。正式测试要求所有gold样本是 `human_approved`。

## 其他教师与候选生成

`compat` 支持标准chat-completions协议，需要自行配置可用的模型、endpoint与密钥。它不是免费ChatGPT会话接口。不同模型的参数支持可能不同，使用前先按对应官方文档检查。以下占位值不能原样执行：

```powershell
& $intentPython .\generate_candidates.py --base-url YOUR_HTTPS_BASE_URL --model YOUR_TEACHER_MODEL `
  --shard nayang-context-001 --coverage '预问诊补充回答、否定、问路换题、ASR口语' `
  --batches 1 --per-batch 20 --output data/nayang-context-001.jsonl

& $intentPython .\intent_lab.py predict --provider compat --base-url YOUR_HTTPS_BASE_URL `
  --model YOUR_INDEPENDENT_TEACHER_MODEL --api-key-env TEACHER_API_KEY `
  --input data/seed_candidates.jsonl --output reports/teacher_b_v0.2.jsonl --max-requests 120
```

不要把教师key放进命令参数；只传环境变量名称。生成器写入的label是候选标签；审核状态强制pending。返回不符合schema时停止，不把错误输出凑成数据。标准脚本支持普通JSON模式；特殊的Responses或推理API需单独适配，不能假设所有平台完全兼容。

两位独立教师出结果后：

```powershell
& $intentPython .\intent_lab.py consensus --input data/seed_candidates.jsonl `
  --teacher-a reports/jev_v0.2.jsonl --teacher-b reports/teacher_b_v0.2.jsonl --output data/review_v0.2
```

accepted.jsonl只表示普通弱标签通过初筛；仍须按方案分层抽查。边界/急症/多意图/注入/needs_human标签强制进人工队列。审阅者需补齐这些tags，不能依赖生成模型自动标对覆盖标签。两个相同模型不能充当独立教师。

## 切分、审核和SFT导出

真正的人工审核由你、娜样及独立复核者完成，记录reviewer与政策版本。人工通过写 `review_status=human_approved`；教师一致且完成抽查的弱标签保持 `teacher_consensus`。不要批量把pending改名为已审核。

```powershell
& $intentPython .\intent_lab.py split --input data/review_v0.2/accepted.jsonl --output data/splits
& $intentPython .\intent_lab.py audit --files data/splits/train.jsonl data/splits/dev.jsonl `
  data/splits/audit_holdout.jsonl --output reports/leak_audit.json
& $intentPython .\intent_lab.py export-sft --input data/splits/train.jsonl --output data/train_sft.jsonl
& $intentPython .\intent_lab.py export-sft --input data/splits/dev.jsonl --output data/dev_sft.jsonl
```

split只是确定性分组切分，默认约80/10/10；受组大小影响，配额需要继续收集。audit_holdout不等于独立人工盲测。正式3,000条人工盲测单独采集与保管，不从生成候选池随机切出来。先做语义家族聚类并填写scenario_group，再运行精确分组检查。已有split字段会保留，同族冲突直接停止。

## 训练与推理

先准备独立CUDA训练环境。当前默认Windows Python是CPU版torch，不要直接执行GPU训练。

模板验证可以在CPU上运行，不加载模型权重：

```powershell
& $intentPython .\train_qlora.py --model YOUR_LOCAL_QWEN3_8B_PATH `
  --train data/train_sft.jsonl --dev data/dev_sft.jsonl --validate-only --local-files-only
```

在已验证的Linux/WSL CUDA环境里，先20step，再正式训练：

```bash
python train_qlora.py --model /models/Qwen3-8B --train data/train_sft.jsonl \
  --dev data/dev_sft.jsonl --max-steps 20 --output checkpoints/smoke --local-files-only

python train_qlora.py --model /models/Qwen3-8B --train data/train_sft.jsonl \
  --dev data/dev_sft.jsonl --epochs 2 --output checkpoints/qwen3-8b-intent --local-files-only
```

先查看显存、loss和run_manifest；脚本会拒绝VL底座、空数据、无标签、token边界不一致和超长样本。`reports/template_check_fixture.jsonl`只用于模板验证，里面train/dev相同是测试夹具，不能拿它做训练或评测。

部署adapter到独立候选服务后，用同一个predict/score链评测；先不要替换当前诊疗服务。Qwen推理端用 `--qwen` 发送硬关闭thinking的模板参数。只接受四个精确标签，格式不合法算失败，不静默解析成成功。

```powershell
& $intentPython .\intent_lab.py predict --provider compat --base-url http://127.0.0.1:8010/v1 `
  --model qwen3-8b-intent --qwen --api-key-env STUDENT_API_KEY `
  --input data/independent_blind.jsonl --output reports/student_blind.jsonl --max-requests 3000
& $intentPython .\intent_lab.py score --gold data/independent_blind.jsonl `
  --predictions reports/student_blind.jsonl --output reports/student_blind_metrics.json
```

未提供adapter部署自动化、语义聚类、正式人工标注界面、校准分数服务或线上灰度改造；这些在完整方案中有明确负责人和验收条件，不能因脚本运行成功就视为已完成。
