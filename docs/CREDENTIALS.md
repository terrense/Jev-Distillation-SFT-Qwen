# Jev凭据与手动标注入口

仓库公开。真实密钥以GitHub Actions Secret `JEVSTATION_API_KEY`保存；源码、报告、历史ZIP与文档不包含该值。GitHub不会提供把Secret读回成明文的接口。

## Nayang如何使用

如果你有本仓库的Actions运行权限，进入Actions，选择 **Jev labeling (manual budget)**，点击Run workflow：

1. 选择合成输入包，只允许仓库内预先列出的两个候选文件。
2. 选择1/5/10/25的显式请求上限，建议先1或5。
3. 等任务结束，下载`jev-annotations-RUN_ID` artifact，里面是原始教师响应和输入/政策指纹。
4. 复核并按新批次归档；任务不会自动提交新标签，更不会自动把它们变成人工金标。

工作流只在手动触发时读取Jev Secret；PR和普通CI测试不读取它。默认预算5次，无定时任务或自动跑满余额。API返回401/402/403时停止继续调度，已经在途的结果仍会记录。每次新任务是新批次，因此不要反复点击来重复标注同一组样本。

Artifact不是永久数据仓库。下载后将结果与账本作为新提交/PR保存；不要只依赖临时下载链接。

如果没有Run workflow权限，联系仓库owner授予适当写入权限；公开可读不等于有权使用密钥。不要通过修改workflow把Secret打印出来，也不要把它写进artifact。

## 需要本地API调用时

由owner安全提供有效key，或使用你自己的key，然后只设置进程环境变量。

PowerShell隐藏输入示例：

```powershell
$jevSecure=Read-Host 'Jev API key' -AsSecureString
$env:JEVSTATION_API_KEY=[Net.NetworkCredential]::new('', $jevSecure).Password
try {
  python intent_lab.py predict --provider jev --model jev --transport powershell --api-key-env JEVSTATION_API_KEY --input data/seed_candidates.jsonl --output reports/local_runs/jev_new_batch.jsonl --max-requests 5
} finally {
  Remove-Item Env:JEVSTATION_API_KEY -ErrorAction SilentlyContinue
}
```

Linux/macOS可以使用同名环境变量与urllib传输，但原Windows环境的urllib曾被站点返回403/1010；本仓库验证过PowerShell HTTP客户端。不要无限重试鉴权/余额错误。

旧key由owner计划近期作废。最后记录余额11844是在2026-11-18，key是否仍有效、余额是否改变均未在交接时额外扣费验证。作废后在仓库Settings→Secrets and variables→Actions更新/删除`JEVSTATION_API_KEY`；本地进程也应清掉旧值。

## Owner维护

在具有仓库管理权限的机器，用隐藏输入或stdin设置：

```text
gh secret set JEVSTATION_API_KEY --repo terrense/Jev-Distillation-SFT-Qwen
```

不要把key放进`--body`参数、命令历史、README、issue、截图或Git提交。普通检查不需要此Secret，离线工作可继续进行。
