# 可执行命令手册

下面的正式实验命令用于数据/模型就绪之后；本轮只运行了准备和软件检查。所有路径相对本“实验准备”目录。不要把示例中的占位路径当作已有文件。

## 本轮已可运行

```powershell
Set-Location 'C:\Users\Administrator\hgdoctor-intent-lab\实验准备'
$carePython='C:\Users\Administrator\miniconda3\python.exe'
& $carePython .\prepare.py validate --input data/candidates.canonical.jsonl --output reports/data_validation.json
& $carePython -m unittest -v test_preparation.py
```

标完新版包再合并：

```powershell
& $carePython .\merge_annotations.py --left data/annotation_you.v1.jsonl --right data/annotation_nayang.v1.jsonl --output data/reviewed_v1
```

空白包会得到0条通过。分歧/安全升级继续未决；不要直接把pending批量改成human_approved。若输入需要修改，开新版本、重标；不能沿用旧hash的预测。

## 分割与冻结

先填写真实patient/session/family/site/time分组并按协议分区，再运行validate。外院必须真实hospital_real且有site_id，时间外测须有真实事件时间。合成数据不能伪装外院或未来样本。对近重复报告人工复核后填semantic_cluster_id；当前字符TF-IDF只是上下文近邻筛查，不是完整句向量语义审计。

冻结时保存：数据与分割hash、标注指南版本、教师prompt/版本、模型revision与adapter hash、推理模板、随机种子、训练配置、费用账本。源码和数据冻结后再开正式测试。当前来源文件hash已保存在sources/manifest.json。

## 学生预测文件契约

每条预测至少包含record_id、input_hash、prediction、完整probabilities、format_valid和model_run_id。model_run_id绑定真实checkpoint/adapter版本，不能只填容易漂移的服务别名。

```json
{"record_id":"实际ID","input_hash":"对应inputs的SHA256","prediction":"triage","probabilities":{"triage":0.7,"preconsult":0.1,"info":0.1,"other":0.1},"format_valid":true,"model_run_id":"实际权重和adapter指纹","action":null}
```

这些概率只能来自真实可说明的四类评分接口；不要让生成模型随口写0.99。没有完整概率时可以跑硬标签分类指标，不能拟合温度、做风险—覆盖率或伪造class-KL。

## 未来CPU基线入口

```powershell
& $carePython .\baseline_tfidf.py --train YOUR_APPROVED_TRAIN_GOLD.jsonl --evaluate YOUR_DEV_CALIBRATION.jsonl --output reports/tfidf_dev_calibration.jsonl
```

这是实际训练命令，本轮没有执行。重复seed在固定完整数据上可能得到确定性结果，不能把它当作五次独立随机训练。

先使用相同train/settings给dev_calibration、dev_policy和每个测试集各生成预测。预测内容/分组不能与训练泄漏。8B和中文encoder也应输出同一契约，便于共用评测。

## 校准 → 选阈值 → 冻结 → 评分

```powershell
& $carePython .\prepare.py calibrate --gold YOUR_DEV_CALIBRATION.jsonl --predictions reports/MODEL_dev_calibration.jsonl --output reports/MODEL_temperature.json
& $carePython .\prepare.py fit-policy --gold YOUR_DEV_POLICY.jsonl --predictions reports/MODEL_dev_policy.jsonl --calibration reports/MODEL_temperature.json --output reports/MODEL_policy.json
& $carePython .\prepare.py apply-policy --input YOUR_FROZEN_TEST.jsonl --predictions reports/MODEL_test_raw.jsonl --calibration reports/MODEL_temperature.json --policy reports/MODEL_policy.json --output reports/MODEL_test_policy.jsonl
& $carePython .\prepare.py evaluate --gold YOUR_FROZEN_TEST.jsonl --predictions reports/MODEL_test_policy.jsonl --output reports/MODEL_test_metrics.json
```

每个seed/checkpoint分别校准，不能跨权重套温度。阈值搜索中的成本权重是研究候选，尚未取得临床部署批准。独立安全标记来自实际审核的规则/模块，不得用测试金标risk_trigger作为推理输入；未配置时报告缺失，不能宣称已有临床安全层。

## 主动标注与随机对照

```powershell
& $carePython .\prepare.py select --pool YOUR_TRAIN_UNLABELED.jsonl --predictions YOUR_STUDENT_POOL_PREDICTIONS.jsonl --method hybrid --budget 250 --output plans/active250.jsonl
& $carePython .\prepare.py select --pool YOUR_TRAIN_UNLABELED.jsonl --method random --budget 250 --output plans/random250.jsonl
```

两者只能读同一训练池，均按真实判断票数和人工预算算成本。方法使用上下文字符多样性、学生margin和教师分歧；没有冒充论文中更复杂的句向量/风险模型，也没有执行token级OPD。

## 五seed统计与Holm

```powershell
& $carePython .\prepare.py compare --gold YOUR_FROZEN_TEST.jsonl --left A_seed11.jsonl A_seed23.jsonl A_seed42.jsonl A_seed67.jsonl A_seed101.jsonl --right B_seed11.jsonl B_seed23.jsonl B_seed42.jsonl B_seed67.jsonl B_seed101.jsonl --seeds 11 23 42 67 101 --replicates 10000 --output reports/A_vs_B.json
& $carePython .\prepare.py holm --pvalues YOUR_PREREGISTERED_CONTRAST_PVALUES.json --output reports/holm.json
```

左右同位置必须是同seed和同测试输入；不能挑表现最佳的checkpoint/seed后再改对比。Holm输入对应协议中主要比较族，不能在看到p值后把不喜欢的对比删掉。

## 结果登记

每个run填写plans/results_registry.template.json的副本：数据版本、训练预算、有效样本数、model_run_id、逐seed指标、失败状态、解析/截断率、费用、预测文件hash和耗时环境。没有跑过的值保持null，不能填候选标签一致率代替金标指标。
