# 本次上传范围

2026-11-24，用户明确指定公开仓库terrense/Jev-Distillation-SFT-Qwen作为完整交接目的地。

包含：原distillation lab全部有效代码、候选/教师标注/审核包、研究准备、报告、原始研究稿与参考材料；用户提供的原始研究ZIP和综述；两次历史交付ZIP；原hgdoctor意图模块的13个只读源码文件；本次Nayang本人/agent交接说明、依赖和工作流。

不属于研究材料的Python缓存、虚拟环境、Git内部文件不上传。原业务系统的数据库、患者日志、生产.env和权重不在本次材料包中。现有候选为合成记录。

原Word工作副本已被本机保护软件变成%TSD格式，且本机可能重新保护ZIP内的Word。原用户ZIP位于Downloads，其中58343字节的Word内容有效。上传时将原始ZIP与Word成员直接验证并暂存为Git对象，未修改主机安全设置，也没有尝试解密受保护副本。受保护副本另保留来源记录。

上传前验证了原始ZIP SHA256与Word SHA256。`audit/restored_binary.json`记录恢复依据。原历史manifest不改写；本次清单见`audit/UPLOAD_MANIFEST.json`。

密钥存储为仓库Actions Secret `JEVSTATION_API_KEY`；没有以明文文件上传。本次没有调用Jev、消耗新增credits、启动训练或部署。手动标注工作流仅在有权限的维护者显式运行并选择预算时调用API。
