"""Assistant-authored contrast sets; one surface answer, three genuine business contexts."""
from collections import Counter
from intent_lab import POLICY, save_jsonl, validate, ROOT


ITEMS = [
    ("三天了", "这次头晕持续多久了？", "您的车在医院停车场停了多久？", "我想问医院停车怎么计费"),
    ("昨天开始的", "这次头痛是什么时候开始的？", "预约页面无法打开是从什么时候开始的？", "我想咨询挂号页面打不开的问题"),
    ("没有", "您说头痛，同时有没有恶心？", "这次来带医保卡了吗？", "我想问忘记带医保卡如何就医"),
    ("有", "这次咳嗽有没有同时发热？", "您的手机上有挂号成功的通知吗？", "我想咨询如何取号"),
    ("左边", "您说耳朵疼，是左边还是右边？", "您要找门诊楼左侧入口还是右侧入口的路线？", "我想问门诊楼入口怎么走"),
    ("还没", "您这次因为这个症状看过医生了吗？", "您已经交停车费了吗？", "我想问医院停车场怎么缴费"),
    ("不知道", "这次头痛具体什么时候开始的，您清楚吗？", "您知道报告已经出结果了吗？", "我想问检查报告如何领取"),
    ("是我妈妈", "这次说头晕的是您自己还是家人？", "要查询的是您自己还是家人的挂号？", "我在帮家人问挂号查询手续"),
    ("五十多岁", "您说的这位头晕患者大约多大年龄？", "您要咨询挂号流程的就诊人多大年龄？", "我想问给家里人挂号的流程"),
    ("不记得了", "这次咳嗽开始的具体日期是什么？", "您还记得预约的是上午还是下午吗？", "我想问如何查询自己的预约时段"),
    ("刚才说错了，是右边", "您刚才说左耳不舒服，要再确认一下是哪边吗？", "您刚说要找门诊楼左边的入口，确认是左边吗？", "我想问门诊楼入口怎么走"),
    ("早上", "您这次头晕通常在一天中的什么时段明显？", "您想查询上午还是下午的挂号流程？", "我想咨询上午来医院如何报到")
]


def main():
    rows = []
    for i, (answer, clinical_question, admin_question, admin_goal) in enumerate(ITEMS):
        clinical_goals = {"triage": "我想根据身体不舒服的情况选个合适的科室",
                          "preconsult": "我的号挂好了，看医生前先帮我把病情整理清楚"}
        for label in ["triage", "preconsult", "info"]:
            stage = {"triage": "unregistered", "preconsult": "preconsulting", "info": "unknown"}[label]
            history = [{"role": "user", "content": admin_goal if label == "info" else clinical_goals[label]},
                       {"role": "assistant", "content": admin_question if label == "info" else clinical_question}]
            rows.append({"id": f"contrast-v02-{i:02}-{label}", "family_id": f"contrast-v02-family-{i:02}",
                         "scenario_group": f"contrast-surface-{i:02}", "user_input": answer,
                         "history": history, "context": {"stage": stage, "active_task": label},
                         "label": label, "tags": ["context", "boundary", "minimal_pair", "needs_human"],
                         "source": "assistant_authored", "review_status": "pending", "policy_version": POLICY["version"]})
    validate(rows)
    save_jsonl(ROOT / "data/context_pairs_v0.2.jsonl", rows)
    print({"candidates": len(rows), "by_label": dict(Counter(r['label'] for r in rows)), "human_approved": 0})


if __name__ == "__main__":
    main()
