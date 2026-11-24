"""
挂号意图 - 意图注册

当用户**明确点名某个科室、要直接挂这个科的号**时，识别为 registration 意图。
与 triage（问「该挂什么科」/描述症状让系统推荐）区别开：registration 是用户
已经知道要挂哪个科、直接说科室名挂号，不需要导诊推荐。

extraction_prompt 指示 LLM 抽出科室名，通过 extra_data.department 返回，
dispatch 据此直接路由到 appointment node 挂号。
"""

from ..models import IntentDefinition
from ..registry import IntentRegistry


def register_registration_intent(registry: IntentRegistry) -> None:
    """注册挂号意图到 IntentRegistry。

    Args:
        registry: 意图注册表实例
    """
    registry.register(
        IntentDefinition(
            name="registration",
            description=(
                "用户**明确指定了要挂哪个具体科室**的号，直接挂号，不需要导诊推荐。\n"
                "判定前提：用户输入里出现了明确的科室名（如'消化内科'、'心内科'、'皮肤科'），"
                "且意图是「就挂这个科」。\n"
                "与 triage 的区别：triage 是描述症状或问「该挂什么科」（不知道挂哪个科，需系统推荐）；"
                "registration 是用户已经知道科室、直接点名要挂。\n"
                "**【重要排除】问路/导航/位置不是挂号**：出现科室名但意图是「去/到/怎么走/在哪/"
                "导航/带我去 + 某科室」（如'导航去皮肤科'、'皮肤科怎么走'、'皮肤科在几楼'、'带我去骨科'），"
                "是询问该科室的位置或路线，应判 info_inquiry（location），不是 registration。\n"
                "只有表达「挂号/挂这个科的号/预约这个科」等**挂号动作**时才判 registration。\n"
"**查询自己的挂号也算 registration**：用户想查看/查询自己已挂的号，同属本意图，子动作是「查询」。\n"
                "  查询类的典型说法包括但不限于：「我挂了什么号」「我的挂号」「我的挂号记录」「查询挂号」"
                "「查一下我的挂号」「看看我挂了哪些号」「我有哪些挂号」——只要是想查看已有挂号就算 query，"
                "不要因为句子里没有科室名就判成 idle。\n"
                "输出要求：extra_data.action 返回子动作——要挂号填 \"register\"，查挂号填 \"query\"。"
                "挂号时仍返回 extra_data.department；查询时可省略。\n"
                "与 consultation 的区别：consultation 是已挂号后要做「初步问诊/预问诊」；"
                "registration 只是挂号动作本身，没提问诊。"
            ),
            examples=[
                # —— 挂号（action=register，点名科室）——
                "我要挂消化内科",
                "挂个心内科",
                "帮我挂号皮肤科",
                "挂内分泌科的号",
                "我要挂呼吸内科",
                "给我挂个神经内科",
                "挂号，眼科",
                "我想挂骨科的号",
                # —— 查询挂号（action=query，查看自己已挂的号）——
                "我挂了什么号",
                "我的挂号",
                "我的挂号记录",
                "查询挂号",
                "查一下我的挂号",
                "看看我挂了哪些号",
            ],
            extraction_prompt=(
                "如果识别为 registration，请在 extra_data 中返回子动作 action，并按需返回科室名 department：\n"
                "- 要挂号（用户点名某科室要挂）：action 填 \"register\"，并提取 department（字符串，科室名）；\n"
"- 查挂号（用户要查自己已挂的号，如'我挂了什么号'）：action 填 \"query\"，department 可省略。\n"
                "  【强规则】只要用户表达的是「查看自己已有的挂号」，无论是完整句还是极简名词短语，都判 registration 且 action=query。\n"
                "  以下说法必须判为 registration(query)，不得判成 idle：\n"
                "    '我的挂号'、'查询挂号'、'挂号查询'、'我的挂号记录'、'我的号'、'查一下我的挂号'、\n"
                "    '看看我的挂号'、'我挂了哪些号'、'我有哪些挂号'。\n"
                "  区分：'我要挂号/需要挂号'是想【新挂】号但没说科室（笼统）→ 判 idle；"
                "而上面这些是【查看已有】挂号 → 判 registration(query)。\n"
                '挂号示例: {"action": "register", "department": "消化内科"}\n'
                '查询示例: {"action": "query"}\n'
                "【重要】只有用户明确点出了具体科室名、且意图是直接挂这个科时才判定为 registration（挂号）；\n"
                "用户要查自己已挂的号同属 registration（查询）。\n"
                "如果用户只是描述症状（如'头疼'）或问'该挂什么科'，没有点明具体科室，应判定为 triage；\n"
                "如果用户表达的是挂号后要做初步问诊/预问诊，应判定为 consultation；\n"
                "如果用户是要【去/导航到/问】某科室的位置或路线（如'导航去皮肤科'、'皮肤科怎么走'、"
                "'皮肤科在哪/在几楼'），这是问路不是挂号，应判定为 inquiry，不要判 registration。"
            ),
            priority=3,  # 在 triage/consultation 之后
        )
    )
