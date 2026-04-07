"""
06_multi_agent/writing_agent.py —— 写作专家 Agent

与搜索 Agent 的区别：
  ① 工具集：只有 outline_generator、format_markdown，没有搜索工具
  ② System Prompt：只负责把给定信息写成高质量文章，不负责信息收集
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_agent import BaseAgent

WRITING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_outline",
            "description": "根据主题和关键点生成文章结构大纲",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["topic", "key_points"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_completeness",
            "description": "检查文章是否涵盖了所有要求的知识点",
            "parameters": {
                "type": "object",
                "properties": {
                    "article": {"type": "string"},
                    "required_points": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["article", "required_points"]
            }
        }
    }
]


def _generate_outline(topic: str, key_points: list) -> str:
    from llm import chat
    points_text = "\n".join(f"- {p}" for p in key_points)
    return chat([{"role": "user", "content":
        f"为以下主题生成一个清晰的文章大纲（3-5个章节）：\n主题：{topic}\n关键点：\n{points_text}"}
    ], temperature=0.4)


def _check_completeness(article: str, required_points: list) -> str:
    missing = [p for p in required_points if p[:10] not in article]
    if not missing:
        return "✅ 文章已涵盖所有要求的知识点"
    return f"⚠️ 以下知识点可能需要补充：{', '.join(missing)}"


WRITING_SYSTEM = """你是一个专门负责内容创作的 Agent。

你的目标是基于提供的信息，写出结构清晰、逻辑严密、易于理解的技术文章。

工作方式：
1. 先调用 generate_outline 生成文章结构
2. 按结构逐节撰写内容，保持前后一致
3. 完成后调用 check_completeness 验证覆盖度
4. 如有遗漏，补充相关内容后输出最终文章

写作风格：
- 技术准确，不过度简化
- 用具体例子说明抽象概念
- 适当使用小标题和列表提高可读性

注意：你只使用 【来自其他 Agent 的背景信息】 中的内容，不自行搜索或编造信息。"""


def create_writing_agent() -> BaseAgent:
    return BaseAgent(
        name="写作Agent",
        system=WRITING_SYSTEM,
        tools=WRITING_TOOLS,
        tool_functions={
            "generate_outline": _generate_outline,
            "check_completeness": _check_completeness
        }
    )


if __name__ == "__main__":
    agent = create_writing_agent()
    context = """
关键事实（来自搜索Agent）：
- RAG 通过检索外部知识库弥补模型训练截止日期的限制
- Fine-tuning 把新知识烧录进模型权重，推理时不需要检索
- RAG 优点：实时更新、可解释、成本低；缺点：依赖检索质量
- Fine-tuning 优点：推理速度快、无检索延迟；缺点：更新成本高、可能灾难性遗忘
- 建议：知识更新频繁用 RAG；需要特定风格/格式用 Fine-tuning；可以组合使用
"""
    result = agent.run(
        task="写一篇 500 字左右的技术文章：RAG vs Fine-tuning，什么时候选哪个？",
        context=context
    )
    print(f"\n【写作Agent输出】\n{result}")
