"""
02_react/react_agent.py —— ReAct Agent 核心实现

ReAct 循环：Thought → Action → Observation → Thought → ...
工具定义在 tools.py，这里只负责循环逻辑。

运行：python 02_react/react_agent.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm import chat_with_tools
from tools import TOOL_SCHEMAS, execute_tool

MAX_STEPS = 10   # 防止无限循环的步数上限


SYSTEM_PROMPT = """你是一个能使用工具的智能助手。

每次你需要输出：
Thought: [分析当前情况，说明下一步打算做什么以及为什么]
然后调用合适的工具，或者如果已经有足够信息则直接回答用户。

规则：
- 每次只调用一个工具
- 等待工具结果（Observation）后再继续思考
- 信息足够时直接给出最终答案，不要多余调用工具
"""


def run_react(task: str, verbose: bool = True) -> str:
    """
    ReAct 主循环
    verbose=True 时打印每步的 Thought/Action/Observation
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task}
    ]

    for step in range(MAX_STEPS):
        if verbose:
            print(f"\n─── Step {step + 1} ───")

        # 调用 LLM，可能返回文字回复或工具调用请求
        response = chat_with_tools(messages, tools=TOOL_SCHEMAS)

        # 把 LLM 回复加入历史（无论是文字还是工具调用，都要加）
        messages.append({"role": "assistant", "content": response.content,
                         "tool_calls": response.tool_calls})

        # ── 情况1：LLM 直接给出文字答案（没有工具调用） ──
        if not response.tool_calls:
            if verbose:
                print(f"[最终答案] {response.content}")
            return response.content

        # ── 情况2：LLM 请求调用工具 ──
        for tool_call in response.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"[Action] {name}({args})")

            # 执行工具
            observation = execute_tool(name, args)

            if verbose:
                print(f"[Observation] {observation[:200]}")  # 截断过长的输出

            # 把工具结果追加到对话历史（Observation 注入）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation
            })

    # 超出最大步数，返回当前上下文中最后一条助手回复
    return "[超出最大步数，任务未完成]"


if __name__ == "__main__":
    tasks = [
        "北京今天天气怎么样？",
        "一个圆的半径是 7.5 厘米，它的面积是多少？（保留两位小数）",
        "帮我搜索一下 RAG 技术的最新进展，然后总结一下核心要点。",
    ]

    for task in tasks:
        print("\n" + "=" * 60)
        print(f"【任务】{task}")
        result = run_react(task, verbose=True)
