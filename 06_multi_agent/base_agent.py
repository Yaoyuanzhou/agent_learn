"""
06_multi_agent/base_agent.py —— 所有 Agent 的基类

定义了 Agent 的三要素：
  name:    Agent 名字（调试时显示）
  system:  System Prompt（职责定义）
  tools:   可用工具集（决定 Agent 能做什么）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm import get_client, LLM_MODEL

MAX_STEPS = 8


class BaseAgent:
    def __init__(self, name: str, system: str, tools: list = None,
                 tool_functions: dict = None):
        """
        name:           Agent 名字，用于日志显示
        system:         System Prompt，定义职责和行为边界
        tools:          OpenAI 格式的工具 schema 列表（为空则只能聊天）
        tool_functions: {tool_name: callable} 工具名到实现函数的映射
        """
        self.name = name
        self.system = system
        self.tools = tools or []
        self.tool_functions = tool_functions or {}
        self.client = get_client()

    def run(self, task: str, context: str = "") -> str:
        """
        执行一个任务，返回最终文字结果。
        context: 其他 Agent 传来的背景信息（黑板模式）
        """
        system_prompt = self.system
        if context:
            system_prompt += f"\n\n【来自其他 Agent 的背景信息】\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": task}
        ]

        for step in range(MAX_STEPS):
            kwargs = {"model": LLM_MODEL, "messages": messages}
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            resp = self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # 没有工具调用：返回最终结果
            if not getattr(msg, "tool_calls", None):
                return msg.content

            # 执行工具调用
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                func = self.tool_functions.get(name)

                if func:
                    result = func(**args)
                else:
                    result = f"ERROR: 工具 '{name}' 未在 tool_functions 中注册"

                print(f"  [{self.name}] 调用 {name}({args}) → {str(result)[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        return "[超出最大步数]"
