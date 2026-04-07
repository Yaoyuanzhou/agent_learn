"""
03_tools/function_calling.py —— Function Calling 完整四轮流程演示

四轮对话结构：
  第1轮：用户问题 → LLM 返回 tool_calls 请求
  第2轮：工具执行结果 → LLM 再次生成（可能继续调工具或给最终答案）
  ...直到 LLM 不再调工具

运行：python 03_tools/function_calling.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm import get_client, LLM_MODEL
from tools import TOOL_SCHEMAS, execute_tool

# 直接引用上面 02_react 定义的工具（避免重复）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_react'))


def run_function_calling(user_input: str, verbose: bool = True) -> str:
    """
    演示完整的 Function Calling 四轮对话流程，
    打印每一轮的请求和响应帮助理解底层机制
    """
    client = get_client()
    messages = [
        {"role": "system", "content": "你是一个有工具可用的助手。"},
        {"role": "user",   "content": user_input}
    ]

    round_num = 0
    while True:
        round_num += 1
        if verbose:
            print(f"\n─── Round {round_num}：发送 {len(messages)} 条消息给 LLM ───")

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )
        msg = resp.choices[0].message

        if verbose:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  → LLM 请求调用工具：{tc.function.name}({tc.function.arguments})")
            else:
                print(f"  → LLM 给出最终回复：{msg.content[:100]}")

        # 把 LLM 回复存入历史
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ]
        })

        # 没有工具调用说明任务结束
        if not msg.tool_calls:
            return msg.content

        # 执行所有工具调用，结果逐个追加
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = execute_tool(name, args)

            if verbose:
                print(f"  → 工具 {name} 返回：{result[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })


if __name__ == "__main__":
    # 这个任务会触发：calculator（面积）+ calculator（周长）两次工具调用
    query = "一个半径 5 的圆，面积和周长分别是多少？（pi = 3.14159）"
    print(f"【问题】{query}")
    final = run_function_calling(query)
    print(f"\n【最终回答】{final}")
