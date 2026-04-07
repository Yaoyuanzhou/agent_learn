"""
03_tools/parallel_tools.py —— 并行工具调用（asyncio）

当 LLM 请求同时调用多个工具时，串行等待很慢。
用 asyncio 并发执行，所有工具同时跑，最快的先返回。

运行：python 03_tools/parallel_tools.py
"""
import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_react'))

from llm import get_client, LLM_MODEL
from tools import TOOL_SCHEMAS, execute_tool
import time


async def execute_tool_async(tool_call) -> dict:
    """把同步工具包装成异步（模拟 I/O 耗时）"""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    # 模拟网络 I/O 耗时（真实场景里这里是 await aiohttp.get(...)）
    await asyncio.sleep(0.5)
    result = execute_tool(name, args)

    return {
        "tool_call_id": tool_call.id,
        "content": result,
        "name": name
    }


async def run_parallel_tools(user_input: str) -> str:
    """并行执行所有工具调用"""
    client = get_client()
    messages = [
        {"role": "system", "content": "你是一个助手，尽量在一次调用中使用多个工具。"},
        {"role": "user",   "content": user_input}
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS
    )
    msg = resp.choices[0].message
    messages.append({"role": "assistant", "content": msg.content,
                     "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})

    if not msg.tool_calls:
        return msg.content

    # ── 核心：并发执行所有工具 ──────────────────────────────
    print(f"[并行调用 {len(msg.tool_calls)} 个工具]")
    t0 = time.time()

    tasks = [execute_tool_async(tc) for tc in msg.tool_calls]
    results = await asyncio.gather(*tasks)   # 所有工具同时执行

    elapsed = time.time() - t0
    print(f"[完成，耗时 {elapsed:.2f}s（串行约需 {0.5 * len(msg.tool_calls):.1f}s）]")

    # 把所有工具结果追加到消息历史
    for r in results:
        print(f"  - {r['name']}: {r['content'][:60]}")
        messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})

    # 最终 LLM 汇总所有结果
    final = client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    # 这个问题会触发多个工具同时调用
    query = "帮我查一下北京和上海的天气，同时算一下 sqrt(144) + sqrt(256) 等于多少？"
    print(f"【问题】{query}\n")
    result = asyncio.run(run_parallel_tools(query))
    print(f"\n【回答】{result}")
