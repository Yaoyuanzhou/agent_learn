"""
03_tools/error_handling.py —— Function Calling 三种错误类型演示

错误类型：
  Type 1: 工具本身出错（文件不存在、API 超时）
  Type 2: LLM 传了错误参数（类型不对、必填项缺失）
  Type 3: 工具返回了意料外的格式

正确做法：把错误信息原样返回给 LLM，让它自己决定重试还是换一种方式

运行：python 03_tools/error_handling.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm import get_client, LLM_MODEL

# 定义一些会出错的工具做演示
TOOLS_WITH_ERRORS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "两数相除",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


def execute_with_error(name: str, args: dict) -> tuple[str, bool]:
    """返回 (结果, 是否出错)"""
    if name == "read_file":
        path = args.get("path", "")
        try:
            with open(path, encoding=args.get("encoding", "utf-8")) as f:
                return f.read()[:500], False
        except FileNotFoundError:
            # Type 1：工具本身出错 → 返回清晰的错误信息给 LLM
            return f"ERROR: 文件 '{path}' 不存在。请检查路径是否正确，或选择其他文件。", True
        except Exception as e:
            return f"ERROR: 读取文件时出错：{e}", True

    elif name == "divide":
        a, b = args.get("a"), args.get("b")
        if b == 0:
            # Type 2：业务逻辑错误 → 告诉 LLM 为什么不行
            return "ERROR: 除数不能为 0，请换一个非零的 b 值。", True
        return str(a / b), False

    return f"ERROR: 未知工具 '{name}'", True


def run_with_error_recovery(user_input: str):
    """演示 LLM 如何根据工具错误信息自动恢复"""
    client = get_client()
    messages = [
        {"role": "system", "content": "你是一个助手。遇到错误时，根据错误信息调整策略并重试。"},
        {"role": "user",   "content": user_input}
    ]

    for step in range(5):
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS_WITH_ERRORS
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            print(f"\n[最终答案] {msg.content}")
            return msg.content

        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result, is_error = execute_with_error(tc.function.name, args)

            status = "❌ 错误" if is_error else "✅ 成功"
            print(f"[Step {step+1}] {tc.function.name}({args}) → {status}: {result[:80]}")

            # 无论成功还是失败，都把结果返回给 LLM（让它决定下一步）
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })


if __name__ == "__main__":
    # 会触发 divide(10, 0) → 出错 → LLM 自动重试
    print("=" * 60)
    print("【测试】除零错误恢复")
    run_with_error_recovery("帮我计算 10 除以 0 的结果")

    print("\n" + "=" * 60)
    print("【测试】文件不存在错误")
    run_with_error_recovery("帮我读取 /tmp/not_exist.txt 文件的内容")
