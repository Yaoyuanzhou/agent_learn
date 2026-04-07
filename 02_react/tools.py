"""
02_react/tools.py —— ReAct Agent 可调用的工具集（含 mock 实现）

真实项目里这里替换成真实的 API 调用；
mock 模式不需要联网，方便学习和调试。
"""
import json, math, random


# ── 工具注册表 ─────────────────────────────────────────────
# 格式：{name: (function, openai_schema)}

def _search(query: str) -> str:
    """模拟网络搜索，返回 3 条假结果"""
    results = [
        f"[结果1] 关于「{query}」：根据最新数据，该主题的主要观点是...",
        f"[结果2] 另一来源显示：{query} 的核心原理包括三个方面...",
        f"[结果3] 学术研究表明：{query} 在实践中的应用案例如下...",
    ]
    return "\n".join(results)


def _calculator(expression: str) -> str:
    """安全地计算数学表达式，支持加减乘除、括号、sqrt、pi"""
    # 允许数字、运算符、括号、空格，以及函数名 sqrt/pi/pow/abs
    import re
    clean = re.sub(r'\s+', '', expression)
    # 白名单：数字、运算符、括号、小数点、函数名
    if not re.fullmatch(r'[0-9+\-*/().,sqrtpiowabe]+', clean):
        return f"错误：表达式含不允许的字符，支持：数字 + - * / () sqrt pi pow abs"
    try:
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "sqrt": math.sqrt, "pi": math.pi,
            "pow": math.pow,   "abs": abs,
            "sin": math.sin,   "cos": math.cos
        }
        result = eval(expression, safe_globals, safe_locals)
        return str(round(result, 8))
    except Exception as e:
        return f"计算错误：{e}"


def _get_weather(city: str) -> str:
    """模拟天气查询"""
    weathers = ["晴，25°C", "多云，18°C", "小雨，12°C", "阴，22°C"]
    return f"{city}今日天气：{random.choice(weathers)}"


# ── 工具函数映射 ───────────────────────────────────────────
TOOL_FUNCTIONS = {
    "search":      _search,
    "calculator":  _calculator,
    "get_weather": _get_weather,
}

# ── OpenAI Function Calling 格式的工具 Schema ──────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网获取信息。当需要查找实时信息、背景知识时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，要简洁精准"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。支持加减乘除、括号、sqrt、pi。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "如 '2 * 3.14 * 5' 或 'sqrt(16)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气。当用户询问天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 '北京'"}
                },
                "required": ["city"]
            }
        }
    },
]


def execute_tool(name: str, args: dict) -> str:
    """执行工具，统一入口"""
    if name not in TOOL_FUNCTIONS:
        return f"错误：工具 '{name}' 不存在，可用工具：{list(TOOL_FUNCTIONS.keys())}"
    try:
        return TOOL_FUNCTIONS[name](**args)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"工具执行失败：{e}"
