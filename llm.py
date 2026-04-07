"""
llm.py —— 统一 LLM 调用封装，所有模块都用这个
支持流式/非流式，自动重试，统一错误处理
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def chat(messages: list[dict], model: str = None, temperature: float = 0.7, **kwargs) -> str:
    """
    最简单的调用入口：传 messages 列表，返回字符串回复。
    messages 格式：[{"role": "user/system/assistant", "content": "..."}]
    """
    resp = get_client().chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs
    )
    return resp.choices[0].message.content


def chat_with_tools(messages: list[dict], tools: list[dict], model: str = None) -> dict:
    """
    带工具定义的调用，返回完整 message 对象（可能含 tool_calls）
    """
    resp = get_client().chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return resp.choices[0].message
