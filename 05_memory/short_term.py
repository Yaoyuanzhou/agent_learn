"""
05_memory/short_term.py —— 短期记忆：上下文窗口管理

问题：随着对话轮次增多，messages 列表越来越长，最终超过模型的 token 限制。
解法：
  策略1 - 滚动窗口：只保留最近 N 轮，超出的直接丢弃（简单但会丢信息）
  策略2 - 摘要压缩：把早期对话让 LLM 总结成一段摘要，既保留信息又减少 token

运行：python 05_memory/short_term.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm import chat


class SlidingWindowMemory:
    """策略1：滑动窗口，只保留最近 N 轮对话"""

    def __init__(self, max_turns: int = 10, system_prompt: str = "你是一个助手。"):
        self.max_turns = max_turns   # 保留最近几轮（1轮 = user + assistant 各1条）
        self.system_prompt = system_prompt
        self.history: list[dict] = []  # 存储历史对话

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        # 构建发送给 LLM 的消息（系统 Prompt + 最近 N 轮）
        window = self.history[-(self.max_turns * 2):]   # 每轮2条消息
        messages = [{"role": "system", "content": self.system_prompt}] + window

        response = chat(messages)
        self.history.append({"role": "assistant", "content": response})
        return response

    @property
    def token_estimate(self) -> int:
        """粗略估算当前上下文的 token 数（4字符≈1token）"""
        total = sum(len(m["content"]) for m in self.history)
        return total // 4


class SummaryMemory:
    """策略2：摘要压缩，早期对话总结为摘要 + 保留最近 K 轮"""

    def __init__(self, recent_turns: int = 5, summary_trigger: int = 20):
        self.recent_turns = recent_turns      # 始终保留最近几轮原始对话
        self.summary_trigger = summary_trigger  # 超过多少轮时触发摘要压缩
        self.history: list[dict] = []
        self.summary: str = ""   # 历史对话的压缩摘要

    def _compress(self):
        """把较早的对话压缩成摘要"""
        # 取出不在「最近 recent_turns 轮」范围内的历史
        keep_from = -(self.recent_turns * 2)
        to_compress = self.history[:keep_from]
        self.history = self.history[keep_from:]

        if not to_compress:
            return

        # 让 LLM 把这段历史摘要成几句话
        compress_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_compress
        )
        new_summary = chat([
            {"role": "system", "content": "你是一个对话摘要助手。"},
            {"role": "user", "content":
                f"已有摘要：{self.summary}\n\n新增对话：\n{compress_text}\n\n"
                f"请将已有摘要和新增对话合并成一段简洁的摘要（不超过200字）："}
        ], temperature=0.3)
        self.summary = new_summary
        print(f"[摘要压缩] 压缩了 {len(to_compress)} 条消息，摘要更新")

    def chat(self, user_input: str) -> str:
        if len(self.history) >= self.summary_trigger * 2:
            self._compress()

        self.history.append({"role": "user", "content": user_input})

        # 摘要 + 最近对话 + 新消息
        system = "你是一个助手。"
        if self.summary:
            system += f"\n\n【历史对话摘要】\n{self.summary}"

        messages = [{"role": "system", "content": system}] + self.history
        response = chat(messages)
        self.history.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    print("【滑动窗口记忆 - 多轮对话演示】")
    memory = SlidingWindowMemory(max_turns=3)

    turns = [
        "我叫小明，今年 25 岁。",
        "我喜欢机器学习。",
        "推荐几本入门书？",
        "我之前说我叫什么名字？",   # 测试：是否还记得姓名
    ]

    for turn in turns:
        print(f"\nUser: {turn}")
        resp = memory.chat(turn)
        print(f"AI: {resp[:200]}")
        print(f"   (当前上下文约 {memory.token_estimate} tokens，保留 {len(memory.history)} 条消息)")
