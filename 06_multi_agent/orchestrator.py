"""
06_multi_agent/orchestrator.py —— 协调者 Agent（Map-Reduce + 共享黑板）

实现两种多 Agent 协作模式：
  1. Pipeline 模式：搜索Agent → 写作Agent（串行，结果逐步传递）
  2. 黑板模式：所有 Agent 共享一个状态字典，边工作边写入
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from search_agent import create_search_agent
from writing_agent import create_writing_agent
from llm import chat


# ── 模式1：Pipeline（搜索 → 写作）────────────────────────
def pipeline_run(topic: str) -> str:
    """
    串行 Pipeline：
    Step 1: 搜索 Agent 收集信息
    Step 2: 写作 Agent 基于信息写文章
    """
    print(f"\n{'='*60}")
    print(f"【Pipeline 模式】主题：{topic}")

    # Step 1: 搜索
    print("\n─ Step 1: 搜索Agent 收集信息 ─")
    search_agent = create_search_agent()
    search_result = search_agent.run(f"搜索关于「{topic}」的技术原理、优缺点和应用场景")
    print(f"[搜索结果预览] {search_result[:200]}...")

    # Step 2: 写作（把搜索结果作为 context 传入）
    print("\n─ Step 2: 写作Agent 生成文章 ─")
    writing_agent = create_writing_agent()
    article = writing_agent.run(
        task=f"基于提供的信息，写一篇关于「{topic}」的技术总结文章（500字左右）",
        context=search_result
    )
    return article


# ── 模式2：共享黑板（Shared Blackboard）──────────────────
class Blackboard:
    """所有 Agent 共享的状态字典"""
    def __init__(self):
        self._data: dict = {}

    def write(self, agent_name: str, key: str, value):
        """Agent 写入自己的发现"""
        if agent_name not in self._data:
            self._data[agent_name] = {}
        self._data[agent_name][key] = value

    def read(self, agent_name: str = None) -> dict:
        """读取黑板内容（可指定某个 Agent 的）"""
        if agent_name:
            return self._data.get(agent_name, {})
        return self._data

    def to_context_string(self) -> str:
        """转成文字摘要，方便注入 Prompt"""
        lines = []
        for agent, data in self._data.items():
            lines.append(f"[{agent} 的发现]")
            for k, v in data.items():
                lines.append(f"  {k}: {str(v)[:200]}")
        return "\n".join(lines)


def blackboard_run(topic: str) -> str:
    """
    黑板模式：多 Agent 并发写入，协调者汇总
    （这里串行模拟，真实场景可用 threading/asyncio 并行）
    """
    print(f"\n{'='*60}")
    print(f"【黑板模式】主题：{topic}")

    board = Blackboard()

    # Agent-A: 搜索核心概念
    print("\n─ Agent-A: 搜索核心概念 ─")
    agent_a = create_search_agent()
    result_a = agent_a.run(f"搜索「{topic}」的核心技术原理和架构")
    board.write("搜索Agent-A", "核心概念", result_a[:300])

    # Agent-B: 搜索实际应用
    print("\n─ Agent-B: 搜索实际应用 ─")
    agent_b = create_search_agent()
    result_b = agent_b.run(f"搜索「{topic}」的工程实践案例和最佳实践")
    board.write("搜索Agent-B", "实际应用", result_b[:300])

    # 协调者：读取黑板，汇总生成最终报告
    print("\n─ 协调者: 汇总黑板内容 ─")
    context = board.to_context_string()
    print(f"[黑板内容]\n{context[:400]}...")

    writing_agent = create_writing_agent()
    final = writing_agent.run(
        task=f"基于多个 Agent 的调研结果，写一篇关于「{topic}」的综合技术报告",
        context=context
    )
    return final


if __name__ == "__main__":
    topic = "向量数据库在 AI 应用中的作用"

    # 演示 Pipeline 模式
    result = pipeline_run(topic)
    print(f"\n【Pipeline 最终结果】\n{result[:500]}...")
