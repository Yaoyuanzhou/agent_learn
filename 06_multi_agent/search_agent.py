"""
06_multi_agent/search_agent.py —— 搜索专家 Agent

专业化定义的三个层次（见 notes-agent.html）：
  ① 工具集：只有 search / vector_db_query，没有写作工具（硬限制）
  ② System Prompt：只负责检索，不负责写作
  ③ 基座模型：可以换成检索能力更强的模型（可选）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_agent import BaseAgent

# 搜索工具的 Schema 和实现
SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网，获取某个主题的最新信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索词，越精准越好"},
                    "num_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "在本地知识库（向量数据库）中检索相关内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]


def _mock_web_search(query: str, num_results: int = 3) -> str:
    return f"[搜索结果 for '{query}']\n" + "\n".join([
        f"结果{i+1}：关于{query}的第{i+1}条信息，包含关键数据和主要观点..." 
        for i in range(num_results)
    ])


def _mock_query_kb(query: str, top_k: int = 3) -> str:
    try:
        # 尝试用真实的向量库
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '04_rag'))
        from vector_store import search
        results = search(query, top_k=top_k)
        if results:
            return "\n".join(f"[{r['score']:.2f}] {r['text']}" for r in results)
    except Exception:
        pass
    return f"[知识库查询 for '{query}']\n（暂无相关内容，请先用 rag_pipeline.py 建立索引）"


SEARCH_SYSTEM = """你是一个专门负责信息检索的 Agent。

你的唯一目标是找到最相关、最准确的信息。

工作方式：
1. 分析任务，拆解出需要搜索的关键词
2. 执行多次搜索（换不同角度的关键词）
3. 综合搜索结果，去重合并，输出结构化信息摘要

输出格式：
- 关键事实列表
- 信息来源说明
- 需要进一步确认的不确定点

注意：你只输出信息摘要，不负责写作，不负责最终答案生成。"""


def create_search_agent() -> BaseAgent:
    return BaseAgent(
        name="搜索Agent",
        system=SEARCH_SYSTEM,
        tools=SEARCH_TOOLS,
        tool_functions={
            "web_search": _mock_web_search,
            "query_knowledge_base": _mock_query_kb
        }
    )


if __name__ == "__main__":
    agent = create_search_agent()
    result = agent.run("查找 RAG 和 Fine-tuning 各自的优缺点以及适用场景")
    print(f"\n【搜索Agent输出】\n{result}")
