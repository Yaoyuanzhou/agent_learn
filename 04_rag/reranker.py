"""
04_rag/reranker.py —— Cross-Encoder 两阶段精排

第一阶段（召回）：Bi-Encoder，用向量相似度快速筛选 Top-20
第二阶段（精排）：Cross-Encoder，把 query+文档拼在一起打分，精度更高但更慢
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-Encoder 精排
    candidates: 第一阶段检索的结果，含 'text' 字段
    返回按精排分数降序排列的 top_k 个结果
    """
    from sentence_transformers import CrossEncoder
    # 多语言 Cross-Encoder，支持中文
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 构建 (query, passage) 对
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # 按精排分数排序
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [
        {**doc, "rerank_score": float(score)}
        for doc, score in ranked[:top_k]
    ]


if __name__ == "__main__":
    # 模拟第一阶段召回的结果
    candidates = [
        {"text": "HNSW 是一种高效的近似最近邻搜索算法，广泛应用于向量数据库。"},
        {"text": "向量检索通过计算余弦相似度找到最相关的文档。"},
        {"text": "RAG 技术结合检索和生成两个阶段，提升 LLM 的知识范围。"},
        {"text": "深度学习模型需要大量 GPU 计算资源进行训练。"},
        {"text": "混合检索融合稠密和稀疏检索结果，通常优于单一方法。"},
    ]

    query = "向量搜索的最近邻算法"
    print(f"【精排】查询：{query}")
    results = rerank(query, candidates, top_k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. [rerank={r['rerank_score']:.3f}] {r['text'][:60]}")
