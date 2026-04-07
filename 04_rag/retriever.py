"""
04_rag/retriever.py —— Dense / BM25 / Hybrid+RRF 三种检索策略对比

Dense：向量相似度检索，能找到语义相近但措辞不同的内容
BM25：关键词频率检索，对精确词汇匹配更敏感
Hybrid：两者结果用 RRF（倒数排序融合）合并，通常效果最好
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vector_store import search as dense_search, add_documents
from rank_bm25 import BM25Okapi
# jieba 中文分词（可选，未安装时退回按字符分词）
try:
    import jieba
    _USE_JIEBA = True
except ImportError:
    _USE_JIEBA = False


class BM25Retriever:
    """BM25 稀疏检索封装"""
    def __init__(self):
        self.corpus: list[str] = []
        self.bm25 = None

    def add(self, texts: list[str]):
        self.corpus.extend(texts)
        # 中文按字符分词（简单方式），也可以用 jieba 分词
        tokenized = [list(t) for t in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25:
            return []
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 取 top_k 个最高分
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self.corpus[i], "score": float(scores[i]), "index": i}
            for i in top_indices if scores[i] > 0
        ]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    RRF（Reciprocal Rank Fusion）：合并多个排序列表
    公式：score(d) = Σ 1 / (k + rank(d))
    k=60 是经验值，减少高排名结果对最终分数的主导作用
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            key = item["text"][:50]   # 用文本前50字作为 key 去重
            if key not in scores:
                scores[key] = 0
                texts[key] = item["text"]
            scores[key] += 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": texts[k], "rrf_score": v} for k, v in merged]


class HybridRetriever:
    """混合检索：Dense + BM25 + RRF 合并"""
    def __init__(self):
        self.bm25 = BM25Retriever()

    def add(self, texts: list[str], metadatas: list[dict] = None):
        """同时添加到向量库和 BM25 索引"""
        add_documents(texts, metadatas)
        self.bm25.add(texts)

    def search(self, query: str, top_k: int = 5, dense_k: int = 10, bm25_k: int = 10) -> list[dict]:
        """
        1. Dense 检索 top dense_k 个
        2. BM25 检索 top bm25_k 个
        3. RRF 合并，取前 top_k 个
        """
        dense_results = dense_search(query, top_k=dense_k)
        bm25_results = self.bm25.search(query, top_k=bm25_k)
        merged = reciprocal_rank_fusion([dense_results, bm25_results])
        return merged[:top_k]


if __name__ == "__main__":
    texts = [
        "检索增强生成（RAG）通过引入外部知识库来提升 LLM 的回答质量。",
        "向量数据库存储文本的语义向量，支持快速相似度搜索。",
        "BM25 是一种基于词频和文档频率的经典信息检索算法。",
        "混合检索结合稠密检索和稀疏检索的优点，通常效果更好。",
        "Transformer 是自然语言处理的基础架构，基于自注意力机制。",
    ]

    retriever = HybridRetriever()
    retriever.add(texts)

    query = "如何提升语言模型的检索效果？"
    results = retriever.search(query, top_k=3)
    print(f"【混合检索】查询：{query}")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['rrf_score']:.4f}] {r['text'][:60]}")
