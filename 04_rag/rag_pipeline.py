"""
04_rag/rag_pipeline.py —— 完整 RAG 流水线（串联所有组件）

离线阶段：文档 → 切块 → Embedding → 存入向量库
在线阶段：Query → 混合检索 → Reranker 精排 → LLM 生成答案

运行：python 04_rag/rag_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chunker import sentence_chunk
from embedder import embed_batch
from retriever import HybridRetriever
from llm import chat
from config import TOP_K_RETRIEVE, TOP_K_RERANK


def build_index(documents: list[str], retriever: HybridRetriever):
    """离线阶段：把文档切块并存入索引"""
    all_chunks = []
    all_meta = []
    for doc_id, doc in enumerate(documents):
        chunks = sentence_chunk(doc, max_size=400)
        for chunk in chunks:
            all_chunks.append(chunk.text)
            all_meta.append({"doc_id": doc_id, "chunk_index": chunk.index})

    print(f"[离线] {len(documents)} 篇文档 → {len(all_chunks)} 个 Chunk")
    retriever.add(all_chunks, all_meta)
    return all_chunks


def answer_with_rag(query: str, retriever: HybridRetriever, use_reranker: bool = True) -> str:
    """在线阶段：检索 + 精排 + 生成"""
    # 第一阶段：混合检索
    candidates = retriever.search(query, top_k=TOP_K_RETRIEVE)

    # 第二阶段：Reranker 精排（可选，慢但更准）
    if use_reranker and candidates:
        try:
            from reranker import rerank
            final_docs = rerank(query, candidates, top_k=TOP_K_RERANK)
        except Exception as e:
            print(f"[Reranker 不可用，跳过] {e}")
            final_docs = candidates[:TOP_K_RERANK]
    else:
        final_docs = candidates[:TOP_K_RERANK]

    # 构建 Context
    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(final_docs))

    # 生成答案
    messages = [
        {"role": "system", "content":
            "你是一个知识问答助手。请基于提供的参考资料回答问题。"
            "如果资料中没有相关信息，明确说明不知道，不要编造。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
    ]
    return chat(messages, temperature=0.3)


if __name__ == "__main__":
    # 示例文档
    documents = [
        """RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的技术。
        它通过检索外部知识库来增强语言模型的回答质量。
        RAG 的核心优势是可以使用实时更新的知识，不受模型训练数据截止日期的限制。
        RAG 的主要组件包括：文档向量化、向量索引、检索器、Reranker 和生成器。""",

        """向量数据库是 RAG 系统的核心存储组件。
        常见的向量数据库有 Chroma、Pinecone、Weaviate、Milvus 等。
        它们使用 HNSW（分层小世界图）或 IVF（倒排文件索引）等算法进行高效的近似最近邻搜索。
        Chroma 是一个轻量级的本地向量数据库，适合原型开发和学习。""",

        """Embedding 模型将文本转换为高维向量，使语义相近的文本在向量空间中距离更近。
        常用的 Embedding 模型包括 OpenAI 的 text-embedding-3、BAAI 的 BGE 系列（中文效果好）。
        BGE-small-zh-v1.5 是一个轻量级的中文 Embedding 模型，约 100MB，适合本地运行。""",
    ]

    retriever = HybridRetriever()
    build_index(documents, retriever)

    queries = [
        "RAG 的核心组件有哪些？",
        "推荐什么向量数据库用于学习？",
        "中文 Embedding 模型怎么选？",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"【问题】{q}")
        answer = answer_with_rag(q, retriever, use_reranker=False)  # 先不用 Reranker
        print(f"【答案】{answer}")
