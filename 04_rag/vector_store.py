"""
04_rag/vector_store.py —— Chroma 向量库封装（HNSW 索引）

Chroma 默认使用 HNSW 算法建立索引，比暴力搜索快几个数量级。
数据持久化到磁盘，重启后不需要重新 embed。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHROMA_DIR
from embedder import embed_batch
import chromadb
from chromadb.config import Settings


def get_collection(name: str = "rag_docs"):
    """获取或创建一个 Chroma 集合（自动持久化到磁盘）"""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name)


def add_documents(texts: list[str], metadatas: list[dict] = None,
                  collection_name: str = "rag_docs"):
    """
    向向量库添加文档
    texts: 文本列表
    metadatas: 可选，每条文本的元数据（如来源、章节等）
    """
    collection = get_collection(collection_name)

    # 生成向量
    vecs = embed_batch(texts)

    # 生成唯一 ID（基于当前已有数量偏移）
    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

    if metadatas is None:
        metadatas = [{"index": existing_count + i} for i in range(len(texts))]

    collection.add(
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"[向量库] 已添加 {len(texts)} 条，当前共 {collection.count()} 条")


def search(query: str, top_k: int = 5, collection_name: str = "rag_docs") -> list[dict]:
    """
    向量搜索（Dense 检索）
    返回 top_k 个最相似的文档，含文本、分数、元数据
    """
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return []

    query_vec = embed_batch([query])[0]
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )

    return [
        {
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],  # distance → similarity
            "metadata": results["metadatas"][0][i]
        }
        for i in range(len(results["documents"][0]))
    ]


if __name__ == "__main__":
    # 简单测试：存几条，然后搜索
    docs = [
        "RAG（检索增强生成）结合了信息检索和语言生成的优点。",
        "向量数据库使用近似最近邻算法快速找到相似向量。",
        "HNSW 是一种基于分层图的高效近似最近邻算法。",
        "Transformer 架构是现代大语言模型的基础。",
    ]
    add_documents(docs, [{"source": f"doc{i}"} for i in range(len(docs))])

    results = search("向量检索的算法是什么？", top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['text']}")
