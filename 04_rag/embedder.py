"""
04_rag/embedder.py —— 本地 Embedding 封装（sentence-transformers）

首次运行会自动下载模型到 ~/.cache/huggingface/（约 100MB）
后续运行直接加载本地缓存，不需要联网。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMBED_MODEL
import numpy as np

_model = None


def get_embed_model():
    """懒加载，第一次调用时才加载模型（避免每次 import 都加载）"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[加载 Embedding 模型] {EMBED_MODEL}（首次需要下载）")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(text: str) -> np.ndarray:
    """单条文本向量化，返回归一化向量"""
    model = get_embed_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec


def embed_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    批量向量化，自动分批处理（防止 OOM）
    返回 shape (N, dim) 的矩阵
    """
    model = get_embed_model()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


if __name__ == "__main__":
    texts = ["机器学习是人工智能的子领域", "你是好人", "强化学习通过奖励信号学习"]
    vecs = embed_batch(texts)
    print(f"向量维度：{vecs.shape}")

    # 计算余弦相似度（向量已归一化，点积即余弦）
    sims = vecs @ vecs.T
    print("相似度矩阵：")
    for i, t in enumerate(texts):
        print(f"  [{t[:10]}] 与其他的相似度：{sims[i].round(3)}")
