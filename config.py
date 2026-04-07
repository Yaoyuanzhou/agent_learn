"""
全局配置文件 —— 所有模块都从这里读取配置
用之前把 LLM_API_KEY / LLM_BASE_URL 换成你自己的
"""

# ── LLM 配置 ──────────────────────────────────────────────
# 国产模型示例（OpenAI 兼容接口），替换成你自己的即可
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY  = "sk-你的API_KEY"   # 替换成你自己的 key
LLM_MODEL    = "deepseek-chat"

# ── Embedding 配置（本地 sentence-transformers）────────────
# 首次运行会自动下载到 HuggingFace 缓存目录（~/.cache/huggingface）
EMBED_MODEL  = "BAAI/bge-small-zh-v1.5"        # 中文小模型，约 100MB
EMBED_DIM    = 512

# ── RAG 配置 ──────────────────────────────────────────────
CHUNK_SIZE   = 400    # token 数
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 20   # 第一阶段召回数
TOP_K_RERANK   = 5    # Reranker 精排后保留数

# ── 路径 ──────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_PATH     = os.path.join(DATA_DIR, "memory.db")
CHROMA_DIR  = os.path.join(DATA_DIR, "chroma")

os.makedirs(DATA_DIR, exist_ok=True)
