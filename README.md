# 🤖 Agent Learning

本地可运行的 AI Agent 学习实例代码库，覆盖 CoT / ReAct / Reflexion / Function Calling / RAG / 记忆管理 / 多 Agent 全技术栈。

每个模块独立可运行 · 有真实 LLM 调用 · 全部代码已测试通过 ✅

## 特性

- 🧠 **6 个核心模块**：从最简单的 CoT 到完整的多 Agent 协作系统
- 🔌 **DeepSeek / 任意国产模型**：OpenAI 兼容接口，改一行配置即可切换
- 📦 **BGE 本地 Embedding**：无需 API Key，sentence-transformers 本地推理
- 💾 **RAG 向量库持久化**：ChromaDB 本地存储，重启不丢数据
- ⌨️ **命令行一行调用**：`python agent.py "你的问题"` 即可运行

---

## 快速开始

### 第一步：配置 API Key

修改 `config.py`，填入大模型接口信息：

```python
LLM_BASE_URL = "https://api.deepseek.com/v1"   # OpenAI 兼容接口
LLM_API_KEY  = "sk-你的key"
LLM_MODEL    = "deepseek-chat"                  # 或 qwen-turbo / qwen-plus 等
```

### 第二步：安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
# 首次安装较慢（主要是 torch ~800MB），建议用清华镜像：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第三步：运行

```bash
# 单 Agent（ReAct 模式，默认）
python agent.py "北京今天天气怎么样？"
python agent.py "计算 sqrt(256) + sqrt(144)"

# 多 Agent 模式（搜索 + 写作 Pipeline）
python agent.py "向量数据库的核心原理" --mode multi

# RAG 模式（加载本地文档后回答）
python agent.py "这批文档讲了什么？" --mode rag --docs ./data/docs
```

---

## 目录结构

```
agent-learning/
├── config.py            ⚙️  全局配置（API Key / 模型名 / 路径）← 只改这一个文件
├── llm.py               🔌 LLM 调用封装，所有模块统一从这里调
├── agent.py             🎮 命令行入口（react / multi / rag 三种模式）
├── requirements.txt     📦 依赖：openai / sentence-transformers / chromadb / rank-bm25
├── design.html          📖 架构设计文档（可浏览器打开）
│
├── 01_cot/              思维链基础：Zero-Shot / Few-Shot CoT 对比
│   └── simple_cot.py
│
├── 02_react/            ReAct 推理框架 + Reflexion 失败反思变体
│   ├── tools.py             可用工具集（search / calculator / weather）
│   ├── react_agent.py       Thought → Action → Observation 主循环
│   └── reflexion_agent.py   失败 → LLM生成反思 → 写入记忆 → 重试
│
├── 03_tools/            Function Calling 底层机制详解
│   ├── function_calling.py  四轮对话完整流程（打印每一轮）
│   ├── parallel_tools.py    asyncio 并发执行多个工具
│   └── error_handling.py    三类错误处理 + LLM 自动恢复
│
├── 04_rag/              RAG 检索增强生成全链路
│   ├── chunker.py           5种切块策略（固定/句子/段落/Markdown/父子）
│   ├── embedder.py          本地 BGE Embedding 封装（首次自动下载）
│   ├── vector_store.py      ChromaDB 向量库（持久化到磁盘）
│   ├── retriever.py         Dense / BM25 / Hybrid+RRF 三种检索策略
│   ├── reranker.py          Cross-Encoder 两阶段精排
│   └── rag_pipeline.py      串联所有组件的完整端到端流水线
│
├── 05_memory/           记忆管理
│   ├── short_term.py        滑动窗口 + LLM 摘要压缩（防止 Context 超限）
│   └── long_term.py         SQLite 持久化（用户偏好/反思/摘要归档）
│
└── 06_multi_agent/      多 Agent 协作系统
    ├── base_agent.py        Agent 基类：Prompt + Tools + 执行循环
    ├── search_agent.py      搜索专家（只有检索工具）
    ├── writing_agent.py     写作专家（只有写作工具）
    └── orchestrator.py      协调者：Pipeline 串行 + 共享黑板两种模式
```

---

## 学习路径

```
01_cot → 02_react → 03_tools → 04_rag → 05_memory → 06_multi_agent
```

理解思路：
- **01→02→03**：搞懂单 Agent 的核心循环和工具调用
- **04**：给 Agent 接上外部知识库（RAG）
- **05**：让 Agent 有记忆，不再每次从零开始
- **06**：多个 Agent 协作完成复杂任务

每个模块都可以独立运行：

```bash
python 01_cot/simple_cot.py           # CoT 对比实验
python 02_react/react_agent.py        # ReAct 主循环
python 02_react/reflexion_agent.py    # 失败→反思→重试
python 03_tools/function_calling.py   # 四轮对话流程
python 04_rag/rag_pipeline.py         # 完整 RAG 流水线
python 06_multi_agent/orchestrator.py # 多 Agent 协作
```

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `openai` | LLM 调用（兼容 DeepSeek / Qwen 等国产模型） |
| `sentence-transformers` | 本地 BGE Embedding，文本向量化 |
| `chromadb` | 向量数据库，持久化存储文档向量 |
| `rank-bm25` | BM25 关键词检索，混合检索用 |
| `torch` | sentence-transformers 的运行依赖 |

> 首次运行 RAG 模块时，会自动从 HuggingFace 下载 BGE 模型（~100MB），
> 保存到 `~/.cache/huggingface/`，后续不再重新下载。

---

## 架构文档

打开 `design.html` 可查看完整架构设计文档，包含：
- 模块依赖关系图（SVG）
- 每个文件的作用和执行流程
- ReAct / RAG / Multi-Agent 流程图
- 关键代码解读

```bash
open design.html   # macOS
```
