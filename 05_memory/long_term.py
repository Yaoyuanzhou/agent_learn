"""
05_memory/long_term.py —— 长期记忆：SQLite 持久化存储

短期记忆（Context Window）进程结束就没了。
长期记忆把重要信息写入 SQLite，下次启动也能读到。
常见用途：用户偏好、Reflexion 反思记录、对话摘要归档。

运行：python 05_memory/long_term.py
"""
import sys, os, sqlite3, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_PATH


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """创建表结构（如果不存在）"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            category  TEXT NOT NULL,      -- 类别：reflection / preference / fact / summary
            content   TEXT NOT NULL,      -- 记忆内容
            metadata  TEXT DEFAULT '{}',  -- 额外元数据（JSON）
            created   REAL NOT NULL       -- Unix 时间戳
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
    conn.commit()
    conn.close()


def save(category: str, content: str, metadata: dict = None):
    """写入一条记忆"""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO memories (category, content, metadata, created) VALUES (?, ?, ?, ?)",
        (category, content, json.dumps(metadata or {}), time.time())
    )
    conn.commit()
    conn.close()


def load(category: str = None, limit: int = 20) -> list[dict]:
    """读取记忆，可按类别过滤，按时间降序"""
    conn = _get_conn()
    if category:
        rows = conn.execute(
            "SELECT * FROM memories WHERE category = ? ORDER BY created DESC LIMIT ?",
            (category, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "category": r["category"],
            "content": r["content"],
            "metadata": json.loads(r["metadata"]),
            "created": r["created"]
        }
        for r in rows
    ]


def delete(memory_id: int):
    conn = _get_conn()
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()


# ── 对外接口：专门给 Reflexion Agent 用 ────────────────────
def save_reflection(task: str, reflection: str, attempt: int):
    """保存 Reflexion 反思记录"""
    save("reflection", reflection, metadata={"task": task[:100], "attempt": attempt})


def load_reflections(task_hint: str = None, limit: int = 5) -> list[str]:
    """读取最近几条反思记录"""
    rows = load("reflection", limit=limit)
    return [r["content"] for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"[数据库] {DB_PATH}")

    # 写入测试数据
    save("preference", "用户喜欢用 Python，不喜欢 Java")
    save("fact", "用户的名字叫小明，今年 25 岁")
    save_reflection("排序算法", "我漏掉了时间复杂度的最坏情况，下次要三种情况都列出来", attempt=1)
    save_reflection("排序算法", "快排的空间复杂度我写成了 O(n)，实际递归栈是 O(log n)", attempt=2)

    # 读取
    print("\n【所有记忆】")
    for m in load(limit=10):
        print(f"  [{m['category']}] {m['content'][:60]}")

    print("\n【只看反思】")
    for r in load_reflections():
        print(f"  - {r}")
