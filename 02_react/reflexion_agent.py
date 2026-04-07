"""
02_react/reflexion_agent.py —— Reflexion 变体：失败 → 反思 → 写记忆 → 重试

核心思路：
  任务失败后，让 LLM 写一段反思（「哪里错了？为什么？」）
  把反思存入 memory 列表，下次重试时把历史反思注入 Prompt
  模拟人类「从错误中学习」的过程

运行：python 02_react/reflexion_agent.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm import chat

MAX_RETRIES = 3


def evaluate_answer(answer: str, task: str) -> tuple[bool, str]:
    """
    用 LLM 当评估器，判断回答是否满足要求
    返回 (是否通过, 失败原因)
    真实项目里可以替换成跑单元测试、规则校验等确定性方法
    """
    prompt = f"""请判断以下回答是否完整、准确地回答了问题。

问题：{task}
回答：{answer}

如果回答良好，只输出：PASS
如果有问题，输出：FAIL: [具体问题描述]"""

    result = chat([{"role": "user", "content": prompt}], temperature=0)
    passed = result.strip().startswith("PASS")
    reason = result.strip() if not passed else ""
    return passed, reason


def reflexion_agent(task: str) -> str:
    """
    Reflexion Agent 主循环：
    1. 执行任务
    2. 评估结果
    3. 失败则生成反思，写入记忆
    4. 带着记忆重试
    """
    memory: list[str] = []   # 这就是「写入记忆」的地方——一个普通列表

    for attempt in range(MAX_RETRIES):
        print(f"\n─── 尝试 {attempt + 1}/{MAX_RETRIES} ───")

        # 把历史反思注入 Prompt（如果有的话）
        reflection_context = ""
        if memory:
            reflection_context = "\n\n【历史反思记录，请避免重犯这些错误】\n" + "\n".join(
                f"- 尝试{i+1}的反思：{r}" for i, r in enumerate(memory)
            )

        messages = [
            {"role": "system", "content": "你是一个严谨的助手，请尽力给出完整准确的回答。"},
            {"role": "user",   "content": f"{task}{reflection_context}"}
        ]
        answer = chat(messages, temperature=0.5)
        print(f"[回答] {answer[:300]}...")

        # 评估回答质量
        passed, reason = evaluate_answer(answer, task)

        if passed:
            print(f"[评估] ✅ 通过")
            return answer
        else:
            print(f"[评估] ❌ 未通过：{reason}")

            # 让 LLM 生成反思，写入记忆
            reflection = chat([
                {"role": "user", "content":
                    f"我的回答「{answer[:200]}」被评为不合格，原因是：{reason}\n"
                    f"请用一句话总结：我哪里做错了，下次怎么改进？"}
            ], temperature=0.3)
            memory.append(reflection)  # ← 写入记忆
            print(f"[反思] 已记录：{reflection}")

    return f"[已尝试 {MAX_RETRIES} 次，返回最后一次结果] {answer}"


if __name__ == "__main__":
    task = "列出 5 种常见的排序算法，并说明每种的时间复杂度（最好/最坏/平均）和适用场景。"
    print(f"【任务】{task}")
    result = reflexion_agent(task)
    print(f"\n【最终答案】\n{result}")
