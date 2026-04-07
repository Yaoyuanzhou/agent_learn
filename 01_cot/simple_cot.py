"""
01_cot/simple_cot.py —— Zero-Shot 和 Few-Shot 思维链演示

运行：python 01_cot/simple_cot.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm import chat


# ── Zero-Shot CoT ──────────────────────────────────────────
def zero_shot_cot(question: str) -> str:
    """
    Zero-Shot CoT：在问题末尾加一句 "请一步一步思考"
    不需要任何示例，模型自动展开推理过程
    """
    messages = [
        {"role": "system", "content": "你是一个善于逻辑推理的助手。"},
        {"role": "user",   "content": f"{question}\n\n请一步一步思考，最后给出答案。"}
    ]
    return chat(messages, temperature=0.3)


# ── Few-Shot CoT ───────────────────────────────────────────
# 提供 2 个示例，帮助模型学习「先推理再给答案」的格式
FEW_SHOT_EXAMPLES = """
示例1：
问题：小明有 3 个苹果，买了 2 袋苹果，每袋 4 个，共几个？
思考：
- 购买量：2 袋 × 4 个 = 8 个
- 总量：3 + 8 = 11 个
答案：11 个

示例2：
问题：一列火车以 60km/h 的速度行驶，2.5 小时能走多远？
思考：
- 距离 = 速度 × 时间 = 60 × 2.5 = 150 km
答案：150 km
""".strip()


def few_shot_cot(question: str) -> str:
    """
    Few-Shot CoT：提供带推理过程的示例，引导模型按相同格式输出
    """
    messages = [
        {"role": "system", "content": "你是一个善于逻辑推理的助手，请按示例格式输出推理过程和答案。"},
        {"role": "user",   "content": f"{FEW_SHOT_EXAMPLES}\n\n问题：{question}\n思考："}
    ]
    return chat(messages, temperature=0.3)


# ── 对比实验：有无 CoT 的差异 ──────────────────────────────
def direct_answer(question: str) -> str:
    """不加 CoT，直接要答案（对比用）"""
    return chat([
        {"role": "system", "content": "你是一个助手，请直接给出答案，不要解释。"},
        {"role": "user",   "content": question}
    ], temperature=0.3)


if __name__ == "__main__":
    question = "一家工厂每天生产 240 个零件，连续生产 5 天后休息 2 天为一个周期。12 天内共生产多少零件？"

    print("=" * 60)
    print("【直接回答（无 CoT）】")
    print(direct_answer(question))

    print("\n" + "=" * 60)
    print("【Zero-Shot CoT】")
    print(zero_shot_cot(question))

    print("\n" + "=" * 60)
    print("【Few-Shot CoT】")
    print(few_shot_cot(question))
