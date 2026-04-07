"""
agent.py —— 命令行入口，可直接用这个 Agent 执行任务

用法：
  python agent.py "北京今天天气怎么样？"
  python agent.py "帮我搜索 RAG 的核心组件，然后写一段总结" --mode multi
  python agent.py --demo       # 运行所有模块的演示
  python agent.py --help
"""
import sys, os, argparse

# 把所有子目录加入路径
BASE = os.path.dirname(__file__)
for d in ["02_react", "03_tools", "04_rag", "05_memory", "06_multi_agent"]:
    sys.path.insert(0, os.path.join(BASE, d))


def run_single(task: str, verbose: bool = True) -> str:
    """单 Agent 模式：ReAct Agent 处理任务"""
    from react_agent import run_react
    print(f"\n[单Agent模式] 任务：{task}\n")
    return run_react(task, verbose=verbose)


def run_multi(task: str) -> str:
    """多 Agent 模式：搜索 + 写作 Pipeline"""
    from orchestrator import pipeline_run
    # 从任务描述里提取主题（简单处理）
    topic = task.replace("帮我搜索", "").replace("写一篇关于", "").replace("的文章", "").strip()
    return pipeline_run(topic)


def run_rag(query: str, doc_dir: str = None) -> str:
    """RAG 模式：检索本地文档后回答"""
    from rag_pipeline import HybridRetriever, build_index, answer_with_rag

    retriever = HybridRetriever()

    # 如果指定了文档目录，先建索引
    if doc_dir and os.path.exists(doc_dir):
        docs = []
        for fname in os.listdir(doc_dir):
            if fname.endswith(".txt") or fname.endswith(".md"):
                with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
                    docs.append(f.read())
        if docs:
            build_index(docs, retriever)
            print(f"[RAG] 已加载 {len(docs)} 篇文档")

    return answer_with_rag(query, retriever, use_reranker=False)


def run_demo():
    """依次演示所有模块的核心功能"""
    print("\n" + "="*60)
    print("  Agent Learning Demo  ")
    print("="*60)

    demos = [
        ("01 CoT", "展示 Zero-Shot CoT 和 Few-Shot CoT 的区别"),
        ("02 ReAct", "Agent 调用工具计算圆的面积"),
        ("03 Function Calling", "演示 Function Calling 四轮对话流程"),
        ("05 Memory", "演示滑动窗口短期记忆"),
    ]

    for name, desc in demos:
        print(f"\n──── {name}: {desc} ────")

    print("\n提示：单独运行各模块查看完整演示：")
    print("  python 01_cot/simple_cot.py")
    print("  python 02_react/react_agent.py")
    print("  python 02_react/reflexion_agent.py")
    print("  python 03_tools/function_calling.py")
    print("  python 03_tools/parallel_tools.py")
    print("  python 04_rag/rag_pipeline.py")
    print("  python 05_memory/short_term.py")
    print("  python 06_multi_agent/orchestrator.py")


def main():
    parser = argparse.ArgumentParser(
        description="Agent Learning —— 本地可用的 AI Agent 学习实例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python agent.py "北京今天天气怎么样？"
  python agent.py "一个半径3的圆面积是多少" --mode react
  python agent.py "RAG 和 Fine-tuning 的区别" --mode multi
  python agent.py "什么是向量数据库" --mode rag --docs ./data/docs
  python agent.py --demo
        """
    )
    parser.add_argument("task", nargs="?", help="要执行的任务（自然语言）")
    parser.add_argument("--mode", choices=["react", "multi", "rag"], default="react",
                        help="Agent 模式：react(默认)/multi(多Agent)/rag(检索增强)")
    parser.add_argument("--docs", help="RAG 模式下的本地文档目录路径")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--quiet", action="store_true", help="减少详细输出")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.task:
        parser.print_help()
        return

    verbose = not args.quiet

    if args.mode == "react":
        result = run_single(args.task, verbose=verbose)
    elif args.mode == "multi":
        result = run_multi(args.task)
    elif args.mode == "rag":
        result = run_rag(args.task, doc_dir=args.docs)
    else:
        result = run_single(args.task)

    print(f"\n{'='*60}")
    print(f"【最终答案】\n{result}")


if __name__ == "__main__":
    main()
