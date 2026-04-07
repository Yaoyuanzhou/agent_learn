"""
04_rag/chunker.py —— 5 种文本切块策略

运行：python 04_rag/chunker.py
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int          # 第几块
    start_char: int     # 在原文中的起始字符位置


# ── 策略 1：固定字符数切割 ─────────────────────────────────
def fixed_size_chunk(text: str, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """
    最简单的策略：每隔 size 个字符切一刀，相邻块重叠 overlap 个字符。
    优点：实现简单，速度快。
    缺点：可能把句子切断。
    """
    chunks = []
    step = size - overlap
    for i, start in enumerate(range(0, len(text), step)):
        end = min(start + size, len(text))
        chunks.append(Chunk(text=text[start:end], index=i, start_char=start))
        if end == len(text):
            break
    return chunks


# ── 策略 2：按句子切割 + 合并到目标大小 ─────────────────────
def sentence_chunk(text: str, max_size: int = 400, overlap_sentences: int = 1) -> list[Chunk]:
    """
    先按句子（。！？\n）切分，再合并句子直到接近 max_size。
    优点：不会把句子切断，语义更完整。
    """
    # 简单的中文句子分割
    sentences = re.split(r'([。！？\n])', text)
    # 把分隔符贴回句子后面
    sents = []
    for i in range(0, len(sentences) - 1, 2):
        sents.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
    if len(sentences) % 2 == 1:
        sents.append(sentences[-1])
    sents = [s for s in sents if s.strip()]

    chunks = []
    current_sents = []
    current_len = 0
    char_pos = 0

    for sent in sents:
        if current_len + len(sent) > max_size and current_sents:
            chunk_text = "".join(current_sents)
            chunks.append(Chunk(text=chunk_text, index=len(chunks), start_char=char_pos - current_len))
            # 保留 overlap_sentences 个句子作为重叠
            current_sents = current_sents[-overlap_sentences:]
            current_len = sum(len(s) for s in current_sents)
        current_sents.append(sent)
        current_len += len(sent)
        char_pos += len(sent)

    if current_sents:
        chunks.append(Chunk(text="".join(current_sents), index=len(chunks),
                            start_char=char_pos - current_len))
    return chunks


# ── 策略 3：按段落切割 ─────────────────────────────────────
def paragraph_chunk(text: str, max_size: int = 800) -> list[Chunk]:
    """
    按空行分段，段落太长时进一步切分。
    适合结构化文档（文章/报告）。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    pos = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_size:
            chunks.append(Chunk(text=para, index=len(chunks), start_char=pos))
        else:
            # 段落太长，用固定切割降级处理
            sub = fixed_size_chunk(para, size=max_size, overlap=50)
            for s in sub:
                chunks.append(Chunk(text=s.text, index=len(chunks), start_char=pos + s.start_char))
        pos += len(para) + 2
    return chunks


# ── 策略 4：按 Markdown 标题切割 ────────────────────────────
def markdown_chunk(text: str) -> list[Chunk]:
    """
    按 # ## ### 标题切分，每个标题和其内容为一块。
    适合技术文档、README。
    """
    pattern = re.compile(r'^(#{1,3}\s+.+)$', re.MULTILINE)
    positions = [(m.start(), m.group()) for m in pattern.finditer(text)]

    if not positions:
        return [Chunk(text=text, index=0, start_char=0)]

    chunks = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        chunks.append(Chunk(text=content, index=i, start_char=start))
    return chunks


# ── 策略 5：父子分块（Parent-Child） ─────────────────────────
def parent_child_chunk(text: str, parent_size: int = 800, child_size: int = 200) -> list[dict]:
    """
    父块用于上下文，子块用于精准检索。
    检索到子块后，返回对应的父块给 LLM，上下文更丰富。
    """
    parents = fixed_size_chunk(text, size=parent_size, overlap=100)
    result = []
    for parent in parents:
        children = fixed_size_chunk(parent.text, size=child_size, overlap=20)
        result.append({
            "parent": parent,
            "children": children
        })
    return result


if __name__ == "__main__":
    sample = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质。
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

可以设想，未来人工智能带来的科技产品，将会是人类智慧的「容器」。
人工智能可以对人的意识、思维的信息过程的模拟。
""" * 5  # 重复几次让文本足够长

    print(f"原文长度：{len(sample)} 字符\n")

    for name, chunks in [
        ("固定切割", fixed_size_chunk(sample, 200, 30)),
        ("句子切割", sentence_chunk(sample, 200, 1)),
        ("段落切割", paragraph_chunk(sample, 400)),
    ]:
        print(f"【{name}】→ {len(chunks)} 块，首块预览：{chunks[0].text[:60]}...")
