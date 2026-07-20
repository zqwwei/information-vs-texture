"""
STEP 2 — A / A+ / B 三段 context 草稿 + token 数统计

实验变量隔离设计:
- A  : 三条偏好,以"陈述/结论"形式给出
- B  : 完全相同的三条偏好,以"对话经历+accept/reject 信号"形式给出
       B 中的"纠正语句"直接复用 A 的关键词(short / conclusion first / push, options)
- A+ : A 的内容不变,前面加上一段与三条偏好无关的中性传记性陈述,
       垫到与 B 大致等长。用来分离"长度差异"与"形式差异"。

审稿要点(由用户审):
1. A 与 B 是否逐条对应(三条偏好一一映射)
2. A 与 B 是否共享关键词
3. A+ 的填充内容是否真的与三条偏好(长度/结构/决策支持)无关
"""

from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")

# ---------- A: 陈述形式(压缩)----------
A = (
    "Alex prefers short answers. "
    "Alex likes the conclusion first, then the reasons. "
    "When Alex hesitates, Alex wants to be pushed toward a choice, not given more options."
)

# ---------- B: 经历形式(展开)----------
# 三条偏好,每条 ~5 回合,带 accept/reject 信号(Too long / Better 等)
# 关键:每条的"纠正语句"复用 A 的核心词
B = (
    # 偏好 1: short answers
    "Alex: Can you explain how compound interest works?\n"
    "Assistant: Compound interest occurs when the interest earned on an investment is added back to the principal, so that in the next period you earn interest on a larger amount, and this process repeats again and again, causing your balance to grow faster than it would with simple interest alone.\n"
    "Alex: Too long. Keep it short.\n"
    "Assistant: Your interest earns more interest over time.\n"
    "Alex: Better.\n"
    "\n"
    # 偏好 2: conclusion first, then reasons
    "Alex: Should I take the new job offer?\n"
    "Assistant: There are many factors to consider here, including the salary on offer, the growth opportunities the new role would provide, the work-life balance compared to your current position, the company culture, your long-term career goals, and how well the role aligns with your existing skills.\n"
    "Alex: Give the conclusion first, then the reasons.\n"
    "Assistant: Take it. The growth opportunities outweigh the risks.\n"
    "Alex: Better.\n"
    "\n"
    # 偏好 3: pushed, not more options
    "Alex: I can't decide between moving to Boston or staying here.\n"
    "Assistant: Both cities have their advantages. You might also consider a temporary move, a hybrid arrangement, negotiating remote work, or waiting another six months before deciding.\n"
    "Alex: Don't give me more options. Just push me toward one.\n"
    "Assistant: Go to Boston.\n"
    "Alex: Better."
)

# ---------- A+: A + 中性传记性填充(长度对照组)----------
# 填充内容:关于 Alex 的中性事实,刻意避开:
#   - 长度/简洁/详尽相关 (避免污染偏好 1)
#   - 结构/组织/顺序相关 (避免污染偏好 2)
#   - 决策/犹豫/选择相关 (避免污染偏好 3)
# 选用纯粹的生活/物件/地理类事实。
PADDING = (
    "Alex is thirty-four years old. "
    "Alex lives in a small apartment near the river. "
    "Alex commutes to work by bicycle most mornings. "
    "Alex drinks coffee in the afternoon rather than in the morning. "
    "Alex keeps a journal on the desk by the window. "
    "Alex owns a gray cat named Pepper. "
    "Alex grew up in a coastal town and moved to the city after college. "
    "Alex enjoys cooking on weekends and often tries new recipes. "
    "Alex reads before sleeping every night. "
    "Alex's favorite season is autumn. "
    "Alex collects postcards from places friends have visited. "
    "Alex plays the piano occasionally but never learned to read sheet music. "
    "Alex prefers tea in the evening. "
    "Alex has a sister who lives abroad. "
    "Alex keeps the kitchen window open while cooking. "
    "Alex bought the apartment three years ago. "
    "Alex paints the walls a new color every spring. "
    "Alex grows basil and mint on the kitchen windowsill. "
    "Alex bikes to the farmers market on Saturday afternoons. "
    "Alex's apartment building was built in the nineteen-twenties. "
    "Alex stores winter clothes in a wooden trunk under the bed. "
    "Alex received the piano from a neighbor who was moving away. "
    "Alex carries a leather notebook in the bag for grocery lists. "
    "Alex sometimes walks along the river in the evening after dinner."
)
A_plus = PADDING + " " + A


def n_tokens(s: str) -> int:
    return len(tok(s).input_ids)


if __name__ == "__main__":
    print("=" * 60)
    print("Token counts (GPT-2 tokenizer):")
    print("=" * 60)
    print(f"  A   : {n_tokens(A):4d} tokens")
    print(f"  A+  : {n_tokens(A_plus):4d} tokens   (target ≈ B)")
    print(f"  B   : {n_tokens(B):4d} tokens")
    print()
    print(f"  A+ / B 长度比: {n_tokens(A_plus) / n_tokens(B):.2f}  (越接近 1.0 越好)")
    print()
    print("=" * 60)
    print("A:")
    print("=" * 60)
    print(A)
    print()
    print("=" * 60)
    print("A+:")
    print("=" * 60)
    print(A_plus)
    print()
    print("=" * 60)
    print("B:")
    print("=" * 60)
    print(B)
