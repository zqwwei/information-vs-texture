"""
STEP 4 — B' + B_noise (Phase 0 收口)

B_prime  (偏好对照): 同名 Alex / 同三场景(提问句逐字相同)/ 同镜像结构 / 同回合数,
                     唯一变量 = 三条偏好换成与 Alex 原三条【正交】(不同轴,不是反着来)。
                     纠正关键词 analogy / emotionally hard / generic,与 B 的
                     short / conclusion first / more options 无词汇重叠。
B_noise  (表面噪声地板): 同名 Alex / 同三条偏好 / 同场景 / 同结构 / 同回合数,
                     唯一变量 = 把每句【改写措辞,不改语义】(含 accept 信号 Better → Got it)。
                     作用 = 测「人/偏好都不变、只换说法」时表示移动多少 = 这套 readout 的噪声底。

控制变量(审稿盯这几条):
  1. B / B' 三个场景【提问句逐字相同】,只有偏好(经纠正暴露)不同
  2. B' 三条偏好与 Alex 原三条【不在同一轴】,纠正关键词与 B 无词汇重叠
  3. 每场景「故意错的回答」长度与 B 对应回合大致对齐(错答是 token 大头)
  4. B' / B_noise 总 token 控制在 B ±10 内(跑完核对)
"""

from transformers import GPT2Tokenizer
from step2_contexts import A, A_plus, B  # 复用 STEP 2

tok = GPT2Tokenizer.from_pretrained("gpt2")


# ---------- B': 同名 Alex,正交偏好 ----------
B_prime = (
    # 偏好 1: 用类比,不要干巴巴的定义  (vs Alex 的 short)
    "Alex: Can you explain how compound interest works?\n"
    "Assistant: Compound interest is defined recursively: each period's interest accrues on the accumulated balance rather than the original principal, producing geometric growth whose effective rate depends jointly on the nominal rate, the compounding frequency, and the elapsed term.\n"
    "Alex: Don't give me dry definitions. Use an analogy.\n"
    "Assistant: It's a snowball rolling downhill, growing bigger the farther it goes.\n"
    "Alex: Better.\n"
    "\n"
    # 偏好 2: 认到情绪重量,不要当成纯计算  (vs Alex 的 conclusion first)
    "Alex: Should I take the new job offer?\n"
    "Assistant: From an expected-value standpoint the answer is to accept, since the compensation delta, the projected trajectory, and the option value of the role jointly dominate the downside paths.\n"
    "Alex: Don't treat this like a calculation. It's emotionally hard.\n"
    "Assistant: This is genuinely hard, and the anxiety you feel makes sense. Take it.\n"
    "Alex: Better.\n"
    "\n"
    # 偏好 3: 贴合我的具体处境,不要泛泛而谈  (vs Alex 的 push, not options)
    "Alex: I can't decide between moving to Boston or staying here.\n"
    "Assistant: Big decisions like this always come down to weighing the pros and cons, listening to your gut, and giving yourself enough time before you feel sure about it.\n"
    "Alex: Don't give me generic advice. Tie it to my situation.\n"
    "Assistant: Your whole support network is here, and that pull is real.\n"
    "Alex: Better."
)


# ---------- B_noise: 同名 Alex,同三条偏好,改写措辞 ----------
B_noise = (
    # 偏好 1: short answers (改写)
    "Alex: How does compound interest actually work?\n"
    "Assistant: When you earn interest, that interest gets added onto your original deposit, and the next round of interest is then figured on the new and larger total, and because this keeps repeating, your money ends up growing faster than it would.\n"
    "Alex: That's way too much. Trim it down.\n"
    "Assistant: Over time, your interest starts earning interest of its own.\n"
    "Alex: Got it.\n"
    "\n"
    # 偏好 2: conclusion first (改写)
    "Alex: Should I accept this new job offer or not?\n"
    "Assistant: There's a lot feeding into a decision like this — what they'd pay you, how much room you'd have to grow, how the hours stack up against your current job, where you're hoping to end up, and whether the work itself is a fit.\n"
    "Alex: Lead with the verdict, then walk me through why.\n"
    "Assistant: Accept it — the room to grow is worth more than the risk.\n"
    "Alex: Got it.\n"
    "\n"
    # 偏好 3: pushed, not options (改写)
    "Alex: I'm torn between relocating to Boston and just staying put.\n"
    "Assistant: Each option has its merits. You could also try a temporary move, set up a part-remote arrangement, ask about working from home, or wait before deciding.\n"
    "Alex: Stop handing me alternatives. Just tell me which one.\n"
    "Assistant: Relocate to Boston.\n"
    "Alex: Got it."
)


def n_tokens(s: str) -> int:
    return len(tok(s).input_ids)


if __name__ == "__main__":
    print("=" * 60)
    print("Token counts (GPT-2 tokenizer) — 目标:B' / B_noise 在 B ±10 内")
    print("=" * 60)
    print(f"  B        : {n_tokens(B):4d} tokens   (基准)")
    print(f"  B'       : {n_tokens(B_prime):4d} tokens   (Δ vs B = {n_tokens(B_prime)-n_tokens(B):+d})")
    print(f"  B_noise  : {n_tokens(B_noise):4d} tokens   (Δ vs B = {n_tokens(B_noise)-n_tokens(B):+d})")
    print()
    print("若任一 |Δ| > 10:只在【错答】回合上加/减一个从句微调,")
    print("不要动纠正语句 / 偏好关键词 / 提问句。")
