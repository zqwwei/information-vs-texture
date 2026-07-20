# HAND-OFF SPEC: STEP 4 — Phase 0 收口 (B' + B_noise)

## 给 Claude Code 的话
接 STEP 3。这是**已经锁定**的 Phase 0,核心设计决定都做完了——**不要再问那三件事**(偏好 / 场景 / 长度),直接执行:
1. 建两段新 context(草稿在 §1,可直接粘贴成 `step4_contexts.py`)
2. 跑五段、逐层算距离、出三张图、出表、打印 quick read(代码在 §2)
3. 把结果带回给 Ziqi

原则不变:**跑出结果再说下一步。** 这一轮跑完**不要**往前做 Phase 1(linear probe)——那是另开的 spec。

---

## 这一轮回答的精确问题

STEP 3 留下的问题:`A+↔B`(纯形式)有差异,但里面混着「对话结构」和「对人 / 偏好的表示」,distance 分不开。
这一轮用两段新 context 把它收窄:

- **B'**:和 B **同名(Alex)、同三个场景、同镜像结构、同回合数**,唯一变量是**三条偏好换成与 Alex 原三条正交的新三条**。
- **B_noise**:和 B **同名、同三条偏好、同场景、同结构**,唯一变量是**把每句改写措辞(不改语义)**。这是**噪声地板**。

**核心判据(连续,不是二元):**
> 在中间带(layer 5–9)的 cosine 上,`B↔B'`(换偏好内容)是否**明显比 `B↔B_noise`(只改措辞)更不相似**?

- 是、且差距明显 → 换偏好内容比单纯改写更能移动表示 → 模型对「偏好内容」敏感(初步、单实例)
- 否、贴着地板 → 在 GPT-2 这个尺度 + distance 这个框架下,没看到「偏好内容」的额外信号

两个都是有用的 claim。**用差值 / 比例描述,不要说「假设成立 / 不成立」。**

为什么用 floor 而不是拿 `A+↔B` 当尺子:`A+↔B` 本身脏(混了形式+密度+对话结构+额外词);`A↔A+` 在 layer 0 是纯位置 artifact。`B↔B_noise` 才是和 `B↔B'` 唯一变量对齐的干净对照。

---

## 已锁定(不要改、不要再问)

- **同名**:B 和 B' 都用 `Alex`(消除「Alex vs Sam」名字 token 的 embedding 差异这个 confound)。
- **正交、不反向**:B' 三条偏好与 Alex 原三条**不在同一轴**,纠正关键词与 B **无词汇重叠**(避免「同轴反向」造成的词汇对立 → 否则差异可能只是浅层词汇,不是深层)。
- **同三个场景**:复利 / 工作 offer / 搬家,且**提问句逐字相同**,只有偏好(经纠正暴露)不同。
- **逐回合长度对齐**:每场景「故意错的回答」长度与 B 对应回合大致对齐;总 token 控制在 B ±10 内。
- **读点**:**mean-pool 为主**,**last-token 仅作对照**。(last-token 在本轮是最差读点:B/B'/B_noise 结尾都几乎是 `Alex: Better.`,人的差异被压在结尾之外;mean-pool 不引入新 token、不重塑 attention,对 `B↔B'` 与 `B↔B_noise` 的稀释同向 → 比值仍可解释。)
- **headline 用中间带(layer 5–9)cosine。**

**明确否掉、不要重新引入的:**
- ❌ probe 后缀(会重塑 attention、把 claim 换成「被询问时如何总结」;真要做 probe-style 读出直接上 Phase 1 的 linear probe)。
- ❌ B' 用「反向偏好」(同轴对立词汇 confound)。
- ❌ 把 layer 8–11 的 L2 平均成一个 headline summary(深层 L2 大,大半是 residual stream 范数自然膨胀,不是内容)。
- ❌ 用最后一层 cosine 比内容(末层各向异性:cos 全 0.98+ 但 L2 仍差很多,方向已不可区分)。

---

## §1 — 两段新 context 草稿(Ziqi 审一件事)

**Ziqi 只盯一件事:B' 那三条偏好是否和 Alex 原三条真正正交。** 结构和每回合长度我已对齐,不满意就只换偏好内容,结构不用动。

| 轴 | Alex (B) | Alex (B') | 纠正关键词 B → B' |
|---|---|---|---|
| 1 | 简短 (length) | 用类比 (modality) | short → analogy / dry definitions |
| 2 | 结论先行 (order) | 认到情绪重量 (tone) | conclusion first → emotionally hard / not a calculation |
| 3 | 被推一把 (decisiveness) | 贴合我的处境 (specificity) | more options / push → generic / my situation |

检查:三对关键词左右无词汇重叠 → 没有「同轴反向」的对立。✓

```python
# step4_contexts.py
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
    "Assistant: Compound interest is defined recursively: each period's interest accrues on the accumulated balance rather than the original principal, producing geometric growth whose effective rate depends jointly on the nominal rate, the compounding frequency, and the elapsed term, following directly from the standard exponential accumulation function.\n"
    "Alex: Don't give me dry definitions. Use an analogy.\n"
    "Assistant: It's a snowball rolling downhill, growing bigger the farther it goes.\n"
    "Alex: Better.\n"
    "\n"
    # 偏好 2: 认到情绪重量,不要当成纯计算  (vs Alex 的 conclusion first)
    "Alex: Should I take the new job offer?\n"
    "Assistant: From an expected-value standpoint the answer is to accept, since the compensation delta, the projected trajectory, and the option value of the role jointly dominate the downside paths, conditional on the stated terms holding and transition friction staying within an ordinary band.\n"
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
    "Assistant: When you earn interest, that interest gets added onto your original deposit, and the next round of interest is then figured on the new and larger total, and because this keeps repeating every period, your money ends up growing faster than it would if the interest were never reinvested.\n"
    "Alex: That's way too much. Trim it down.\n"
    "Assistant: Over time, your interest starts earning interest of its own.\n"
    "Alex: Got it.\n"
    "\n"
    # 偏好 2: conclusion first (改写)
    "Alex: Should I accept this new job offer or not?\n"
    "Assistant: There's a lot feeding into a decision like this — what they'd pay you, how much room you'd have to grow, how the hours stack up against your current job, what the team is like, where you're hoping to end up, and whether the work itself is a fit for you.\n"
    "Alex: Lead with the verdict, then walk me through why.\n"
    "Assistant: Accept it — the room to grow is worth more than the risk.\n"
    "Alex: Got it.\n"
    "\n"
    # 偏好 3: pushed, not options (改写)
    "Alex: I'm torn between relocating to Boston and just staying put.\n"
    "Assistant: Each option has its merits. You could also try a temporary move, set up a part-remote arrangement, ask about working from home, or simply wait half a year before you decide.\n"
    "Alex: Stop handing me alternatives. Just tell me which one.\n"
    "Assistant: Relocate to Boston.\n"
    "Alex: Got it."
)


def n_tokens(s: str) -> int:
    return len(tok(s).input_ids)


print("=" * 60)
print("Token counts (GPT-2 tokenizer) — 目标:B' / B_noise 在 B ±10 内")
print("=" * 60)
print(f"  B        : {n_tokens(B):4d} tokens   (基准)")
print(f"  B'       : {n_tokens(B_prime):4d} tokens   (Δ vs B = {n_tokens(B_prime)-n_tokens(B):+d})")
print(f"  B_noise  : {n_tokens(B_noise):4d} tokens   (Δ vs B = {n_tokens(B_noise)-n_tokens(B):+d})")
print()
print("若任一 |Δ| > 10:只在【错答】回合上加/减一个从句微调,")
print("不要动纠正语句 / 偏好关键词 / 提问句。")
```

> 注:`step2_contexts.py` 在模块层有一堆 print,被 import 时会刷屏。可选清理:把它的打印块包进 `if __name__ == "__main__":`。不做也行,只是输出更干净。

---

## §2 — 对比脚本

```python
# step4_compare.py
"""
STEP 4 — 五段对比 (Phase 0 收口)

五段: A / A+ / B / B' / B_noise
读点: 每层对整段 mean-pool(主) + 取 last token(对照)
五对距离:
  A ↔ B        form + length   (原始)
  A ↔ A+       length only     (原始;layer0 是纯位置 artifact,别当内容信号)
  A+ ↔ B       form only       (原始)
  B ↔ B'       preference      (新增 ← 关键:换偏好内容)
  B ↔ B_noise  surface floor   (新增 ← 噪声地板:同人同偏好,只改措辞)

核心判据(连续,不是二元):
  中间带 layer 5–9 的 cosine 上,B↔B' 是否明显比 B↔B_noise 更不相似(cos 更低)?

读数硬约束:
  - headline 用【中间带 layer 5–9 的 cosine】。
  - 不用最后一层 cosine 比内容(末层各向异性)。
  - 不把 layer 8–11 的 L2 平均成一个 summary 数(深层 L2 大半是范数膨胀)。
  - L2 曲线照画(延续 step3),解读以 cosine 为准。
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

from step2_contexts import A, A_plus, B
from step4_contexts import B_prime, B_noise

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

CONTEXTS = {"A": A, "A_plus": A_plus, "B": B, "B_prime": B_prime, "B_noise": B_noise}
MID = list(range(5, 10))  # 中间带 layer 5–9


def per_layer(text, mode):
    """mode='mean' → 每层对所有 token 平均; mode='last' → 每层取最后一个 token。
    单序列、batch=1、无 padding,所以直接对 seq 维平均即可。"""
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    if mode == "mean":
        return [hs[0].mean(dim=0).clone() for hs in out.hidden_states]
    else:  # "last"
        return [hs[0, -1, :].clone() for hs in out.hidden_states]


def compare(v1, v2):
    cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    l2 = (v1 - v2).norm().item()
    return cos, l2


PAIRS = {
    "A ↔ B        (form+len)":   ("A", "B"),
    "A ↔ A+       (len only)":   ("A", "A_plus"),
    "A+ ↔ B       (form only)":  ("A_plus", "B"),
    "B ↔ B'       (preference)": ("B", "B_prime"),
    "B ↔ B_noise  (floor)":      ("B", "B_noise"),
}


def run(mode):
    vecs = {name: per_layer(txt, mode) for name, txt in CONTEXTS.items()}
    n_layers = len(next(iter(vecs.values())))
    res = {}
    for name, (l, r) in PAIRS.items():
        cos, l2 = [], []
        for k in range(n_layers):
            c, d = compare(vecs[l][k], vecs[r][k])
            cos.append(c); l2.append(d)
        res[name] = {"cos": cos, "l2": l2}
    return res, n_layers


print("Encoding 5 contexts through GPT-2 (mean-pool + last-token)...")
res_mean, n_layers = run("mean")
res_last, _        = run("last")
print(f"  layers = {n_layers}\n")


def print_table(res, title):
    print("=" * 104); print(title); print("=" * 104)
    header = "layer | " + " | ".join(f"{n[:24]:>24}" for n in PAIRS)
    print(header); print("-" * len(header))
    for k in range(n_layers):
        cells = " | ".join(f"{res[n]['cos'][k]:>24.4f}" for n in PAIRS)
        print(f"{k:>5} | {cells}")


print_table(res_mean, "Cosine per layer — MEAN-POOL (primary). higher = more similar")
print()
print_table(res_last, "Cosine per layer — LAST-TOKEN (control). higher = more similar")


# ---- 图 1 + 1b: 逐层曲线 ----
def layer_plot(res, fname, suptitle):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    xs = list(range(n_layers))
    for name in PAIRS:
        ax[0].plot(xs, res[name]["cos"], marker="o", label=name)
        ax[1].plot(xs, res[name]["l2"],  marker="o", label=name)
    for a in ax:
        a.axvspan(5, 9, alpha=0.08, color="gray")  # 中间带
        a.set_xlabel("Layer (0=embed, 1–12=blocks)"); a.set_xticks(xs)
        a.grid(True, alpha=0.3); a.legend(fontsize=8, loc="best")
    ax[0].set_title("Cosine by layer (higher=more similar)\nshaded = mid-band 5–9 (headline)")
    ax[0].set_ylabel("cosine")
    ax[1].set_title("L2 by layer (read cautiously: late-layer L2 ≈ norm growth)")
    ax[1].set_ylabel("L2")
    fig.suptitle(suptitle, fontsize=11); fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight"); print(f"saved -> {fname}")


layer_plot(res_mean, "step4_layers.png",
           "5-pair activation differences — MEAN-POOL (primary readout)")
layer_plot(res_last, "step4_layers_lasttoken.png",
           "5-pair activation differences — LAST-TOKEN (control readout)")


# ---- 图 2: 中间带 cosine 柱状,mean-pool vs last-token 并排 ----
def midband_cos(res):
    return {name: float(np.mean([res[name]["cos"][k] for k in MID])) for name in PAIRS}


mm, ml = midband_cos(res_mean), midband_cos(res_last)
names = list(PAIRS); x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - w/2, [mm[n] for n in names], w, label="mean-pool")
ax.bar(x + w/2, [ml[n] for n in names], w, label="last-token")
ax.set_xticks(x)
ax.set_xticklabels([n.split("(")[0].strip() for n in names], rotation=20, ha="right")
ax.set_ylabel("mean cosine over layers 5–9  (lower = more different)")
ax.set_title("Mid-band cosine per pair — readout sensitivity (mean-pool vs last-token)")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("step4_bar.png", dpi=140, bbox_inches="tight")
print("saved -> step4_bar.png")


# ---- quick read(连续语言 + n=1 caveat)----
print()
print("=" * 104); print("QUICK READ (mean-pool, mid-band 5–9 cosine)"); print("=" * 104)
b_bp = mm["B ↔ B'       (preference)"]
b_bn = mm["B ↔ B_noise  (floor)"]
a_ap = mm["A ↔ A+       (len only)"]
ap_b = mm["A+ ↔ B       (form only)"]
print(f"  B↔B'      (换偏好内容) mid-band cos : {b_bp:.4f}")
print(f"  B↔B_noise (只改措辞,地板) cos      : {b_bn:.4f}")
print(f"  → 差 (floor − B') = {b_bn - b_bp:+.4f}   (>0 表示换偏好比改写更不相似 = 有信号方向)")
print(f"  参照: A↔A+ (长度底) = {a_ap:.4f} | A+↔B (形式) = {ap_b:.4f}")
sep = [res_mean["B ↔ B_noise  (floor)"]["cos"][k] - res_mean["B ↔ B'       (preference)"]["cos"][k]
       for k in range(n_layers)]
print(f"  B' 与地板分得最开的层: layer {int(np.argmax(sep))} (Δcos = {max(sep):+.4f})")
print()
print("解读规则:")
print("  - 用差值/比例描述,不要说『假设成立/不成立』。")
print("  - n=1:每个条件只有一段文本,没有误差棒 → 这是【方向/效应量】,不是显著性。")
print("    若 floor−B' 看着明显,最便宜的下一步:把 B'/B_noise 各做 ~5 个改写实例,")
print("    跑出误差棒再下结论(= optional Phase 0.5,这轮不做)。")
print("  - 对照 step4_bar.png:若 mean-pool 与 last-token 结论方向不同,以 mean-pool 为准,")
print("    并把『读点选择改变了结论』本身记成一个方法论发现。")
print()
print("Limitations:")
print("  L1. B/B'/B_noise 都是『对话中陈述』,不是真实经历(继承 step3)。")
print("  L2. A+ 与 B 偏好密度未控(继承 step3)。")
print("  L4 (本轮已解决). B 与 B' 同名(都 Alex)→ 名字 token 差异 confound 已消除。")
print("  L5 (本轮处理). last-token 受结尾 Better 主导 → 改 mean-pool 为主读点;last-token 仅对照。")
print("  L6. n=1,无误差棒(见上)。")
print("  L7. B' 与 B 同名但偏好相互矛盾(语义上不是同一个连贯的人)——这是【刻意的】,")
print("      为把『偏好内容』与『名字 token』分离;不是叙事 bug。")
print()
print("本轮是 Phase 0(distance 框架)。不要在这里训 linear probe —— 那是 Phase 1,另开 spec。")
```

---

## §3 — 读数硬约束(再强调一遍,别走样)

1. **headline = 中间带 layer 5–9 的 cosine**(两头都脏:0/1 是位置,11/12 是各向异性 + 范数)。
2. **mean-pool 为主,last-token 为对照**。`step4_bar.png` 直接并排两种读点——这本身是个干净的方法论小发现(读点是 design decision,不是默认设定)。
3. **核心比较 = `B↔B'` vs `B↔B_noise`**,不是 `B↔B'` 的绝对值,也不是拿 `A+↔B` 当尺子。
4. **不**用末层 cosine 比内容;**不**把深层 L2 压成单一 summary 数。
5. 全程**连续语言**(差值 / 比例 / 哪层峰值),不要二元判定。

---

## §4 — Limitations(写进脚本输出,已在 §2 代码里)

- **L1**:B/B'/B_noise 都是「对话中陈述」,非真实经历(继承)。
- **L2**:A+ 与 B 偏好密度未控(继承)。
- **L4(已解决)**:同名消除了名字 token confound。
- **L5(已处理)**:mean-pool 取代 last-token 为主读点,化解结尾 Better 主导。
- **L6(新)**:n=1,无误差棒,结论是方向 / 效应量而非显著性;便宜跟进 = 各 ~5 个改写实例(Phase 0.5,这轮不做)。
- **L7(新)**:B' 与 B 同名但偏好矛盾,是刻意为分离「偏好内容 vs 名字 token」,不是 bug。

---

## §5 — 边界 & 文件命名

**边界:** 这一整轮(含 B'/B_noise)是 **Phase 0 = distance 框架**。distance 有硬天花板,它分不开「模型表示了这个人 / 表示了但没用 / 没表示」三种。真正逼近假设的下一刀(readout 从距离换成**可解码性 / 行为**:linear probe 或 behavioral readout)是 **Phase 1,另开 spec,这轮不做**。

```
texture_experiment/
├── step1_microscope.py
├── step2_contexts.py            (已有;可选:print 块包进 __main__)
├── step3_compare.py             (已有)
├── step3_layer_vs_difference.png
├── step4_contexts.py            ← 新建 (B_prime, B_noise) — §1
├── step4_compare.py             ← 新建 (五对距离) — §2
├── step4_layers.png             ← mean-pool 逐层(主)
├── step4_layers_lasttoken.png   ← last-token 逐层(对照)
└── step4_bar.png                ← 中间带 cosine,两种读点并排
```

**带回原对话的东西:** ① `B↔B'` 与 `B↔B_noise` 的中间带数字 + 差值;② 五对曲线长什么样(尤其 `B↔B'` 与地板分得最开的层);③ mean-pool vs last-token 差多少;④ 任何 surprise;⑤ 过程体感(停不下来 / 想逃)。
