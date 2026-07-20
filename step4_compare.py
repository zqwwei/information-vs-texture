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
