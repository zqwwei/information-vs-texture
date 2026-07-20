"""
STEP 3 — Activation 对比

设计:
- 三段 context (A / A+ / B) 各跑一遍 GPT-2 (output_hidden_states=True)
- 每段取「最后一个 token」在 13 层中每一层的 activation 向量 (768 维)
- 对每一层算两段之间的 cosine similarity 和 L2 距离
- 对比三对:
    A  ↔ B   : 形式差异 + 长度差异 (混在一起)
    A  ↔ A+  : 纯长度差异 (对照组)
    A+ ↔ B   : 形式差异 (长度已对齐,最干净的「形式」信号)
- 输出一条「层 vs 差异」曲线 + 数字表

记录在结果里的两条已知局限:
1. B 是「在对话中陈述」而非纯经历(用户与你模拟的是文本对话,不是实际行为序列)
2. A+ / B 的偏好密度未控:A+ 末尾的偏好句密度高于 B 中分散嵌入的偏好密度
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

from step2_contexts import A, A_plus, B  # 复用 STEP 2 的草稿

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()


def last_token_per_layer(text: str) -> list[torch.Tensor]:
    """返回 13 个 768 维向量,每层一个,都是该 context 最后一个 token 的 activation"""
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [hs[0, -1, :].clone() for hs in out.hidden_states]  # 13 layers


print("Encoding A, A+, B through GPT-2...")
vecs_A     = last_token_per_layer(A)
vecs_Aplus = last_token_per_layer(A_plus)
vecs_B     = last_token_per_layer(B)
n_layers = len(vecs_A)
print(f"  Got {n_layers} layers per context.\n")


def compare(v1, v2):
    cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    l2  = (v1 - v2).norm().item()
    return cos, l2


pairs = {
    "A ↔ B   (form + length)": (vecs_A, vecs_B),
    "A ↔ A+  (length only)":  (vecs_A, vecs_Aplus),
    "A+ ↔ B  (form only)":    (vecs_Aplus, vecs_B),
}

results = {}  # name -> {"cos": [...], "l2": [...]}
for name, (v_left, v_right) in pairs.items():
    cos_per_layer, l2_per_layer = [], []
    for layer in range(n_layers):
        c, d = compare(v_left[layer], v_right[layer])
        cos_per_layer.append(c)
        l2_per_layer.append(d)
    results[name] = {"cos": cos_per_layer, "l2": l2_per_layer}


# ---------- 打印数字表 ----------
print("=" * 92)
print("Cosine similarity per layer (higher = more similar):")
print("=" * 92)
header = "layer | " + " | ".join(f"{name[:24]:>24}" for name in pairs)
print(header)
print("-" * len(header))
for layer in range(n_layers):
    cells = " | ".join(f"{results[name]['cos'][layer]:>24.4f}" for name in pairs)
    print(f"{layer:>5} | {cells}")

print()
print("=" * 92)
print("L2 distance per layer (higher = more different):")
print("=" * 92)
print(header)
print("-" * len(header))
for layer in range(n_layers):
    cells = " | ".join(f"{results[name]['l2'][layer]:>24.4f}" for name in pairs)
    print(f"{layer:>5} | {cells}")


# ---------- 画曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
layers = list(range(n_layers))

colors = {
    "A ↔ B   (form + length)": "tab:red",
    "A ↔ A+  (length only)":  "tab:blue",
    "A+ ↔ B  (form only)":    "tab:green",
}

for name in pairs:
    axes[0].plot(layers, results[name]["cos"], marker="o", label=name, color=colors[name])
    axes[1].plot(layers, results[name]["l2"],  marker="o", label=name, color=colors[name])

axes[0].set_title("Cosine similarity by layer\n(higher = more similar)")
axes[0].set_xlabel("Layer  (0 = embedding, 1–12 = transformer blocks)")
axes[0].set_ylabel("Cosine similarity")
axes[0].set_xticks(layers)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", fontsize=9)

axes[1].set_title("L2 distance by layer\n(higher = more different)")
axes[1].set_xlabel("Layer  (0 = embedding, 1–12 = transformer blocks)")
axes[1].set_ylabel("L2 distance")
axes[1].set_xticks(layers)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best", fontsize=9)

fig.suptitle(
    "GPT-2 last-token activation differences\n"
    "A: statement form  |  A+: statement + neutral padding (length-matched to B)  |  B: dialogue/experience form",
    fontsize=11,
)
fig.tight_layout()
out_png = "step3_layer_vs_difference.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nSaved plot -> {out_png}")


# ---------- 简短解读提示 ----------
print()
print("=" * 92)
print("Quick read:")
print("=" * 92)
# 关键比较:A↔B 是否显著大于 A↔A+
diff_form_plus_len = np.array(results["A ↔ B   (form + length)"]["l2"])
diff_len_only      = np.array(results["A ↔ A+  (length only)"]["l2"])
diff_form_only     = np.array(results["A+ ↔ B  (form only)"]["l2"])

print(f"  Avg L2  A↔B    (form+len) : {diff_form_plus_len.mean():.3f}")
print(f"  Avg L2  A↔A+   (len only) : {diff_len_only.mean():.3f}")
print(f"  Avg L2  A+↔B   (form only): {diff_form_only.mean():.3f}")
print()
print(f"  Layer with biggest A↔B   L2 : layer {int(diff_form_plus_len.argmax())} ({diff_form_plus_len.max():.3f})")
print(f"  Layer with biggest A+↔B  L2 : layer {int(diff_form_only.argmax())} ({diff_form_only.max():.3f})")
print()
print("Interpretation guide:")
print("  - If  A↔B  >>  A↔A+, the extra is form (experience-vs-statement) contribution.")
print("  - A+↔B is the cleanest 'form-only' signal (length controlled out).")
print("  - Watch *where* (which layer) the difference peaks — early = lexical, late = semantic.")
print()
print("Known limitations (per user):")
print("  L1. B is 'stated within dialogue', not pure lived experience.")
print("  L2. Preference density differs between A+ (concentrated at end) and B (distributed).")
