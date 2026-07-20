# Texture Experiment — 实验索引

**假设:** LLM 对「只有信息(陈述)」vs「信息+真实交互历史(经历)」的内部表示不同。
这个差异如果存在,就是"质地"(对一个人的理解)的计算对应物。
显微镜: GPT-2 small(117M)→ Qwen2.5-3B-Instruct(3B),本地 MPS 运行。

---

## STEP 1 — 显微镜开机

**问题:** 能否加载 GPT-2、抽出中间层 activation、看到正确形状?

| 文件 | 说明 |
|---|---|
| `step1_microscope.py` | 加载模型,输出每层 hidden state 形状 |

**结果:** 13 层(1 embedding + 12 transformer),每层形状 `[1, n_tokens, 768]`。通过。

---

## STEP 2 — 构造 A/B/A+ context

**问题:** 如何设计「控制变量干净」的三段文本?

| 文件 | 说明 |
|---|---|
| `step2_contexts.py` | A / A+ / B 三段文本定义 + token 数核对 |

**设计:**

| 段 | 内容 | Token 数 |
|---|---|---|
| **A** | Alex 三条偏好,纯陈述形式 | 34 |
| **A+** | 同 A + 中性传记填充(对照,控制长度) | 283 |
| **B** | 同三条偏好,以对话+accept/reject 信号演示 | 272 |

**三条偏好(正交轴):** 简短回答(length)/ 结论先行(order)/ 犹豫时要被推一把(decisiveness)

**控制变量:** A 是 B 的压缩;B 中纠正句复用 A 的核心词;A+ / B 长度比 1.04。

---

## STEP 3 — 激活对比(A / A+ / B)

**问题:** 「陈述 vs 经历」的形式差异,在 activation 层面存不存在?

| 文件 | 说明 |
|---|---|
| `step3_compare.py` | 三段各跑 GPT-2,逐层取 last-token,算 cosine + L2 |
| `results/step3_layer_vs_difference.png` | 层 vs 差异曲线 |

**核心结果(last-token, cosine):**

| 对比 | avg L2 | mid-band cos |
|---|---|---|
| A ↔ A+(纯长度) | 39.6 | 0.85 |
| **A+ ↔ B(纯形式)** | **52.3** | **0.95** |

形式差异 > 长度差异;layer 8–11 最大;末层塌回(各向异性)。**质地假设初步方向性支持。**

---

## STEP 4 — Phase 0 收口(B' + B_noise)

**问题:** A+↔B 的差异里,能否把「对话结构」和「偏好内容」分开?

| 文件 | 说明 |
|---|---|
| `step4_contexts.py` / `step4_compare.py` | 五对距离,mean-pool(主)+ last-token(对照) |
| `results/step4_layers.png` / `step4_bar.png` | 逐层曲线 + 中间带柱状 |

**核心结果:** B↔B'(换偏好) vs B↔B_noise(改措辞) 差值 Δcos=+0.0016 → 贴着地板。
**最大 surprise:** last-token 读点符号反转(L5 防住)。Distance 框架到顶。

---

## STEP 5 — Phase 1(linear probe,GPT-2)

**问题:** 偏好信息是否「线性在场」?激活 probe 能否超过词面 lexical baseline?

| 文件 | 说明 |
|---|---|
| `step5_dataset.py` / `step5_probe.py` | 64 A-form + 64 B-form;逐层 logistic probe + GroupKFold |
| `data/step5_data.json` | 96 instances(3 轴 × 8 profiles × 8 phrasings) |
| `results/step5_decode_*.png` | 三轴解码曲线 |

**结果:** B-form 激活与 lexical baseline 同时打满 1.000 → **天花板问题**:纠正句含关键词,词面已完美可解,测不出「激活是否超越词面」。ORDER 轴 layer 0→5 爬升是唯一有意义的信号。

**教训:** 这不是 L10(显微镜太小),是设计缺陷 → 改用干净演示(见 STEP 5b)。

---

## STEP 5b — Phase 1 重做(lexical baseline 归零)

**问题:** 重新设计让 lexical baseline ≈ 0.5,使「激活是否超越词面」成为可测问题。

| 文件 | 说明 |
|---|---|
| `step5b_dataset.py` | ORDER 轴专项:同词异序,generic correction,group by scenario |
| `data/step5b_data.json` | 96 instances(12 scenarios × 2 labels × 4 variants) |
| `step5b_probe.py` | 同 Phase 1,但以 b_final_span 为读点 |
| `results/step5b_decode_order.png` | ORDER 轴解码曲线(含 lexical 基线) |

**关键设计:** 两个 label 的 B-form 含**完全相同的词**,只有被 accept 的顺序不同 → CountVectorizer 在构造上无法区分 → B-form lexical = **0.500**。

**核心结果(GPT-2, GroupKFold-5 by scenario):**

| 读点 | layer 0 | layer 8(峰值) | B-form lexical | shuffle |
|---|---|---|---|---|
| mean-over-final-answer | 0.667 | **0.971 ± 0.036** | **0.500** | 0.481 |

- 激活 probe 从 layer 0(0.67)爬升到 layer 8(0.97),**高于 lexical 基线 4.9 个标准差**
- **结论:GPT-2 从 answer tokens 的上下文激活里算出了词面无法解码的结构信息 → 质地假设在 decodability 层面得到支持。**
- L9: 只测线性可解码;L8: decodability ≠ use(是否被用于生成,需 Phase 2)

---

## STEP 6 — Phase 2(causal patching,Qwen2.5-3B-Instruct)

**问题:** 把 claim 从「在场」升到「使用」——外科手术式改写 order 方向分量,行为是否随之翻转?

**模型:** Qwen2.5-3B-Instruct(3B,MPS 本地运行)

### 6a — 两个前置闸门(v1→v3 三轮修复)

| 文件 | 说明 |
|---|---|
| `step6_data_chat.py` | B-form → chat template + held-out 行为测试对 |
| `step6a_handles.py` | Gate A(Qwen 上 decodability)+ Gate B v1(system prompt,已废) |
| `step6a_v2.py` | Gate B v2:多轮 chat(B-fix-1)+配对测量(B-fix-2)+B0/B1 梯子 |
| `step6a_v3.py` | Gate B v3:聚类 n=12(Audit-1)+干净正向演示(Audit-2) |
| `results/step6_handles.png` / `step6_handles_v2.png` / `step6_handles_v3.png` | 各版行为把手分布 |

**修复历史与教训:**
- **v1 失败(system prompt):** B-form 放 system 整段,instruct 模型不遵从 → 伪负
- **v2 修正(多轮 chat + 聚类):** B1_corr t=−2.49 → 聚合后 t=−1.47(null);发现 wrong-first primacy confound
- **v3 干净演示:** 去掉 wrong-first 纠正结构,assistant 直接用偏好顺序作答 → B1_clean **t=+6.27**

**最终 Gate 结果(v3,n=12 聚类):**

| 闸门 | 结果 |
|---|---|
| **Gate A** | Qwen2.5 上 L\*=8,acc=1.000(L8→L29 持续满分) ✓ |
| **B0** 显式指令 | Δ=+0.454,t=+2.18 ✓ |
| **B1_clean** 正向演示 | Δ=+0.436,t=+6.27 ✓ |
| **B1_corr** 纠正演示(聚类) | Δ=−0.100,t=−1.47 → null |

**Δ_nat = +0.4363**(B1_clean 自然行为摆幅,Phase 2 因果阈值分母)

### 6d — Interchange Patching（clean Phase 2，最终结果）

| 文件 | 说明 |
|---|---|
| `step6d_interchange.py` | d̂(diff-of-means,L\*=8)+ interchange patch(surgical/whole)+ multi-layer + 全套控制 |
| `results/step6_interchange.png` | 各场景 Δ 分布图 |

**方向质量(clean-demo,span-mean readpoint):** decodability=1.000，cosine(d̂,probe)=0.881 ✓

**因果结果(n=12 held-out scenarios):**

| 条件 | Δ | SE | t | Recovery R |
|---|---|---|---|---|
| Surgical cross (L\*=8) | −0.0004 | 0.0015 | −0.30 | **0.001** |
| Whole cross (L\*=8) | −0.0200 | 0.0170 | −1.18 | 0.046 |
| Same-ctrl（期望≈0） | +0.0012 | 0.0019 | +0.67 | — |
| Surgical reverse（期望+） | +0.0004 | 0.0015 | +0.27 | — |
| Surgical neutral | +0.0038 | 0.0032 | +1.18 | — |
| **Multi-layer surgical (L 6–12)** | −0.0024 | 0.0016 | −1.51 | 0.005 |
| **Multi-layer whole (L 6–12)** | −0.0340 | 0.0158 | −2.16 (p≈0.054) | 0.078 |

**注意：上方数字为修正后结果（修正了 output_hidden_states off-by-one bug）。**

**结论：decodability-vs-causal-use dissociation（干净结论，硬停）。**
- Surgical null 全程（L\*=8: t=−0.30；L6–12: t=−1.51）。Surgical patch 将 d̂ 投影从 −3.05 推至 −1.26（全摆幅），order_score 不动 → d̂ 方向因果 epiphenomenal。
- Multi-layer whole t=−2.16，n=12，df=11，临界值 t=2.201（p<0.05 双尾）→ **未达显著**（p≈0.054）。加之 7 个条件中只有此一项边界值，不作结论，仅报告为 suggestive trend。
- 同向控制、反向、中性组全部 ≈0 → 控制干净。
- **Honest bound**：interchange 只能作用于 Q'+tail token，demo 位置因 token 顺序不同无法对齐交换。结论限定为"d̂ 在 Q' 位置不是因果通道"，不等于"order 从不被因果使用"。
> "ORDER is 100% linearly decodable at L8, but the decodable direction is causally epiphenomenal at the tested (Q') positions."

---

### 6bc — 因果 patching(方向 + 干预 + 控制)【已被 6d 取代，结果不可信】

| 文件 | 说明 |
|---|---|
| `step6bc_causal.py` | d̂(diff-of-means,L\*=8)+ 加性 patch + 投影置换 + 全套控制 |
| `results/step6_causal.png` | dose-response + layer sweep + neutral vs persona |
| `data/step6_test_items.json` / `step6_test_items_annotated.json` | held-out 行为测试对 |

**方向质量(clean-demo,span-mean readpoint):** decodability=1.000,cosine(d̂,probe_weight)=0.893 ✓

**因果结果(n=12 held-out scenarios):**

| α | Δorder_score | 连贯项 |
|---|---|---|
| 0 | 0 | 12/12 |
| **+0.5** | **−0.220 ± 0.161** | **12/12** |
| +1.0 | −0.843 | 2/12 |
| ≥+2 | NaN | 0/12 |

| 控制测试 | 结果 |
|---|---|
| 判据(a)方向特异性 | FAIL(null 在 α=1,true 在 α=0.5 — 不同剂量比较) |
| 判据(b)dose-response ρ | NaN(coherence guard 过滤多数 α) |
| 判据(c)显著且符号对 | PASS(t=−2.64) |
| 消融(投影掉 d̂) | 几乎无效果(necessity ratio=1.01) |
| 投影置换 recovery R | **0.020**(近零) |
| Neutral vs persona(α=+0.5) | Persona:−0.22 vs Neutral:+0.19 → **persona 特异** |
| Layer sweep(L=6,8,10,α=1) | 三层效应相近(无 L\* 特异性) |

**解读:**
- d̂ 在 α=+0.5 有 persona 特异的因果效应(正确方向,12/12 连贯)
- 但 d̂ 不是行为的主要中介:消融无效(R=0.02),任何大扰动都让模型不连贯
- 需要在 α=+0.5 也计算 random null 才能给 criterion(a)公平比较(下一步可选)

**当前限制(L8/L11/L15):**
- L8: decodability ≠ use;Phase 2 测的是「在此 setup/此方向有因果作用」
- L11: 线性方向;非线性中介看不到
- L15: B1_clean 强效应可能是 format-induction(复制最近 assistant 顺序),claim 封顶在「L\*=8 order 方向因果控制输出顺序」,**不说**「模型使用了人物表示」

---

## 文件结构

```
texture_experiment/
├── EXPERIMENT_INDEX.md
├── data/
│   ├── step5_data.json
│   ├── step5b_data.json
│   ├── step6_test_items.json
│   └── step6_test_items_annotated.json
├── results/
│   ├── step3_layer_vs_difference.png
│   ├── step4_bar.png / step4_layers.png / step4_layers_lasttoken.png
│   ├── step5_decode_length/order/decisiveness.png
│   ├── step5b_decode_order.png
│   ├── step6_handles.png / step6_handles_v2.png / step6_handles_v3.png
│   ├── step6_causal.png
│   └── step6_interchange.png
├── specs/
│   ├── handoff_spec_step1_实验.md
│   ├── handoff_spec_step4_draft_v1.md / handoff_spec_step4_phase0.md
│   ├── handoff_spec_step5_draft_v1.md / handoff_spec_step5_phase1.md
│   └── handoff_spec_step6_phase2_v1~v4.md
├── step1_microscope.py ~ step5b_probe.py   (GPT-2 实验)
├── step6_data_chat.py
├── step6a_handles.py / step6a_v2.py / step6a_v3.py
├── step6bc_causal.py
└── step6d_interchange.py
```

---

## 实验进展一览

| 阶段 | 模型 | 核心发现 | 状态 |
|---|---|---|---|
| STEP 1 | GPT-2 | 显微镜正常工作 | ✓ |
| STEP 2–3 | GPT-2 | 形式差异 > 长度差异;layer 8–11 信号最强 | ✓ |
| STEP 4(Phase 0) | GPT-2 | Distance 到顶;读点选择是真陷阱 | ✓ |
| STEP 5(Phase 1) | GPT-2 | Lexical 天花板=1.000;设计缺陷,不是显微镜太小 | ✓ |
| STEP 5b(Phase 1 重做) | GPT-2 | Lexical=0.500;激活 probe layer 8 达 0.971(+4.9σ above lexical) | ✓ |
| STEP 6 Gate A | Qwen2.5-3B | L\*=8,acc=1.000;ORDER 方向强可解码 | ✓ |
| STEP 6 Gate B(v1→v3) | Qwen2.5-3B | B1_clean t=+6.27;wrong-first primacy confound 是关键陷阱 | ✓ |
| STEP 6bc 因果 patching | Qwen2.5-3B | 加性 steering，4个bug，结果不可信 → 被 6d 取代 | ✗ 废弃 |
| **STEP 6d Interchange Patching** | **Qwen2.5-3B** | **Surgical null（全强度干预零效应）；whole trend suggestive (p≈0.054，未显著)；decodability-vs-causal-use dissociation，硬停** | **✓ 最终** |
| Phase 3(可选) | 更大模型 | 机制定位/规模曲线 | 未开始 |
