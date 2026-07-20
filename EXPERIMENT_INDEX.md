# Texture Experiment — Step-by-Step Log

**Hypothesis:** An LLM forms different internal representations for the same information depending on whether it arrives as a statement ("told") or through real interaction history ("shown"). That difference, if it exists, is the computational correlate of *texture* — the depth of understanding a model builds about a person through accumulated interaction.

Microscope: GPT-2 small (117M) → Qwen2.5-3B-Instruct (3B), running locally on Apple MPS.

---

## STEP 1 — Microscope check

**Question:** Can we load GPT-2, extract intermediate activations, and verify the correct shapes?

| File | Description |
|---|---|
| `step1_microscope.py` | Load model, print hidden state shapes per layer |

**Result:** 13 layers (1 embedding + 12 transformer blocks), each with shape `[1, n_tokens, 768]`. Pass.

---

## STEP 2 — Build A / B / A+ contexts

**Question:** How do we design three contexts that isolate variables cleanly?

| File | Description |
|---|---|
| `step2_contexts.py` | Define A / A+ / B with token counts |

**Design:**

| Context | Content | Tokens |
|---|---|---|
| **A** | Three Alex preferences, statement form | 34 |
| **A+** | Same as A + neutral biographical filler (length control) | 283 |
| **B** | Same three preferences, demonstrated through six dialogue rounds with accept/reject signals | 272 |

**Three preference axes (orthogonal):** short answers (length) / conclusion first (order) / push toward a choice when hesitating (decisiveness)

**Controls:** A is a compressed form of B; B's correction sentences reuse A's core keywords; A+/B length ratio is 1.04.

---

## STEP 3 — Activation comparison (A / A+ / B)

**Question:** Does the form difference (statement vs interaction) show up in activations?

| File | Description |
|---|---|
| `step3_compare.py` | Run all three through GPT-2, extract last-token per layer, compute cosine + L2 |
| `results/step3_layer_vs_difference.png` | Layer vs. difference curve |

**Core results (last-token, cosine):**

| Pair | avg L2 | mid-band cos |
|---|---|---|
| A ↔ A+ (length only) | 39.6 | 0.85 |
| **A+ ↔ B (form only)** | **52.3** | **0.95** |

Form difference > length difference; signal peaks at layers 8–11; collapses at the last layer (anisotropy). **Initial directional support for the texture hypothesis.**

---

## STEP 4 — Phase 0 wrap-up (B' + B_noise)

**Question:** Can we separate "dialogue structure" from "preference content" within A+↔B?

| File | Description |
|---|---|
| `step4_contexts.py` / `step4_compare.py` | Five context pairs, mean-pool (primary) + last-token (control) |
| `results/step4_layers.png` / `step4_bar.png` | Layer curves + mid-band bar chart |

**Core result:** B↔B' (swap preferences) vs. B↔B_noise (rephrase same preferences): Δcos = +0.0016 — at the noise floor.

**Key surprise:** Last-token readout reversed the sign of one comparison (caught at L5). Distance framework has reached its ceiling.

---

## STEP 5 — Phase 1 (linear probe, GPT-2)

**Question:** Is preference information *linearly present* in the activations? Can a probe beat the lexical baseline?

| File | Description |
|---|---|
| `step5_dataset.py` / `step5_probe.py` | 64 A-form + 64 B-form; layer-by-layer logistic probe + GroupKFold |
| `data/step5_data.json` | 96 instances (3 axes × 8 profiles × 8 phrasings) |
| `results/step5_decode_*.png` | Decode curves per axis |

**Result:** Both B-form activations and the lexical baseline hit 1.000 simultaneously → **ceiling problem**: correction sentences contain the preference keyword, so word frequency already perfectly decodes the label. The ORDER axis's L0→L5 rise is the only informative signal.

**Lesson:** This is a design flaw, not a microscope limitation → rebuild with clean demonstrations (see STEP 5b).

---

## STEP 5b — Phase 1 redo (lexical baseline zeroed)

**Question:** Redesign so that lexical baseline ≈ 0.5, making "does activation beat lexical?" a testable question.

| File | Description |
|---|---|
| `step5b_dataset.py` | ORDER axis only: same words, different accepted order, generic corrections, grouped by scenario |
| `data/step5b_data.json` | 96 instances (12 scenarios × 2 labels × 4 variants) |
| `step5b_probe.py` | Same as Phase 1, readout at b_final_span |
| `results/step5b_decode_order.png` | ORDER decode curve with lexical baseline |

**Key design:** Both labels use **identical tokens** in their B-form contexts; only the *accepted* order differs → CountVectorizer cannot distinguish them by construction → B-form lexical baseline = **0.500**.

**Core results (GPT-2, GroupKFold-5 by scenario):**

| Readout | Layer 0 | Layer 8 (peak) | B-form lexical | Shuffle |
|---|---|---|---|---|
| mean-over-final-answer | 0.667 | **0.971 ± 0.036** | **0.500** | 0.481 |

- Probe rises from L0 (0.67) to L8 (0.97), **4.9 standard deviations above the lexical floor**
- **Conclusion: GPT-2 computes structural order information from context activations that word frequency cannot decode → texture hypothesis supported at the decodability level**
- Caveat: decodability ≠ causal use (whether the representation actually drives generation requires Phase 2)

---

## STEP 6 — Phase 2 (causal patching, Qwen2.5-3B-Instruct)

**Question:** Upgrade the claim from "present" to "used" — can surgical rewriting of the ORDER direction flip behavior?

**Model:** Qwen2.5-3B-Instruct (3B, local MPS)

### 6a — Two prerequisite gates (v1→v3, three iterations)

| File | Description |
|---|---|
| `step6_data_chat.py` | B-form → chat template + held-out behavioral test pairs |
| `step6a_handles.py` | Gate A (Qwen decodability) + Gate B v1 (system prompt, deprecated) |
| `step6a_v2.py` | Gate B v2: multi-turn chat (B-fix-1) + paired measurement (B-fix-2) + B0/B1 ladder |
| `step6a_v3.py` | Gate B v3: cluster n=12 (Audit-1) + clean positive demonstration (Audit-2) |
| `results/step6_handles*.png` | Behavioral handle distributions per version |

**Iteration history:**
- **v1 failed (system prompt):** Putting B-form in the system prompt — instruct model ignores it → false negative from design
- **v2 fixed (multi-turn + clustering):** B1_corr t=−2.49 → after aggregation t=−1.47 (null); discovered wrong-first primacy confound
- **v3 clean demonstration:** Removed the wrong-first correction structure; assistant answers directly in the preferred order → B1_clean **t=+6.27**

**Final gate results (v3, n=12 clusters):**

| Gate | Result |
|---|---|
| **Gate A** | Qwen2.5 L*=8, acc=1.000 (L8→L29 saturated) ✓ |
| **B0** explicit instruction | Δ=+0.454, t=+2.18 ✓ |
| **B1_clean** positive demonstration | Δ=+0.436, t=+6.27 ✓ |
| **B1_corr** correction demonstration (clustered) | Δ=−0.100, t=−1.47 → null |

**Δ_nat = +0.436** (B1_clean natural behavioral swing — denominator for Phase 2 causal recovery R)

### 6d — Interchange Patching (clean Phase 2, final result)

| File | Description |
|---|---|
| `step6d_interchange.py` | d̂ (diff-of-means, L*=8) + interchange patch (surgical/whole) + multi-layer + full controls |
| `results/step6_interchange.png` | Δ distribution across conditions |

**Direction quality (clean demo, span-mean readpoint):** decodability=1.000, cos(d̂, probe)=0.881 ✓

**Causal results (n=12 held-out scenarios):**

| Condition | Δ | SE | t | Recovery R |
|---|---|---|---|---|
| Surgical cross (L*=8) | −0.0004 | 0.0015 | −0.30 | **0.001** |
| Whole cross (L*=8) | −0.0200 | 0.0170 | −1.18 | 0.046 |
| Same-ctrl (expect ≈0) | +0.0012 | 0.0019 | +0.67 | — |
| Surgical reverse (expect +) | +0.0004 | 0.0015 | +0.27 | — |
| Surgical neutral | +0.0038 | 0.0032 | +1.18 | — |
| **Multi-layer surgical (L6–12)** | −0.0024 | 0.0016 | −1.51 | 0.005 |
| **Multi-layer whole (L6–12)** | −0.0340 | 0.0158 | −2.16 (p≈0.054) | 0.078 |

*Numbers above are from the corrected rerun (fixed output_hidden_states off-by-one bug).*

**Conclusion: decodability-vs-causal-use dissociation (clean finding, hard stop).**
- Surgical null throughout (L*=8: t=−0.30; L6–12: t=−1.51). Surgical patch moved d̂ projection from −3.05 to −1.26 (full swing); order_score did not move → d̂ direction is causally epiphenomenal.
- Multi-layer whole t=−2.16, n=12, df=11, critical value t=2.201 (p<0.05 two-tailed) → **not significant** (p≈0.054). Only one of seven conditions at the boundary; reported as suggestive trend only.
- All controls ≈0 → clean intervention.
- **Honest bound:** interchange could only target Q'+tail tokens; demo positions are not token-alignable across contexts. Conclusion scoped to "d̂ at Q' positions is not a causal channel," not "ORDER is never causally used."

> "ORDER is 100% linearly decodable at L8, but the decodable direction is causally epiphenomenal at the tested (Q') positions."

---

### 6bc — Causal patching attempt (additive steering) — **DEPRECATED, results not reliable**

| File | Description |
|---|---|
| `step6bc_causal.py` | d̂ (diff-of-means, L*=8) + additive patch + projection swap + full controls |

This attempt accumulated four implementation bugs, including an off-by-one indexing error in `output_hidden_states` that caused patches to land at the wrong layer. Results are not reliable and are superseded by `step6d_interchange.py`. Included for completeness only.

---

## File Structure

```
texture_experiment/
├── README.md                    ← full writeup
├── EXPERIMENT_INDEX.md          ← this file
├── LICENSE
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
│   ├── step6_causal.png          ← deprecated (6bc)
│   ├── step6_interchange.png     ← final (6d)
│   └── step6d_rerun.log
├── specs/                       ← handoff specs used across experimental iterations
├── step1_microscope.py ~ step5b_probe.py   (GPT-2 experiments)
├── step6_data_chat.py
├── step6a_handles.py / step6a_v2.py / step6a_v3.py
├── step6bc_causal.py            ← deprecated
└── step6d_interchange.py        ← final
```

---

## Progress Summary

| Phase | Model | Key Finding | Status |
|---|---|---|---|
| STEP 1 | GPT-2 | Microscope working | ✓ |
| STEP 2–3 | GPT-2 | Form difference > length difference; signal peaks L8–11 | ✓ |
| STEP 4 (Phase 0) | GPT-2 | Distance at ceiling; readout choice is a real trap | ✓ |
| STEP 5 (Phase 1) | GPT-2 | Lexical ceiling=1.000; design flaw, not microscope limit | ✓ |
| STEP 5b (Phase 1 redo) | GPT-2 | Lexical=0.500; probe L8=0.971 (+4.9σ above floor) | ✓ |
| STEP 6 Gate A | Qwen2.5-3B | L*=8, acc=1.000; ORDER direction strongly decodable | ✓ |
| STEP 6 Gate B (v1→v3) | Qwen2.5-3B | B1_clean t=+6.27; wrong-first primacy confound was key trap | ✓ |
| STEP 6bc causal patching | Qwen2.5-3B | Additive steering, 4 bugs, results unreliable → superseded by 6d | ✗ deprecated |
| **STEP 6d interchange patching** | **Qwen2.5-3B** | **Surgical null; whole trend suggestive (p≈0.054, not significant); decodability-vs-causal-use dissociation, hard stop** | **✓ final** |
| Phase 3 (optional) | Larger models | Mechanism localization / scale curve | Not started |
