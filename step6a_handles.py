"""
STEP 6a — Gate A (decodability on Qwen2.5-3B-Instruct, find L*)
           Gate B (unpatched order_score separation)

Gate A: same Phase-1 probe but on Qwen2.5. Confirms order preference is
        linearly decodable in the new model. Identifies L* (peak layer).

Gate B: without any patching, does the model already show order_score separation
        between conclusion-first and reasons-first personas?
        order_score = mean_logP(C_concl|prompt) - mean_logP(C_reas|prompt)
        (per-token mean to correct for BPE length differences)

Outputs: step6_handles.png, prints L* and gate decisions.
"""

import json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

SEED = 42
np.random.seed(SEED)

BASE    = "/Users/williams/Desktop/texture_experiment"
DATA5B  = f"{BASE}/data/step5b_data.json"
ITEMS   = f"{BASE}/data/step6_test_items.json"
OUT_PNG = f"{BASE}/results/step6_handles.png"

with open(DATA5B) as f:  dataset5b = json.load(f)
with open(ITEMS)  as f:  items = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── Load Qwen2.5-3B-Instruct ─────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID} ...")
qtok = AutoTokenizer.from_pretrained(MODEL_ID)
qmodel = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)
qmodel.eval()
N_LAYERS_Q = qmodel.config.num_hidden_layers
print(f"  Qwen layers: {N_LAYERS_Q}  |  hidden_dim: {qmodel.config.hidden_size}")


# ── GATE A — decodability on Qwen2.5 ─────────────────────────────────────────
print("\n" + "="*70)
print("GATE A — Per-layer decodability probe on Qwen2.5-3B-Instruct")
print("="*70)

def get_qwen_activations(text, char_span=None):
    """
    Returns list of N_LAYERS+1 tensors (768->hidden_size dim each).
    Readpoint: mean over char_span tokens (or all tokens if span is None).
    """
    fast_tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = fast_tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = qmodel(**enc, output_hidden_states=True)
    hs = out.hidden_states  # (N_LAYERS+1) tuples of (1, T, H)

    if char_span is not None:
        s, e = char_span
        mask = torch.tensor(
            [(ts >= s and te <= e and ts < te) for (ts, te) in offsets],
            dtype=torch.bool
        )
        if mask.sum() == 0:
            mask = torch.ones(hs[0].shape[1], dtype=torch.bool)
        vecs = [hs[k][0][mask].float().mean(dim=0).cpu() for k in range(len(hs))]
    else:
        vecs = [hs[k][0].float().mean(dim=0).cpu() for k in range(len(hs))]
    return vecs

print(f"Encoding {len(dataset5b)} B-forms through Qwen2.5 ...")
q_act = []
for i, d in enumerate(dataset5b):
    if (i+1) % 16 == 0: print(f"  {i+1}/{len(dataset5b)} ...")
    vecs = get_qwen_activations(d["b_form"], char_span=tuple(d["b_final_span"]))
    q_act.append(vecs)
print("  Done.")

y      = np.array([d["label"] for d in dataset5b])
groups = np.array([d["scenario_id"] for d in dataset5b])

def probe_layer(X, y, groups):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
    sc = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=groups, scoring="accuracy")
    return sc.mean(), sc.std()

n_layers_total = len(q_act[0])
gate_a_res = []
print("\nPer-layer probe (Qwen2.5, final-answer readpoint, GroupKFold-5 by scenario):")
print(f"{'layer':>6}  {'acc':>7}  {'std':>6}")
for k in range(n_layers_total):
    X = np.stack([q_act[i][k].numpy() for i in range(len(dataset5b))])
    m, s = probe_layer(X, y, groups)
    gate_a_res.append((m, s))
    mid = "*" if 5 <= k <= n_layers_total-5 else " "
    print(f"{k:>6}{mid}  {m:.3f}    {s:.3f}")

# Find L* = layer with peak accuracy (excluding last 2 layers)
search_range = range(1, n_layers_total - 2)
L_star = max(search_range, key=lambda k: gate_a_res[k][0])
peak_acc, peak_std = gate_a_res[L_star]
GATE_A_PASS = peak_acc > 0.65
print(f"\nL* = layer {L_star}  acc = {peak_acc:.3f} +/- {peak_std:.3f}")
print(f"GATE A: {'PASS' if GATE_A_PASS else 'FAIL'}  "
      f"({'decodable on Qwen2.5, proceed' if GATE_A_PASS else 'not decodable, stop'})")

if not GATE_A_PASS:
    print("Gate A failed — stopping.")
    raise SystemExit(1)


# ── GATE B — unpatched order_score separation ─────────────────────────────────
print("\n" + "="*70)
print("GATE B — Unpatched order_score separation")
print(f"  order_score = mean_logP(C_concl|prompt) - mean_logP(C_reas|prompt)")
print("="*70)

def apply_chat(messages):
    """Apply Qwen2.5 chat template, return string."""
    return qtok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def mean_logprob(prompt_str, completion_str):
    """Per-token mean log probability of completion given prompt."""
    # Tokenize prompt and full sequence separately to find completion start
    prompt_ids = qtok(prompt_str, return_tensors="pt").input_ids
    full_ids   = qtok(prompt_str + completion_str, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[1]
    n_compl  = full_ids.shape[1] - n_prompt
    if n_compl <= 0:
        return float("nan")

    full_ids = full_ids.to(device)
    with torch.no_grad():
        out = qmodel(full_ids)
    logits = out.logits[0].float()  # (T, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Completion token log probs: logit at position [n_prompt-1 .. -2] predicting [n_prompt .. -1]
    labels = full_ids[0, n_prompt:]
    lp = log_probs[n_prompt-1 : n_prompt-1+len(labels), :]
    token_lp = lp[range(len(labels)), labels]
    return token_lp.mean().item()

print(f"Computing order_score for {len(items)} items (2 forward passes each) ...")
order_scores = []
for idx, item in enumerate(items):
    if (idx+1) % 16 == 0: print(f"  {idx+1}/{len(items)} ...")
    prompt = apply_chat(item["messages"])
    lp_concl = mean_logprob(prompt, item["C_concl"])
    lp_reas  = mean_logprob(prompt, item["C_reas"])
    os_ = lp_concl - lp_reas
    order_scores.append(os_)
    items[idx]["order_score_unpatched"] = os_

scores = np.array(order_scores)
label0 = scores[[i for i,it in enumerate(items) if it["label"]==0]]
label1 = scores[[i for i,it in enumerate(items) if it["label"]==1]]

delta_nat = label0.mean() - label1.mean()   # ≡ Δ_nat from §5
sep = abs(delta_nat) / (scores.std() + 1e-9)

print(f"\n  conclusion-first (label 0):  mean={label0.mean():.4f}  std={label0.std():.4f}")
print(f"  reasons-first   (label 1):  mean={label1.mean():.4f}  std={label1.std():.4f}")
print(f"  Δ_nat (natural behaviour swing) = {delta_nat:+.4f}")
print(f"  Separation (|Δ_nat| / pooled std) = {sep:.2f}")

GATE_B_PASS = sep > 0.3 and np.sign(delta_nat) == 1.0
print(f"\nGATE B: {'PASS' if GATE_B_PASS else 'FAIL'}  "
      f"({'handle exists, proceed to patching' if GATE_B_PASS else 'handle absent, patching has no target'})")


# ── Figure: unpatched order_score distributions ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: per-layer probe accuracy (Gate A)
xs = list(range(n_layers_total))
means = [gate_a_res[k][0] for k in xs]
stds  = [gate_a_res[k][1] for k in xs]
axes[0].plot(xs, means, "o-", color="tab:blue", lw=2, label="Qwen2.5 probe (final-answer)")
axes[0].fill_between(xs, [m-s for m,s in zip(means,stds)],
                         [m+s for m,s in zip(means,stds)], alpha=0.2, color="tab:blue")
axes[0].axhline(0.5, color="gray", ls="--", lw=1.2, label="chance (0.5)")
axes[0].axvline(L_star, color="tab:red", ls=":", lw=1.5, label=f"L*={L_star} ({peak_acc:.2f})")
axes[0].set_xlabel("Layer"); axes[0].set_ylabel("CV accuracy")
axes[0].set_title(f"Gate A — Decodability on Qwen2.5\nL*={L_star}, acc={peak_acc:.3f}")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Right: unpatched order_score distributions (Gate B)
axes[1].hist(label0, bins=15, alpha=0.6, color="tab:blue",  label=f"conclusion-first (μ={label0.mean():.3f})")
axes[1].hist(label1, bins=15, alpha=0.6, color="tab:orange", label=f"reasons-first (μ={label1.mean():.3f})")
axes[1].axvline(0, color="black", lw=1.2, ls="--")
axes[1].set_xlabel("order_score (logP concl − logP reas)")
axes[1].set_ylabel("count")
axes[1].set_title(f"Gate B — Unpatched order_score separation\nΔ_nat={delta_nat:+.4f}  sep={sep:.2f}σ")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

fig.suptitle("Step 6a: Gates A+B  —  Qwen2.5-3B-Instruct", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")

# Save annotated items (with unpatched order_score) for step6b/c
ITEMS_ANNOT = f"{BASE}/data/step6_test_items_annotated.json"
with open(ITEMS_ANNOT, "w") as f:
    json.dump(items, f, indent=2)
print(f"Saved annotated items -> {ITEMS_ANNOT}")

# Summary for handoff
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Gate A (decodability):  {'PASS' if GATE_A_PASS else 'FAIL'}  L*={L_star}  acc={peak_acc:.3f}")
print(f"  Gate B (behaviour):     {'PASS' if GATE_B_PASS else 'FAIL'}  Δ_nat={delta_nat:+.4f}  sep={sep:.2f}σ")
print(f"  -> {'Both gates pass. Proceed to step6b (direction + patching).' if (GATE_A_PASS and GATE_B_PASS) else 'Gate failed. See output above.'}")
print(f"\n  L* = {L_star}  (use this for diff-of-means direction in step6b)")
print(f"  Δ_nat = {delta_nat:.4f}  (natural behaviour swing, denominator for recovery R in step6c)")
