"""
STEP 6a v2 — Gate B re-run (B0 + B1, three fixes applied)

Gate A already passed: L*=8, acc=1.000 on Qwen2.5-3B-Instruct.

Fixes vs v1:
  B-fix-1  Real multi-turn chat (not system-prompt dump).
           B-form parsed into user/assistant message pairs; Q' appended as
           final user turn.  Instruction-tuned models only respond to roles.
  B-fix-2  Paired measurement: same Q', concl-persona vs reas-persona.
           Δ_pair(Q') = order_score(concl) − order_score(reas) cancels the
           model's own conclusion-first prior (same Q' → prior identical in
           both arms → differences from prior).
  B-fix-3  Diagnostic ladder:
           B0 = explicit system instruction (upper bound: does readout work at all?)
           B1 = real multi-turn demo      (texture condition: what the hypothesis needs)

Decision tree (spec §3):
  B0-pass + B1-pass   → proceed to §4/§5 patching (full texture causal test)
  B0-pass + B1-null   → direction-causal fallback (§3.5); decodability≠use confirmed
  B0-null             → behavioral arm absent in this model; report decodability-only

Outputs: step6_handles_v2.png
"""

import json, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

BASE    = "/Users/williams/Desktop/texture_experiment"
DATA5B  = f"{BASE}/data/step5b_data.json"
OUT_PNG = f"{BASE}/results/step6_handles_v2.png"

SCENARIOS = [
    dict(q="Should I take the new job offer?",
         verdict="Take it.",
         rationale="The growth outweighs the risk."),
    dict(q="Should I negotiate this salary offer?",
         verdict="Negotiate.",
         rationale="The downside of a polite ask is almost nothing."),
    dict(q="Should I go back for a graduate degree?",
         verdict="Only if it unlocks a specific door.",
         rationale="The cost is high and the payoff depends entirely on the field."),
    dict(q="Should I switch careers to something I want more?",
         verdict="Do it.",
         rationale="Regret for staying usually outlasts the cost of moving."),
    dict(q="Should I buy a house or keep renting?",
         verdict="Buy, if you are staying five years.",
         rationale="Past that horizon the equity usually beats renting's flexibility."),
    dict(q="Should I quit my job to start a business?",
         verdict="Do not quit yet.",
         rationale="Validating on the side costs you nothing but time."),
    dict(q="Should I take the promotion into management?",
         verdict="Take it.",
         rationale="You will not know if you like leading until you try."),
    dict(q="Should I take a sabbatical from work?",
         verdict="Take it, if you have savings.",
         rationale="Unaddressed burnout compounds into worse decisions."),
    dict(q="Should I get a dog right now?",
         verdict="Wait.",
         rationale="Your schedule right now would not be fair to the animal."),
    dict(q="Should I learn to drive?",
         verdict="Learn.",
         rationale="The independence pays off even if you rarely drive."),
    dict(q="Should I move in with my partner?",
         verdict="Talk about money first.",
         rationale="Most cohabitation conflict traces back to unspoken money assumptions."),
    dict(q="Should I keep this old car or replace it?",
         verdict="Keep it for now.",
         rationale="Repairs are still cheaper than a year of payments."),
]
N_SC = len(SCENARIOS)

with open(DATA5B) as f:
    dataset = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)
model.eval()
print(f"  Loaded.  Layers={model.config.num_hidden_layers}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_logprob(prompt_str: str, completion_str: str) -> float:
    """Per-token mean log-probability of completion given prompt (teacher-forcing)."""
    prompt_ids = tok(prompt_str, return_tensors="pt").input_ids
    full_ids   = tok(prompt_str + completion_str, return_tensors="pt").input_ids.to(device)
    n_prompt = prompt_ids.shape[1]
    n_compl  = full_ids.shape[1] - n_prompt
    if n_compl <= 0:
        return float("nan")
    with torch.no_grad():
        out = model(full_ids)
    logits   = out.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    labels   = full_ids[0, n_prompt:]
    lp       = log_probs[n_prompt-1 : n_prompt-1+n_compl, :]
    token_lp = lp[range(n_compl), labels]
    return token_lp.mean().item()

def order_score(prompt_str: str, C_concl: str, C_reas: str) -> float:
    return mean_logprob(prompt_str, C_concl) - mean_logprob(prompt_str, C_reas)

def apply_chat(messages, add_gen=True) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_gen
    )

def parse_bform_messages(b_form: str) -> list:
    """B-fix-1: parse '{name}: ...' / 'Assistant: ...' lines into role dicts."""
    messages = []
    for line in b_form.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": line[len("Assistant: "):]})
        else:
            m = re.match(r"^[A-Za-z]+: (.+)$", line)
            if m:
                messages.append({"role": "user", "content": m.group(1)})
    return messages

# ── Build complete pairs (same scenario, variant; both labels present) ─────────
pairs_dict = defaultdict(dict)
for d in dataset:
    pairs_dict[(d["scenario_id"], d["variant"])][d["label"]] = d

pairs = [(k, v[0], v[1]) for k, v in pairs_dict.items()
         if 0 in v and 1 in v]          # (key, d_concl, d_reas)
print(f"Complete pairs: {len(pairs)}  (scenario×variant, both labels present)")

# Each pair shares the SAME held-out scenario h = (scenario_id+1)%N_SC
# Verify consistency
for key, d0, d1 in pairs:
    assert d0["scenario_id"] == d1["scenario_id"]
    assert d0["variant"]     == d1["variant"]


# ── B0 — Explicit system instruction (upper bound) ────────────────────────────
print("\n" + "="*68)
print("B0 — Explicit instruction (upper bound)")
print("="*68)

INSTR_CONCL = "When answering questions, always state your conclusion first, then explain your reasoning."
INSTR_REAS  = "When answering questions, always explain your reasoning first, then give your conclusion."

# B0 only depends on Q' (held-out scenario), not on the persona.
# Compute once per unique held-out sid.
b0_cache = {}    # held_out_sid → {C_concl: score_concl, C_reas: score_reas, ...}

unique_held_outs = set((p+1) % N_SC for (p, _), _, _ in pairs)
for h in sorted(unique_held_outs):
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"
    prompt_concl_instr = apply_chat([
        {"role": "system", "content": INSTR_CONCL},
        {"role": "user",   "content": sc["q"]},
    ])
    prompt_reas_instr = apply_chat([
        {"role": "system", "content": INSTR_REAS},
        {"role": "user",   "content": sc["q"]},
    ])
    b0_cache[h] = {
        "C_concl": C_concl, "C_reas": C_reas,
        "os_concl": order_score(prompt_concl_instr, C_concl, C_reas),
        "os_reas":  order_score(prompt_reas_instr,  C_concl, C_reas),
    }
    print(f"  h={h:2d}  os(concl-instr)={b0_cache[h]['os_concl']:+.4f}  "
          f"os(reas-instr)={b0_cache[h]['os_reas']:+.4f}  "
          f"Δ={b0_cache[h]['os_concl']-b0_cache[h]['os_reas']:+.4f}")

b0_deltas = [b0_cache[h]["os_concl"] - b0_cache[h]["os_reas"]
             for h in sorted(unique_held_outs)]
b0_mean = np.mean(b0_deltas)
b0_se   = np.std(b0_deltas, ddof=1) / np.sqrt(len(b0_deltas))
b0_t    = b0_mean / (b0_se + 1e-9)
B0_PASS = abs(b0_t) >= 2.0 and b0_mean > 0
print(f"\n  B0 Δ_pair: mean={b0_mean:+.4f}  SE={b0_se:.4f}  t={b0_t:+.2f}")
print(f"  GATE B0: {'PASS' if B0_PASS else 'FAIL'}  "
      f"({'instruction drives order_score' if B0_PASS else 'instruction does NOT drive order_score'})")


# ── B1 — Real multi-turn demo (texture condition) ─────────────────────────────
print("\n" + "="*68)
print("B1 — Real multi-turn demo (B-fix-1: parsed user/assistant turns)")
print("="*68)

b1_delta_pairs = []
for idx, (key, d_concl, d_reas) in enumerate(pairs):
    if (idx+1) % 12 == 0:
        print(f"  {idx+1}/{len(pairs)} ...")
    p = key[0]
    h = (p + 1) % N_SC
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"

    # Build multi-turn chat for each persona
    def persona_prompt(d):
        msgs = parse_bform_messages(d["b_form"])
        # Merge the trailing 'Better.'/'Got it.' user turn with Q'
        # (avoids consecutive user messages at template boundary)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = msgs[-1]["content"] + "\n" + sc["q"]
        else:
            msgs.append({"role": "user", "content": sc["q"]})
        return apply_chat(msgs)

    prompt_concl = persona_prompt(d_concl)
    prompt_reas  = persona_prompt(d_reas)

    os_concl = order_score(prompt_concl, C_concl, C_reas)
    os_reas  = order_score(prompt_reas,  C_concl, C_reas)
    delta = os_concl - os_reas
    b1_delta_pairs.append({
        "held_out": h, "persona_sid": p, "variant": key[1],
        "os_concl": os_concl, "os_reas": os_reas, "delta": delta
    })

b1_deltas = [x["delta"] for x in b1_delta_pairs]
b1_mean = np.mean(b1_deltas)
b1_se   = np.std(b1_deltas, ddof=1) / np.sqrt(len(b1_deltas))
b1_t    = b1_mean / (b1_se + 1e-9)
B1_PASS = abs(b1_t) >= 2.0 and b1_mean > 0
print(f"\n  B1 Δ_pair: mean={b1_mean:+.4f}  SE={b1_se:.4f}  t={b1_t:+.2f}  n={len(b1_deltas)}")
print(f"  GATE B1: {'PASS' if B1_PASS else 'FAIL'}  "
      f"({'demo drives order_score' if B1_PASS else 'demo does NOT drive order_score'})")


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(len(b0_deltas)), sorted(b0_deltas), color="tab:blue", alpha=0.7)
axes[0].axhline(0, color="black", lw=1)
axes[0].axhline(b0_mean, color="tab:red", lw=2, ls="--",
                label=f"mean={b0_mean:+.4f}  t={b0_t:.2f}")
axes[0].set_xlabel("Held-out scenario")
axes[0].set_ylabel("Δ_pair  (os_concl-instr − os_reas-instr)")
axes[0].set_title(f"B0 — Explicit instruction\n{'PASS' if B0_PASS else 'FAIL'}  "
                  f"mean={b0_mean:+.4f}  SE={b0_se:.4f}")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].hist(b1_deltas, bins=12, color="tab:orange", alpha=0.7)
axes[1].axvline(0, color="black", lw=1)
axes[1].axvline(b1_mean, color="tab:red", lw=2, ls="--",
                label=f"mean={b1_mean:+.4f}  t={b1_t:.2f}")
axes[1].set_xlabel("Δ_pair  (os_concl-demo − os_reas-demo)")
axes[1].set_ylabel("count")
axes[1].set_title(f"B1 — Real multi-turn demo (texture)\n{'PASS' if B1_PASS else 'FAIL'}  "
                  f"mean={b1_mean:+.4f}  SE={b1_se:.4f}")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

fig.suptitle("Step 6a v2 — Gate B (B0/B1 paired Δ_pair, Qwen2.5-3B-Instruct)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")

# ── Summary + decision tree ───────────────────────────────────────────────────
print("\n" + "="*68)
print("SUMMARY — Gate B v2")
print("="*68)
print(f"  B0 (explicit instr):  {'PASS' if B0_PASS else 'FAIL'}  "
      f"Δ={b0_mean:+.4f}  SE={b0_se:.4f}  t={b0_t:+.2f}")
print(f"  B1 (multi-turn demo): {'PASS' if B1_PASS else 'FAIL'}  "
      f"Δ={b1_mean:+.4f}  SE={b1_se:.4f}  t={b1_t:+.2f}")

print("\nDecision tree (spec §3):")
if B0_PASS and B1_PASS:
    print("  B0✓ B1✓ → Both gates pass.")
    print("  Next: step6b/c — full patching + causal controls (texture causal test).")
    print(f"  Δ_nat = {b1_mean:+.4f} (denominator for recovery R in step6c)")
elif B0_PASS and not B1_PASS:
    print("  B0✓ B1✗ → Real finding: order perfectly decodable + explicit instruction")
    print("  drives behavior, but single-round demo does NOT drive generation.")
    print("  = decodability ≠ use (L8 confirmed on generative side).")
    print("  Next (§3.5): direction-causal fallback — patch L*=8 directly,")
    print("  test if the activation direction causally controls output order")
    print("  (separate from 'does demo drive behavior').")
    print(f"  Note: no Δ_nat denominator (B1≈0); report raw Δorder_score + controls.")
else:
    print("  B0✗ → Behavioral arm absent in Qwen2.5-3B.")
    print("  Order_score readout cannot detect instruction-level order preference.")
    print("  Possible causes: model too small, format mismatch, prior too dominant.")
    print("  Options: (a) report decodability-only result; (b) retry with stronger model.")

print("\nLimitations:")
print("  L12: B-form is a single-scenario single-round demo; persona is minimal.")
print("  L13: model prior (conclusion-first) may dominate weak persona signal.")
print("  L14: Qwen2.5-3B instruct; results may differ at larger scale.")
