"""
STEP 6a v3 — Gate B re-run, two audits fixed

Audit-1 (clustering): 4 variants per scenario are pseudo-replicates.
  Cluster: average 4 variant-level Δ_pair per scenario → n=12 independent points.

Audit-2 (primacy confound): wrong-first correction structure may push model
  backwards via primacy (first answer seen). Fix: B1_clean uses consistent
  positive demos (assistant answers directly in preferred order, user: "Perfect.").

Three conditions reported (all with clustered n≈12 statistics):
  B0           explicit system instruction         (upper bound; already n=12)
  B1_corr      correction-based demo, clustered    (v2 B1 recomputed properly)
  B1_clean     positive-only demo, no primacy      (the target B1 condition)

Decision tree (spec §3.6 v3):
  B1_clean > 0 and significant  → handle exists → proceed to §4/§5 patching
  B1_clean ≈ 0                  → decodability ≠ use confirmed → §3.5 fallback
  B1_clean < 0 and significant  → genuine reverse effect → report carefully
"""

import json, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

BASE       = "/Users/williams/Desktop/texture_experiment"
DATA5B     = f"{BASE}/data/step5b_data.json"
ANNOT_V2   = f"{BASE}/data/step6_test_items_annotated.json"
OUT_PNG    = f"{BASE}/results/step6_handles_v3.png"

SCENARIOS = [
    dict(q="Should I take the new job offer?",
         verdict="Take it.", rationale="The growth outweighs the risk."),
    dict(q="Should I negotiate this salary offer?",
         verdict="Negotiate.", rationale="The downside of a polite ask is almost nothing."),
    dict(q="Should I go back for a graduate degree?",
         verdict="Only if it unlocks a specific door.",
         rationale="The cost is high and the payoff depends entirely on the field."),
    dict(q="Should I switch careers to something I want more?",
         verdict="Do it.", rationale="Regret for staying usually outlasts the cost of moving."),
    dict(q="Should I buy a house or keep renting?",
         verdict="Buy, if you are staying five years.",
         rationale="Past that horizon the equity usually beats renting's flexibility."),
    dict(q="Should I quit my job to start a business?",
         verdict="Do not quit yet.",
         rationale="Validating on the side costs you nothing but time."),
    dict(q="Should I take the promotion into management?",
         verdict="Take it.", rationale="You will not know if you like leading until you try."),
    dict(q="Should I take a sabbatical from work?",
         verdict="Take it, if you have savings.",
         rationale="Unaddressed burnout compounds into worse decisions."),
    dict(q="Should I get a dog right now?",
         verdict="Wait.", rationale="Your schedule right now would not be fair to the animal."),
    dict(q="Should I learn to drive?",
         verdict="Learn.", rationale="The independence pays off even if you rarely drive."),
    dict(q="Should I move in with my partner?",
         verdict="Talk about money first.",
         rationale="Most cohabitation conflict traces back to unspoken money assumptions."),
    dict(q="Should I keep this old car or replace it?",
         verdict="Keep it for now.", rationale="Repairs are still cheaper than a year of payments."),
]
N_SC = len(SCENARIOS)

with open(DATA5B)   as f: dataset = json.load(f)
with open(ANNOT_V2) as f: v2_items = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto")
model.eval()
print(f"  Loaded.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_logprob(prompt_str, completion_str):
    prompt_ids = tok(prompt_str, return_tensors="pt").input_ids
    full_ids   = tok(prompt_str + completion_str, return_tensors="pt").input_ids.to(device)
    n_p = prompt_ids.shape[1]
    n_c = full_ids.shape[1] - n_p
    if n_c <= 0: return float("nan")
    with torch.no_grad():
        out = model(full_ids)
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    labels = full_ids[0, n_p:]
    return lp[n_p-1:n_p-1+n_c, :][range(n_c), labels].mean().item()

def order_score(prompt_str, C_concl, C_reas):
    return mean_logprob(prompt_str, C_concl) - mean_logprob(prompt_str, C_reas)

def chat(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def cluster_deltas(raw_deltas_by_key):
    """Average variants within each scenario, return list of scenario-level means."""
    by_scenario = defaultdict(list)
    for (p, v), delta in raw_deltas_by_key.items():
        by_scenario[p].append(delta)
    return [np.mean(vals) for vals in by_scenario.values()]

def report(label, deltas_n12):
    arr = np.array(deltas_n12)
    m, se = arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))
    t = m / (se + 1e-9)
    sig = abs(t) >= 2.0
    direction = "+" if m > 0 else "-"
    status = ("PASS" if (sig and m > 0) else
              "FAIL(neg)" if (sig and m < 0) else "null")
    print(f"  {label:<20}  Δ={m:+.4f}  SE={se:.4f}  t={t:+.2f}  n={len(arr)}  → {status}")
    return m, se, t, arr


# ── B0 — re-cluster from v2 annotated items ──────────────────────────────────
print("=" * 68)
print("B0 — Explicit instruction (re-read from v2; already n=12 by held-out)")
print("=" * 68)
INSTR_CONCL = "When answering questions, always state your conclusion first, then explain your reasoning."
INSTR_REAS  = "When answering questions, always explain your reasoning first, then give your conclusion."

# B0 is per held-out scenario; recompute fresh (fast: n=12 × 4 passes)
b0_deltas_by_h = {}
for h in range(N_SC):
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"
    p_concl = chat([{"role":"system","content":INSTR_CONCL},{"role":"user","content":sc["q"]}])
    p_reas  = chat([{"role":"system","content":INSTR_REAS}, {"role":"user","content":sc["q"]}])
    b0_deltas_by_h[h] = (order_score(p_concl, C_concl, C_reas)
                       - order_score(p_reas,  C_concl, C_reas))
    print(f"  h={h:2d}  Δ={b0_deltas_by_h[h]:+.4f}")

b0_m, b0_se, b0_t, b0_arr = report("B0(n=12)", list(b0_deltas_by_h.values()))


# ── B1_corr — correction-based demo, properly clustered (n=12) ───────────────
print("\n" + "=" * 68)
print("B1_corr — Correction-based demo, clustered by scenario (n=12)")
print("=" * 68)

def parse_bform(b_form):
    msgs = []
    for line in b_form.strip().split("\n"):
        line = line.strip()
        if not line: continue
        if line.startswith("Assistant: "):
            msgs.append({"role":"assistant","content":line[len("Assistant: "):]})
        else:
            m = re.match(r"^[A-Za-z]+: (.+)$", line)
            if m: msgs.append({"role":"user","content":m.group(1)})
    return msgs

# Build pairs from step5b_data
pairs_dict = defaultdict(dict)
for d in dataset:
    pairs_dict[(d["scenario_id"], d["variant"])][d["label"]] = d
pairs = [(k, v[0], v[1]) for k,v in pairs_dict.items() if 0 in v and 1 in v]

b1c_deltas_by_pv = {}
for idx, (key, d0, d1) in enumerate(pairs):
    if (idx+1) % 12 == 0: print(f"  {idx+1}/{len(pairs)} ...")
    p, v = key
    h  = (p + 1) % N_SC
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"

    def persona_prompt(d):
        msgs = parse_bform(d["b_form"])
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += "\n" + sc["q"]
        else:
            msgs.append({"role":"user","content":sc["q"]})
        return chat(msgs)

    os0 = order_score(persona_prompt(d0), C_concl, C_reas)
    os1 = order_score(persona_prompt(d1), C_concl, C_reas)
    b1c_deltas_by_pv[(p, v)] = os0 - os1

b1c_n12 = cluster_deltas(b1c_deltas_by_pv)
print(f"  Clustered: {len(b1c_n12)} scenario-level points (4 variants averaged per scenario)")
b1c_m, b1c_se, b1c_t, b1c_arr = report("B1_corr(n=12)", b1c_n12)


# ── B1_clean — positive-only demo, no primacy, clustered ─────────────────────
print("\n" + "=" * 68)
print("B1_clean — Positive-only demo (no wrong-first), clustered (n=12)")
print("=" * 68)

# For each (persona scenario p, variant v, held-out h):
#   Show model answering 2 context scenarios in preferred order directly.
#   Context scenarios: (h + 2 + 2*v) % N_SC and (h + 4 + 2*v) % N_SC
#   (vary by variant to get real spread; always ≠ h)
# Accept signal: "Perfect."

def build_clean_demo(label, p, v, h):
    """Build multi-turn clean demo (no primacy): 2 Q&As in preferred order."""
    ctx1 = (h + 2 + 2*v) % N_SC
    ctx2 = (h + 4 + 2*v) % N_SC
    # avoid collision with p or h
    ctx1 = ctx1 if ctx1 not in (p, h) else (ctx1 + 1) % N_SC
    ctx2 = ctx2 if ctx2 not in (p, h, ctx1) else (ctx2 + 2) % N_SC

    msgs = []
    for sc_idx in [ctx1, ctx2]:
        sc = SCENARIOS[sc_idx]
        if label == 0:  # conclusion-first
            answer = f"{sc['verdict']} {sc['rationale']}"
        else:           # reasons-first
            answer = f"{sc['rationale']} {sc['verdict']}"
        msgs.append({"role": "user",      "content": sc["q"]})
        msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user",      "content": "Perfect."})
    return msgs

b1k_deltas_by_pv = {}
for idx, (key, d0, d1) in enumerate(pairs):
    if (idx+1) % 12 == 0: print(f"  {idx+1}/{len(pairs)} ...")
    p, v = key
    h  = (p + 1) % N_SC
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"

    def clean_prompt(label):
        msgs = build_clean_demo(label, p, v, h)
        # Merge last "Perfect." with Q' (avoid consecutive user messages)
        msgs[-1]["content"] += "\n" + sc["q"]
        return chat(msgs)

    os0 = order_score(clean_prompt(0), C_concl, C_reas)
    os1 = order_score(clean_prompt(1), C_concl, C_reas)
    b1k_deltas_by_pv[(p, v)] = os0 - os1

b1k_n12 = cluster_deltas(b1k_deltas_by_pv)
print(f"  Clustered: {len(b1k_n12)} scenario-level points")
b1k_m, b1k_se, b1k_t, b1k_arr = report("B1_clean(n=12)", b1k_n12)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def bar_plot(ax, arr, title, color):
    ax.bar(range(len(arr)), sorted(arr), color=color, alpha=0.7)
    ax.axhline(0,          color="black", lw=1)
    ax.axhline(arr.mean(), color="tab:red", lw=2, ls="--",
               label=f"mean={arr.mean():+.4f}\nt={arr.mean()/(arr.std(ddof=1)/len(arr)**0.5+1e-9):.2f}")
    ax.set_xlabel("Scenario (sorted Δ)"); ax.set_ylabel("Δ_pair"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

bar_plot(axes[0], b0_arr,  f"B0 — Explicit instruction\nmean={b0_m:+.4f}  t={b0_t:.2f}",  "tab:blue")
bar_plot(axes[1], b1c_arr, f"B1_corr — Correction demo (clustered)\nmean={b1c_m:+.4f}  t={b1c_t:.2f}", "tab:orange")
bar_plot(axes[2], b1k_arr, f"B1_clean — Positive demo (no primacy)\nmean={b1k_m:+.4f}  t={b1k_t:.2f}", "tab:green")

fig.suptitle("Step 6a v3 — Gate B, clustered n=12  (Qwen2.5-3B-Instruct)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")


# ── Summary + decision tree ───────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SUMMARY  (all n=12 clustered, honest SE)")
print("=" * 68)
print(f"  B0       (explicit instr):   Δ={b0_m:+.4f}  SE={b0_se:.4f}  t={b0_t:+.2f}")
print(f"  B1_corr  (correction demo):  Δ={b1c_m:+.4f}  SE={b1c_se:.4f}  t={b1c_t:+.2f}  (v2 t=-2.49 was pseudo-rep inflated)")
print(f"  B1_clean (positive demo):    Δ={b1k_m:+.4f}  SE={b1k_se:.4f}  t={b1k_t:+.2f}")

print("\nDecision tree (spec §3.6 v3, using B1_clean):")
if b1k_m > 0 and abs(b1k_t) >= 2.0:
    print(f"  B1_clean PASS  Δ={b1k_m:+.4f}  t={b1k_t:+.2f}")
    print("  Demo drives order_score → behavioral handle exists.")
    print(f"  Δ_nat = {b1k_m:+.4f} (denominator for recovery R in §5).")
    print("  Next: step6b — compute diff-of-means direction at L*=8,")
    print("         implement additive + projection patching hooks.")
elif abs(b1k_t) < 2.0:
    print(f"  B1_clean NULL  Δ={b1k_m:+.4f}  t={b1k_t:+.2f} (|t|<2)")
    print("  Clean demo does not drive order_score.")
    print("  Combined with B0 PASS: decodability ≠ use confirmed (L8 on gen side).")
    print("  Next: §3.5 direction-causal fallback — patch L*=8 directly,")
    print("        test if the activation direction causally controls output order.")
    print("        No Δ_nat denominator; report raw Δorder_score + all controls.")
else:
    print(f"  B1_clean NEGATIVE  Δ={b1k_m:+.4f}  t={b1k_t:+.2f}")
    print("  Consistent positive demo reversed the expected direction.")
    print("  Genuine reverse effect — investigate before patching.")

print("\nAudit notes recorded:")
print("  Audit-1: v2 B1 t=-2.49 was inflated by pseudo-replication (n=48→12).")
print("  Audit-2: correction-based B1 had primacy confound (wrong-first). B1_clean fixes it.")
print("  B1_corr (clustered) reports what the corrected v2 experiment actually shows.")
