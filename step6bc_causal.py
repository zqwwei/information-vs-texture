"""
STEP 6bc — Direction (§4) + Causal Patching (§5)

Gate A: PASS (L*=8, acc=1.000)
Gate B: PASS (B1_clean Δ_nat=+0.4363, t=6.27)

Two tightenings (spec §3.7 v4):
  收紧-1: direction, Δ_nat, and patching all on CLEAN-DEMO construction.
          (Not correction-based step5b. End-to-end consistent.)
  收紧-2: also test direction patch on NEUTRAL prompt (no persona).
          discriminates generic "order feature" vs persona-specific pathway.

L* = 8 (Qwen2.5-3B-Instruct)
Δ_nat = 0.4363 (B1_clean natural swing, recovery denominator)

Structure:
  Part A  Sanity: re-confirm decodability on clean-demo activations; compute d̂
  Part B  Causal test: additive patch, 12 held-out Q's, dose-response
  Part C  Controls: random null | necessity ablation | layer sweep
  Part D  Neutral prompt control (same patch, no persona)
  Part E  Projection-replacement (surgical: only swap order component)

Outputs:
  step6_doseresponse.png   Δorder_score vs α (true d̂ + random null)
  step6_layer_sweep.png    causal effect vs layer
  step6_neutral_ctrl.png   persona vs neutral Δorder_score at α=1
"""

import json, re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from collections import defaultdict

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE    = "/Users/williams/Desktop/texture_experiment"
DATA5B  = f"{BASE}/data/step5b_data.json"

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
N_SC   = len(SCENARIOS)
L_STAR = 8
DELTA_NAT = 0.4363   # from B1_clean Gate B

with open(DATA5B) as f:
    dataset = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}  |  L*={L_STAR}  |  Δ_nat={DELTA_NAT}")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto")
model.eval()
H_DIM = model.config.hidden_size
print(f"  Loaded. hidden_dim={H_DIM}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def chat(messages, add_gen=True):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_gen)

def build_clean_demo(label, p, v, h, include_q=True):
    """2-warmup positive demo (no wrong-first); optionally append Q'."""
    ctx1 = (h + 2 + 2*v) % N_SC
    ctx2 = (h + 4 + 2*v) % N_SC
    ctx1 = ctx1 if ctx1 not in (p, h) else (ctx1 + 1) % N_SC
    ctx2 = ctx2 if ctx2 not in (p, h, ctx1) else (ctx2 + 2) % N_SC
    msgs = []
    for ci in [ctx1, ctx2]:
        sc = SCENARIOS[ci]
        ans = (f"{sc['verdict']} {sc['rationale']}" if label == 0
               else f"{sc['rationale']} {sc['verdict']}")
        msgs += [{"role": "user",      "content": sc["q"]},
                 {"role": "assistant", "content": ans},
                 {"role": "user",      "content": "Perfect."}]
    if include_q:
        sc_h = SCENARIOS[h]
        msgs[-1]["content"] += "\n" + sc_h["q"]
    return chat(msgs)

def mean_logprob_with_hook(prompt_str, completion_str, hook_handle=None):
    """Per-token mean logprob of completion|prompt (teacher-forcing)."""
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

def order_score_fn(prompt_str, C_concl, C_reas):
    return mean_logprob_with_hook(prompt_str, C_concl) - \
           mean_logprob_with_hook(prompt_str, C_reas)


# ── Part A — Direction ────────────────────────────────────────────────────────
print("=" * 68)
print("Part A — d̂ at L*=8 (clean-demo, last-token readpoint)")
print("=" * 68)

pairs_dict = defaultdict(dict)
for d in dataset:
    pairs_dict[(d["scenario_id"], d["variant"])][d["label"]] = d
pairs = [(k, v[0], v[1]) for k, v in pairs_dict.items() if 0 in v and 1 in v]

# Get last-token activation at L_STAR for persona-only clean-demo (no Q')
_fast_tok = AutoTokenizer.from_pretrained(MODEL_ID)   # supports offset_mapping

def get_asst_mean_act(text, layer):
    """Mean over assistant-content token positions (preferred-order answer spans).
    Analogous to Gate A's mean-over-final-answer readpoint, which gave decodability=1.000."""
    enc = _fast_tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    # Find character spans of assistant content
    asst_spans = [(m.start(1), m.end(1))
                  for m in re.finditer(r'<\|im_start\|>assistant\n(.*?)(?=<\|im_end\|>)',
                                       text, re.DOTALL)]
    # Diagnostic (printed once)
    if not hasattr(get_asst_mean_act, "_printed"):
        get_asst_mean_act._printed = True
        print(f"  [span diag] found {len(asst_spans)} asst spans in text of len {len(text)}")
        for i, (s, e) in enumerate(asst_spans):
            print(f"    span {i}: chars {s}-{e}: '{text[s:e][:60]}'")
    mask = torch.tensor(
        [any(s <= ts and te <= e for s, e in asst_spans) for ts, te in offsets],
        dtype=torch.bool)
    if mask.sum() == 0:   # fallback: all tokens
        print("  [span diag] WARNING: no tokens matched spans, using all-token fallback")
        mask = torch.ones(len(offsets), dtype=torch.bool)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[layer][0][mask].float().mean(0).cpu()

print(f"Extracting L*={L_STAR} activations for {len(pairs)} pairs (full prompt incl. Q')...")
acts_label0, acts_label1 = [], []
labels_all, groups_all = [], []
for idx, (key, d0, d1) in enumerate(pairs):
    if (idx+1) % 16 == 0: print(f"  {idx+1}/{len(pairs)} ...")
    p, v = key
    h = (p + 1) % N_SC
    # Use full prompt (persona + Q') so last token carries order-preference info
    # Same readpoint as Gate A (which gave decodability=1.000)
    # include_q=True so the context is full (persona+Q'), same as used in order_score.
    # But readpoint = mean over warmup-answer tokens (not last-token), analogous to Gate A.
    prompt0 = build_clean_demo(0, p, v, h, include_q=True)
    prompt1 = build_clean_demo(1, p, v, h, include_q=True)
    a0 = get_asst_mean_act(prompt0, L_STAR)
    a1 = get_asst_mean_act(prompt1, L_STAR)
    acts_label0.append(a0); acts_label1.append(a1)
    labels_all += [0, 1]
    groups_all += [p, p]

A0 = torch.stack(acts_label0)  # (N_pairs, H)
A1 = torch.stack(acts_label1)

# Diff-of-means direction (reasons-first − conclusion-first)
d_vec = A1.mean(0) - A0.mean(0)
d_hat = (d_vec / d_vec.norm()).to(device)
print(f"  d̂ norm={d_vec.norm():.3f}")

# Sanity: re-confirm decodability on clean-demo activations
# Fix: X_all stacks [A0, A1] in that order, so labels must match [0..0, 1..1]
X_all = torch.cat([A0, A1], dim=0).numpy()          # (2N, H)
y_all = np.array([0]*len(A0) + [1]*len(A1))          # [0..0, 1..1]
# groups: scenario_id of each pair, repeated for label0 and label1 blocks
pair_groups = [key[0] for (key, _, _) in pairs]
g_all = np.array(pair_groups + pair_groups)            # (2N,)

# Quick sanity print
print(f"  X_all shape={X_all.shape}  label counts: "
      f"{(y_all==0).sum()} label-0, {(y_all==1).sum()} label-1")
pipe = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
cv_acc = cross_val_score(pipe, X_all, y_all, cv=GroupKFold(5), groups=g_all,
                         scoring="accuracy").mean()
print(f"  Decodability (clean-demo, GroupKFold-5): {cv_acc:.3f}  "
      f"({'confirmed ✓' if cv_acc > 0.80 else 'LOW ⚠'})")

# Cosine vs probe weight direction
pipe.fit(X_all, y_all)
w = torch.tensor(pipe.named_steps["clf"].coef_[0]).float()
w_hat = w / w.norm()
cos = F.cosine_similarity(d_hat.cpu().unsqueeze(0), w_hat.unsqueeze(0)).item()
print(f"  Cosine(d̂, probe_weight) = {cos:.3f}  "
      f"({'consistent ✓' if abs(cos) > 0.3 else 'low ⚠'}) "
      f"[sign: {'aligned' if cos > 0 else 'antiparallel'}]")

# c_target for projection-replacement (reasons-first centroid projection onto d̂)
c_target = float((A1.to(device) @ d_hat).mean().cpu())
print(f"  c_target (reasons-first centroid on d̂) = {c_target:.4f}\n")


# ── Hook factory ─────────────────────────────────────────────────────────────

def make_hook(alpha, direction, mode="add"):
    d = direction.float()
    def fn(module, input, output):
        # Qwen2 decoder layers may return a plain Tensor (with accelerate offload)
        # or a tuple (hidden_states, ...). Handle both.
        is_tensor = isinstance(output, torch.Tensor)
        h = (output if is_tensor else output[0]).float()
        if mode == "add":
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            h = h + alpha * norm * d
        elif mode == "ablate":
            proj = (h * d).sum(-1, keepdim=True) * d
            h = h - proj
        elif mode == "project_set":
            proj = (h * d).sum(-1, keepdim=True) * d
            h = h - proj + alpha * d   # alpha = c_target here
        h = h.to(output.dtype if is_tensor else output[0].dtype)
        return h if is_tensor else (h,) + output[1:]
    return fn


def with_hook(layer_idx, alpha, direction, mode, fn, *args, **kwargs):
    """Run fn(*args) with a hook registered, remove after."""
    handle = model.model.layers[layer_idx].register_forward_hook(
        make_hook(alpha, direction, mode))
    try:
        result = fn(*args, **kwargs)
    finally:
        handle.remove()
    return result


# ── Test items (12 held-out scenarios, variant 0, conclusion-first persona) ──

def get_test_item(h):
    p = (h - 1 + N_SC) % N_SC
    v = 0
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"
    prompt_persona  = build_clean_demo(0, p, v, h, include_q=True)  # concl-first persona
    prompt_neutral  = chat([{"role": "user", "content": sc["q"]}])   # no persona
    return prompt_persona, prompt_neutral, C_concl, C_reas

# Pre-compute unpatched baselines
print("=" * 68)
print("Part B — Causal test: dose-response (additive, α × 12 held-out Q's)")
print("=" * 68)

ALPHAS = [0, -2, -1, -0.5, 0.5, 1, 2, 4]
N_RAND_SEEDS = 3
LAYER_SWEEP  = [L_STAR - 2, L_STAR, L_STAR + 2]

os_persona_by_h = {}    # {h: {alpha: order_score}}
abs_lp_by_h_alpha = {}  # for coherence guard

print("Computing dose-response (persona context)...")
for h in range(N_SC):
    p_pers, p_neut, C_concl, C_reas = get_test_item(h)
    os_persona_by_h[h] = {}
    abs_lp_by_h_alpha[h] = {}
    for alpha in ALPHAS:
        if alpha == 0:
            os = order_score_fn(p_pers, C_concl, C_reas)
        else:
            os = with_hook(L_STAR, alpha, d_hat, "add",
                           order_score_fn, p_pers, C_concl, C_reas)
        # coherence: absolute logprob of C_concl under patch
        if alpha == 0:
            abs_lp = mean_logprob_with_hook(p_pers, C_concl)
        else:
            abs_lp = with_hook(L_STAR, alpha, d_hat, "add",
                               mean_logprob_with_hook, p_pers, C_concl)
        os_persona_by_h[h][alpha] = os
        abs_lp_by_h_alpha[h][alpha] = abs_lp
    if (h + 1) % 4 == 0: print(f"  h={h}: done  (α=0 os={os_persona_by_h[h][0]:.3f})")

# Δorder_score per h per alpha (relative to α=0 baseline)
delta_persona = {h: {a: os_persona_by_h[h][a] - os_persona_by_h[h][0]
                     for a in ALPHAS} for h in range(N_SC)}

# Coherence guard: flag α where |abs_lp| drops > 0.5 vs α=0
coherence_ok = {h: {a: (abs_lp_by_h_alpha[h][a] - abs_lp_by_h_alpha[h][0]) > -0.5
                    for a in ALPHAS} for h in range(N_SC)}

# Mean Δorder_score across 12 h, per alpha (only coherent items)
dose_mean, dose_se = {}, {}
for a in ALPHAS:
    vals = [delta_persona[h][a] for h in range(N_SC) if coherence_ok[h][a]]
    arr = np.array(vals)
    dose_mean[a] = arr.mean() if len(arr) else float("nan")
    dose_se[a]   = (arr.std(ddof=1) / len(arr)**0.5) if len(arr) > 1 else float("nan")
    flag = "" if coherence_ok[h][a] else "  ⚠ some items incoherent"
    print(f"  α={a:+.1f}  Δos={dose_mean[a]:+.4f}±{dose_se[a]:.4f}"
          f"  (n coherent={sum(coherence_ok[h][a] for h in range(N_SC))}){flag}")

# Spearman dose-response (using α ∈ positive side for reasons-first push)
pos_alphas = [a for a in ALPHAS if a >= 0]
pos_deltas = [dose_mean[a] for a in pos_alphas]
rho, _ = spearmanr(pos_alphas, pos_deltas)
print(f"\n  Dose-response Spearman ρ (α≥0 vs Δos) = {rho:.3f}  "
      f"({'monotone ✓' if rho < -0.8 else 'not monotone ⚠'})")
print(f"  [expected: negative ρ since α>0 pushes toward reas-first, lowering order_score]")


# ── Part C — Controls ─────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Part C — Controls")
print("=" * 68)

# (1) Random direction null (α=1, N_RAND_SEEDS random directions)
print("\n(1) Random direction null  (α=+1)")
rng = np.random.default_rng(SEED)
rand_deltas_all = []
for seed in range(N_RAND_SEEDS):
    r_np = rng.standard_normal(H_DIM).astype(np.float32)
    r_np /= np.linalg.norm(r_np)
    r_hat = torch.tensor(r_np).to(device)
    seed_deltas = []
    for h in range(N_SC):
        p_pers, _, C_concl, C_reas = get_test_item(h)
        os0 = os_persona_by_h[h][0]
        os_r = with_hook(L_STAR, 1.0, r_hat, "add", order_score_fn, p_pers, C_concl, C_reas)
        seed_deltas.append(os_r - os0)
    rand_deltas_all.extend(seed_deltas)
    print(f"  seed={seed}  Δos_mean={np.mean(seed_deltas):+.4f}")

sigma_rand = np.std(rand_deltas_all)
mean_rand  = np.mean(rand_deltas_all)
true_delta_alpha1 = dose_mean.get(1, float("nan"))
print(f"  Null  mean={mean_rand:+.4f}  σ={sigma_rand:.4f}")
print(f"  True (α=1) Δos={true_delta_alpha1:+.4f}")
criterion_a = abs(true_delta_alpha1 - mean_rand) >= 2 * sigma_rand
print(f"  Criterion (a) direction-specific: "
      f"|Δtrue−Δnull|/σ = {abs(true_delta_alpha1-mean_rand)/sigma_rand:.1f}  "
      f"→ {'PASS ✓' if criterion_a else 'FAIL ✗'}")

# (2) Necessity ablation (project out d̂, no add)
print("\n(2) Necessity ablation  (project out d̂ only)")
abl_deltas = []
for h in range(N_SC):
    p_pers, _, C_concl, C_reas = get_test_item(h)
    os0  = os_persona_by_h[h][0]
    os_a = with_hook(L_STAR, 0.0, d_hat, "ablate", order_score_fn, p_pers, C_concl, C_reas)
    abl_deltas.append(os_a - os0)
abl_mean = np.mean(abl_deltas)
necessity_ratio = abs(abl_mean) / (abs(os_persona_by_h[0][0]) + 1e-9)
# better: compare ablated score to unpatched score
all_os0 = np.array([os_persona_by_h[h][0] for h in range(N_SC)])
all_os_abl = np.array([os_persona_by_h[h][0] + abl_deltas[h] for h in range(N_SC)])
necessity_frac = abs(all_os_abl.mean()) / (abs(all_os0.mean()) + 1e-9)
print(f"  Ablated order_score mean: {all_os_abl.mean():+.4f}  "
      f"(unpatched: {all_os0.mean():+.4f})")
print(f"  Necessity fraction: |ablated|/|unpatched| = {necessity_frac:.2f}  "
      f"{'(≤0.5 → direction necessary ✓)' if necessity_frac <= 0.5 else '(>0.5 → not very necessary)'}")

# (3) Layer sweep  (α=+1, layers L*-2 to L*+2)
print("\n(3) Layer sweep  (α=+1, toward reasons-first)")
layer_deltas = {}
for L in LAYER_SWEEP:
    deltas = []
    for h in range(N_SC):
        p_pers, _, C_concl, C_reas = get_test_item(h)
        os0 = os_persona_by_h[h][0]
        os_L = with_hook(L, 1.0, d_hat, "add", order_score_fn, p_pers, C_concl, C_reas)
        deltas.append(os_L - os0)
    layer_deltas[L] = np.array(deltas)
    print(f"  Layer {L:2d}  Δos={layer_deltas[L].mean():+.4f}±"
          f"{layer_deltas[L].std(ddof=1)/len(deltas)**0.5:.4f}")


# ── Part D — Neutral prompt control ───────────────────────────────────────────
print("\n" + "=" * 68)
print("Part D — Neutral prompt control (same patch, no persona)")
print("=" * 68)
ALPHAS_NEUTRAL = [0, 0.5, 1, 2, 4]
os_neutral_by_h = {}
for h in range(N_SC):
    _, p_neut, C_concl, C_reas = get_test_item(h)
    os_neutral_by_h[h] = {}
    for alpha in ALPHAS_NEUTRAL:
        if alpha == 0:
            os = order_score_fn(p_neut, C_concl, C_reas)
        else:
            os = with_hook(L_STAR, alpha, d_hat, "add",
                           order_score_fn, p_neut, C_concl, C_reas)
        os_neutral_by_h[h][alpha] = os

neutral_delta_by_alpha = {}
for a in ALPHAS_NEUTRAL:
    vals = [os_neutral_by_h[h][a] - os_neutral_by_h[h][0] for h in range(N_SC)]
    neutral_delta_by_alpha[a] = np.mean(vals)
    print(f"  α={a:+.1f}  neutral Δos={neutral_delta_by_alpha[a]:+.4f}"
          f"  vs persona Δos={dose_mean.get(a, float('nan')):+.4f}")

baseline_neutral = np.mean([os_neutral_by_h[h][0] for h in range(N_SC)])
baseline_persona = all_os0.mean()
print(f"\n  Unpatched baselines: neutral={baseline_neutral:+.4f}  persona={baseline_persona:+.4f}")
print(f"  → persona vs neutral diff at α=0: {baseline_persona - baseline_neutral:+.4f}  (= Gate-B signal)")
neutral_alpha1 = neutral_delta_by_alpha.get(1, float("nan"))
persona_alpha1 = dose_mean.get(1, float("nan"))
print(f"  Patch effect (α=1): neutral Δ={neutral_alpha1:+.4f}  persona Δ={persona_alpha1:+.4f}")
if abs(neutral_alpha1) > 0.1:
    print("  → patch works on neutral too: GENERIC order feature (not persona-specific)")
else:
    print("  → patch mostly needs persona context: more PERSONA-SPECIFIC")


# ── Part E — Projection-replacement (surgical) ───────────────────────────────
print("\n" + "=" * 68)
print("Part E — Projection-replacement at L*=8 (set order component = c_target)")
print("=" * 68)
proj_deltas = []
for h in range(N_SC):
    p_pers, _, C_concl, C_reas = get_test_item(h)
    os0 = os_persona_by_h[h][0]
    os_proj = with_hook(L_STAR, c_target, d_hat, "project_set",
                        order_score_fn, p_pers, C_concl, C_reas)
    proj_deltas.append(os_proj - os0)

proj_mean = np.mean(proj_deltas)
proj_se   = np.std(proj_deltas, ddof=1) / len(proj_deltas)**0.5
proj_t    = proj_mean / (proj_se + 1e-9)
recovery_R = abs(proj_mean) / (DELTA_NAT + 1e-9)
print(f"  Projection-set Δos = {proj_mean:+.4f} ± {proj_se:.4f}  t={proj_t:+.2f}")
print(f"  Recovery R = |Δos_proj| / Δ_nat = {abs(proj_mean):.4f} / {DELTA_NAT} = {recovery_R:.3f}")


# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Fig 1: Dose-response
ax = axes[0]
a_vals = ALPHAS
y_true = [dose_mean[a] for a in a_vals]
y_se   = [dose_se[a]   for a in a_vals]
ax.plot(a_vals, y_true, "o-", color="tab:blue", lw=2, label=f"True d̂  (ρ={rho:.2f})")
ax.fill_between(a_vals, [y-s for y,s in zip(y_true,y_se)],
                         [y+s for y,s in zip(y_true,y_se)], alpha=0.2, color="tab:blue")
ax.axhline(mean_rand, color="gray", ls="--", lw=1.5, label=f"Random null μ={mean_rand:+.3f}")
ax.axhspan(mean_rand-2*sigma_rand, mean_rand+2*sigma_rand, alpha=0.1, color="gray")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("α (patch scale)"); ax.set_ylabel("Δorder_score")
ax.set_title(f"Dose-response (persona+Q', toward reas-first)\n"
             f"crit(a)={'✓' if criterion_a else '✗'}  ρ={rho:.2f}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Fig 2: Layer sweep
ax = axes[1]
Ls   = list(layer_deltas.keys())
Lm   = [layer_deltas[L].mean()                                  for L in Ls]
Lse  = [layer_deltas[L].std(ddof=1)/len(layer_deltas[L])**0.5  for L in Ls]
ax.bar(Ls, Lm, color="tab:blue", alpha=0.7)
ax.errorbar(Ls, Lm, yerr=[2*s for s in Lse], fmt="none", color="black", capsize=5)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(mean_rand, color="gray", ls="--", lw=1.2, label=f"Random null μ")
ax.set_xlabel("Layer"); ax.set_ylabel("Δorder_score (α=+1)")
ax.set_title(f"Layer sweep (α=1, toward reas-first)\nL*={L_STAR} should peak")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Fig 3: Neutral vs Persona
ax = axes[2]
a_neut = ALPHAS_NEUTRAL
y_neut = [neutral_delta_by_alpha[a] for a in a_neut]
y_pers = [dose_mean.get(a, float("nan")) for a in a_neut]
ax.plot(a_neut, y_neut, "s--", color="tab:orange", lw=2, label="Neutral (no persona)")
ax.plot(a_neut, y_pers, "o-",  color="tab:blue",   lw=2, label="With persona (B1_clean)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("α"); ax.set_ylabel("Δorder_score")
ax.set_title("Neutral vs persona: is patch effect generic\nor persona-specific?")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("Step 6bc — Causal patching at L*=8  (Qwen2.5-3B-Instruct)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{BASE}/results/step6_causal.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {BASE}/results/step6_causal.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SUMMARY — Phase 2 causal patching")
print("=" * 68)

# Criterion (b): Spearman monotone
crit_b = abs(rho) >= 0.8 and rho < 0

# Criterion (c): mean/SE across 12 scenarios at α=1
crit_c_vals = [delta_persona[h][1] for h in range(N_SC) if coherence_ok[h][1]]
crit_c_arr  = np.array(crit_c_vals)
crit_c_t    = crit_c_arr.mean() / (crit_c_arr.std(ddof=1)/len(crit_c_arr)**0.5 + 1e-9)
crit_c      = abs(crit_c_t) >= 2.0 and crit_c_arr.mean() < 0

print(f"\n  Gate A:   L*={L_STAR}, clean-demo decodability={cv_acc:.3f}, cosine(d̂,probe)={cos:.3f}")
print(f"  Gate B:   B1_clean Δ_nat={DELTA_NAT}")
print(f"\n  Causal criteria:")
print(f"  (a) direction-specific:  |Δtrue-Δnull|/σ={abs(true_delta_alpha1-mean_rand)/sigma_rand:.1f}  "
      f"→ {'PASS ✓' if criterion_a else 'FAIL ✗'}")
print(f"  (b) dose-response:       ρ={rho:.3f}  "
      f"→ {'PASS ✓' if crit_b else 'FAIL ✗'}")
print(f"  (c) significant & signed: t={crit_c_t:+.2f}  "
      f"→ {'PASS ✓' if crit_c else 'FAIL ✗'}")
print(f"\n  Recovery R (projection-set) = {recovery_R:.3f}")
if   recovery_R >= 0.5:  print("    → strong causal mediation (R ≳ 0.5)")
elif recovery_R >= 0.2:  print("    → partial causal channel (0.2 ≤ R < 0.5)")
elif criterion_a:        print("    → weak but real causal channel (R < 0.2, but (a) passes)")
else:                    print("    → no detectable causal use on this linear direction")

print(f"\n  Necessity: |ablated|/|unpatched| = {necessity_frac:.2f}  "
      f"{'(direction necessary ✓)' if necessity_frac <= 0.5 else '(not necessary)'}")
print(f"  Neutral Δ(α=1)={neutral_alpha1:+.4f} vs Persona Δ(α=1)={persona_alpha1:+.4f}  "
      f"→ {'generic feature' if abs(neutral_alpha1) > 0.1 else 'persona-specific'}")

print(f"\nLimitations:")
print(f"  L8:  decodability ≠ use; Phase 2 tests 'use in this setup at this layer'.")
print(f"  L11: linear direction; non-linear mediation would be invisible.")
print(f"  L12: behavioral handle is B1_clean (Δ_nat={DELTA_NAT}); "
      f"if weak, recovery upper bound is low.")
print(f"  L15: B1_clean strong effect may reflect format-induction (copy recent-assistant-order),")
print(f"       not persona-model. Claim limited to: 'L*=8 order direction causally controls")
print(f"       output order in this setup.' NOT 'model uses a person-representation'.")
print(f"       Neutral-prompt control (Part D) partially probes this distinction.")
