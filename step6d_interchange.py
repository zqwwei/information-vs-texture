"""
STEP 6d — Interchange (activation) patching  (spec §5.1)

Replaces additive steering (step6bc) with clean interchange patching.
Four bugs in step6bc fixed by design:
  bug-1: no coherence ceiling (in-distribution activations, just swapped)
  bug-2: C tokens not patched (patch scope = Q'+tail only, never continuation)
  bug-3: multi-layer covers redundancy (L8–L29 all decode; try 6-12 together if L8 fails)
  bug-4: same-order control replaces random null (P_c→P_c should give Δ≈0)

Setup:
  P_c = conclusion-first clean-demo persona + Q' (held-out scenario question)
  P_r = reasons-first  clean-demo persona + Q' (same Q', same demo scenarios)
  Patch scope: Q'+tail positions (identical tokens in P_c and P_r; carry order signal via attention)
  Surgical:  replace only d̂ component  → tests "does order direction causally mediate?"
  Whole:     replace entire hidden state → tests "does L8 context rep causally mediate?"
  Comparison is the conclusion (spec §5.1).

Controls:
  Same-order:  P_c tail → P_c tail (different run, same label) → Δ should ≈ 0
  Neutral:     interchange on neutral prompt (no persona)
  Reverse:     P_r tail → P_c tail (opposite direction) → Δ should flip sign
  Multi-layer: if single L fails, try L 6–12 simultaneously

d̂ recomputed from clean-demo activations (same as step6bc Part A; decodability=1.000).

Outputs: results/step6_interchange.png
"""

import json, re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

BASE    = "/Users/williams/Desktop/texture_experiment"
DATA5B  = f"{BASE}/data/step5b_data.json"
OUT_PNG = f"{BASE}/results/step6_interchange.png"

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
DELTA_NAT = 0.4363

with open(DATA5B) as f:
    dataset = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}  |  L*={L_STAR}  |  Δ_nat={DELTA_NAT}")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID} ...")
tok      = AutoTokenizer.from_pretrained(MODEL_ID)
fast_tok = AutoTokenizer.from_pretrained(MODEL_ID)
model    = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto")
model.eval()
H_DIM    = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
print(f"  Loaded. hidden_dim={H_DIM}  layers={N_LAYERS}\n")


# ── Prompt builders ───────────────────────────────────────────────────────────

def chat(messages, add_gen=True):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_gen)

def build_clean_demo(label, p, v, h, include_q=True):
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


# ── Direction d̂  (recompute from clean-demo, same as step6bc Part A) ─────────
print("=" * 68)
print("Part A — Recompute d̂ at L*=8 (clean-demo, span-mean readpoint)")
print("=" * 68)

pairs_dict = defaultdict(dict)
for d in dataset:
    pairs_dict[(d["scenario_id"], d["variant"])][d["label"]] = d
pairs = [(k, v[0], v[1]) for k, v in pairs_dict.items() if 0 in v and 1 in v]

def get_asst_mean_act(text, layer):
    """Use a forward hook to capture only L=layer activations (avoids storing all 37 layers)."""
    enc = fast_tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    asst_spans = [(m.start(1), m.end(1))
                  for m in re.finditer(
                      r'<\|im_start\|>assistant\n(.*?)(?=<\|im_end\|>)',
                      text, re.DOTALL)]
    mask = torch.tensor(
        [any(s <= ts and te <= e for s, e in asst_spans) for ts, te in offsets],
        dtype=torch.bool)
    if mask.sum() == 0:
        mask = torch.ones(len(offsets), dtype=torch.bool)

    captured = {}
    def hook(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        captured["h"] = h[0].float().cpu()  # (T, H)

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**enc)
    handle.remove()
    return captured["h"][mask].mean(0)

acts0, acts1 = [], []
for idx, (key, d0, d1) in enumerate(pairs):
    if (idx+1) % 16 == 0: print(f"  {idx+1}/{len(pairs)} ...")
    p, v = key; h = (p+1) % N_SC
    a0 = get_asst_mean_act(build_clean_demo(0, p, v, h, include_q=True), L_STAR)
    a1 = get_asst_mean_act(build_clean_demo(1, p, v, h, include_q=True), L_STAR)
    acts0.append(a0); acts1.append(a1)

A0, A1 = torch.stack(acts0), torch.stack(acts1)
d_vec = A1.mean(0) - A0.mean(0)
d_hat = (d_vec / d_vec.norm()).to(device)

# Verify decodability (label ordering fix: [A0,A1] → [0..0,1..1])
X = torch.cat([A0, A1]).numpy()
y = np.array([0]*len(A0) + [1]*len(A1))
g = np.array([k[0] for k,_,_ in pairs] * 2)
pipe = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
acc = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=g, scoring="accuracy").mean()
pipe.fit(X, y)
w = torch.tensor(pipe.named_steps["clf"].coef_[0]).float()
cos = F.cosine_similarity(d_hat.cpu().unsqueeze(0), (w/w.norm()).unsqueeze(0)).item()
print(f"  decodability={acc:.3f}  cosine(d̂,probe)={cos:.3f}")
c_reas = float((A1.to(device) @ d_hat).mean().cpu())
c_concl = float((A0.to(device) @ d_hat).mean().cpu())
print(f"  c_reas={c_reas:.3f}  c_concl={c_concl:.3f}\n")


# ── Token position helpers ────────────────────────────────────────────────────

def find_q_token_start(prompt_str, q_text):
    """Return token index of the LAST occurrence of q_text in prompt_str."""
    char_pos = prompt_str.rfind(q_text)
    if char_pos == -1:
        return None
    enc = fast_tok(prompt_str, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    for i, (s, e) in enumerate(offsets):
        if s >= char_pos:
            return i
    return None

def tokenize_prompt(prompt_str):
    return tok(prompt_str, return_tensors="pt").input_ids[0]


# ── Alignment verification ────────────────────────────────────────────────────
print("=" * 68)
print("Alignment verification (Q' start position in P_c vs P_r)")
print("=" * 68)

aligned_h = []   # held-out scenarios where alignment is verified
v_fixed = 0      # use variant 0 for alignment check and main experiment

for h in range(N_SC):
    p = (h - 1 + N_SC) % N_SC
    sc = SCENARIOS[h]
    q_text = sc["q"]
    P_c = build_clean_demo(0, p, v_fixed, h, include_q=True)
    P_r = build_clean_demo(1, p, v_fixed, h, include_q=True)
    ids_c = tokenize_prompt(P_c)
    ids_r = tokenize_prompt(P_r)
    q_start_c = find_q_token_start(P_c, q_text)
    q_start_r = find_q_token_start(P_r, q_text)
    tail_len_c = len(ids_c) - q_start_c if q_start_c else None
    tail_len_r = len(ids_r) - q_start_r if q_start_r else None
    # Check: tail tokens must match
    if q_start_c and q_start_r:
        tail_c = ids_c[q_start_c:].tolist()
        tail_r = ids_r[q_start_r:].tolist()
        ok = (tail_c == tail_r)
    else:
        ok = False
    status = "OK" if ok else "MISMATCH"
    print(f"  h={h:2d}  n_c={len(ids_c):3d}  n_r={len(ids_r):3d}  "
          f"q_start_c={q_start_c}  q_start_r={q_start_r}  tail_len={tail_len_c}  {status}")
    if ok:
        aligned_h.append(h)

print(f"\n  Aligned: {len(aligned_h)}/12 scenarios → {aligned_h}")
if len(aligned_h) < 6:
    print("  WARNING: fewer than 6 aligned scenarios — results may be noisy")


# ── Interchange engine ────────────────────────────────────────────────────────

def get_acts_at_layer(prompt_str, layer, token_start, token_end):
    """Cache hidden states at [token_start:token_end] for given layer via hook."""
    ids = tok(prompt_str, return_tensors="pt").input_ids.to(device)
    captured = {}
    def hook(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        captured["h"] = h[0, token_start:token_end, :].float().cpu()
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    handle.remove()
    return captured["h"]


def make_interchange_hook(cached_acts, patch_start, patch_end, mode, d_hat_dev):
    """
    cached_acts: (patch_len, H) tensor from source prompt
    patch_start, patch_end: token positions to patch in the current (target) prompt
    mode: "surgical" | "whole"
    Positions >= patch_end (C tokens) are NOT touched.
    """
    ca = cached_acts.to(d_hat_dev)
    def hook(module, input, output):
        is_t = isinstance(output, torch.Tensor)
        h = (output if is_t else output[0]).float()
        patch_len = min(patch_end - patch_start, ca.shape[0], h.shape[1] - patch_start)
        if patch_len <= 0:
            return output
        for i in range(patch_len):
            pos = patch_start + i
            if pos >= h.shape[1]: break
            a_tgt = h[0, pos]
            a_src = ca[i].to(h.device)
            if mode == "whole":
                h[0, pos] = a_src.to(h.dtype)
            else:  # surgical
                d = d_hat_dev.float()
                proj_tgt = (a_tgt.float() * d).sum() * d
                proj_src = (a_src.float() * d).sum() * d
                h[0, pos] = (a_tgt.float() - proj_tgt + proj_src).to(h.dtype)
        dtype_out = output.dtype if is_t else output[0].dtype
        return h.to(dtype_out) if is_t else (h.to(dtype_out),) + output[1:]
    return hook


def mean_lp(prompt_str, completion_str):
    """Per-token mean logprob of completion given prompt (no patching)."""
    p_ids = tok(prompt_str, return_tensors="pt").input_ids
    f_ids = tok(prompt_str + completion_str, return_tensors="pt").input_ids.to(device)
    n_p = p_ids.shape[1]; n_c = f_ids.shape[1] - n_p
    if n_c <= 0: return float("nan")
    with torch.no_grad():
        out = model(f_ids)
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    return lp[n_p-1:n_p-1+n_c, :][range(n_c), f_ids[0, n_p:]].mean().item()


def patched_os(prompt_c, prompt_r_or_c2, C_concl, C_reas, layers,
               mode, patch_start_c, patch_start_src):
    """
    Interchange patching: for each token in [patch_start_c:n_c],
    swap activations from prompt_r_or_c2 (source) into prompt_c (target).
    Returns order_score after patching.
    """
    n_c_prompt = tok(prompt_c, return_tensors="pt").input_ids.shape[1]
    patch_end_c = n_c_prompt  # don't touch C tokens (they come after n_c_prompt)

    # Cache source activations via hooks — only capture the layers we need.
    # This avoids output_hidden_states=True which stores all 36 layers (~18 MB per call).
    src_ids = tok(prompt_r_or_c2, return_tensors="pt").input_ids.to(device)
    src_captured = {}
    def make_src_hook(L_):
        def hook(module, inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            src_captured[L_] = h[0, patch_start_src:, :].float().cpu()
        return hook
    src_handles = [model.model.layers[L].register_forward_hook(make_src_hook(L))
                   for L in layers]
    with torch.no_grad():
        model(src_ids)
    for sh in src_handles:
        sh.remove()

    # Register interchange hooks for all requested layers
    handles = []
    for L in layers:
        n_src = src_ids.shape[1]
        src_tail_start = patch_start_src
        src_tail_end   = n_src
        ca = src_captured[L]
        h_fn = make_interchange_hook(ca, patch_start_c, patch_end_c, mode, d_hat)
        handles.append(model.model.layers[L].register_forward_hook(h_fn))

    # Run target prompt + both completions with hooks active
    def _lp(completion):
        full_ids = tok(prompt_c + completion, return_tensors="pt").input_ids.to(device)
        n_p = n_c_prompt; n_comp = full_ids.shape[1] - n_p
        if n_comp <= 0: return float("nan")
        with torch.no_grad():
            out = model(full_ids)
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        return lp[n_p-1:n_p-1+n_comp, :][range(n_comp), full_ids[0, n_p:]].mean().item()

    lp_concl = _lp(C_concl)
    lp_reas  = _lp(C_reas)
    for h_handle in handles:
        h_handle.remove()
    return lp_concl - lp_reas


# ── Main experiment ───────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Main interchange experiment (surgical + whole, aligned scenarios)")
print("=" * 68)

results = {
    "surgical_cross": [],   # P_c patched with P_r (cross-label) — main
    "whole_cross":    [],   # whole activation swap (cross-label)
    "surgical_same":  [],   # P_c patched with P_c copy (same-label control → Δ≈0)
    "surgical_rev":   [],   # P_r patched with P_c (reverse direction)
    "surgical_neutral":[],  # neutral prompt patched from P_r
}
unpatched_os = []

for h in aligned_h:
    p = (h - 1 + N_SC) % N_SC
    sc = SCENARIOS[h]
    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"

    P_c = build_clean_demo(0, p, v_fixed, h, include_q=True)
    P_r = build_clean_demo(1, p, v_fixed, h, include_q=True)
    P_neutral = chat([{"role": "user", "content": sc["q"]}])

    q_start_c = find_q_token_start(P_c, sc["q"])
    q_start_r = find_q_token_start(P_r, sc["q"])
    n_c = tok(P_c, return_tensors="pt").input_ids.shape[1]
    n_r = tok(P_r, return_tensors="pt").input_ids.shape[1]

    # For neutral: Q' is at position 0 (the only user message)
    q_start_neutral = 0

    # Baseline (no patching)
    os_base = mean_lp(P_c, C_concl) - mean_lp(P_c, C_reas)
    unpatched_os.append(os_base)

    # Cross-label: P_c ← P_r  (conclusion ← reasons)
    os_surg = patched_os(P_c, P_r, C_concl, C_reas,
                         [L_STAR], "surgical", q_start_c, q_start_r)
    os_whole = patched_os(P_c, P_r, C_concl, C_reas,
                          [L_STAR], "whole",    q_start_c, q_start_r)

    # Same-label control: P_c ← P_c (same label, control should give Δ≈0)
    # Use another variant (v=1) of the same label as the "source"
    P_c2 = build_clean_demo(0, p, (v_fixed+1)%4, h, include_q=True)
    q_start_c2 = find_q_token_start(P_c2, sc["q"])
    os_same = patched_os(P_c, P_c2, C_concl, C_reas,
                         [L_STAR], "surgical", q_start_c, q_start_c2)

    # Reverse: P_r ← P_c
    os_rev = patched_os(P_r, P_c, C_concl, C_reas,
                        [L_STAR], "surgical", q_start_r, q_start_c)
    os_rev_base = mean_lp(P_r, C_concl) - mean_lp(P_r, C_reas)

    # Neutral control: neutral ← P_r
    q_start_n = find_q_token_start(P_neutral, sc["q"]) or 0
    os_neut = patched_os(P_neutral, P_r, C_concl, C_reas,
                         [L_STAR], "surgical", q_start_n, q_start_r)
    os_neut_base = mean_lp(P_neutral, C_concl) - mean_lp(P_neutral, C_reas)

    results["surgical_cross"].append(os_surg - os_base)
    results["whole_cross"].append(os_whole - os_base)
    results["surgical_same"].append(os_same - os_base)
    results["surgical_rev"].append(os_rev - os_rev_base)
    results["surgical_neutral"].append(os_neut - os_neut_base)

    print(f"  h={h:2d}  base={os_base:+.3f}  "
          f"surg_cross={results['surgical_cross'][-1]:+.3f}  "
          f"whole_cross={results['whole_cross'][-1]:+.3f}  "
          f"same_ctrl={results['surgical_same'][-1]:+.3f}  "
          f"reverse={results['surgical_rev'][-1]:+.3f}  "
          f"neutral={results['surgical_neutral'][-1]:+.3f}")


def summarize(name, deltas):
    arr = np.array(deltas)
    m, se = arr.mean(), arr.std(ddof=1)/len(arr)**0.5 if len(arr)>1 else 0
    t = m/(se+1e-9)
    return arr, m, se, t

print("\n" + "=" * 68)
print("Results summary")
print("=" * 68)

sc_arr, sc_m, sc_se, sc_t = summarize("surgical_cross",   results["surgical_cross"])
wh_arr, wh_m, wh_se, wh_t = summarize("whole_cross",      results["whole_cross"])
sa_arr, sa_m, sa_se, sa_t = summarize("surgical_same",    results["surgical_same"])
rv_arr, rv_m, rv_se, rv_t = summarize("surgical_rev",     results["surgical_rev"])
nt_arr, nt_m, nt_se, nt_t = summarize("surgical_neutral", results["surgical_neutral"])
un_arr = np.array(unpatched_os)

print(f"  Unpatched base     mean={un_arr.mean():+.4f}  (≈ Gate-B os_concl-persona)")
print(f"  Surgical cross     Δ={sc_m:+.4f}  SE={sc_se:.4f}  t={sc_t:+.2f}  n={len(sc_arr)}")
print(f"  Whole    cross     Δ={wh_m:+.4f}  SE={wh_se:.4f}  t={wh_t:+.2f}")
print(f"  Surgical same-ctrl Δ={sa_m:+.4f}  SE={sa_se:.4f}  t={sa_t:+.2f}  (expect ≈0)")
print(f"  Surgical reverse   Δ={rv_m:+.4f}  SE={rv_se:.4f}  t={rv_t:+.2f}  (expect +)")
print(f"  Surgical neutral   Δ={nt_m:+.4f}  SE={nt_se:.4f}  t={nt_t:+.2f}  (generic vs persona)")

R_surgical = abs(sc_m) / (DELTA_NAT + 1e-9)
R_whole    = abs(wh_m) / (DELTA_NAT + 1e-9)
print(f"\n  Recovery R (surgical) = {abs(sc_m):.4f}/{DELTA_NAT} = {R_surgical:.3f}")
print(f"  Recovery R (whole)    = {abs(wh_m):.4f}/{DELTA_NAT} = {R_whole:.3f}")

# Decision: surgical vs whole
sym_ok  = (sc_m < 0 and rv_m > 0)      # expected direction
ctrl_ok = abs(sa_t) < 2.0              # same-order control near 0
sig_ok  = abs(sc_t) >= 2.0 and sc_m < 0
print(f"\n  Direction check (surgical Δ<0, reverse Δ>0): {'✓' if sym_ok else '✗'}")
print(f"  Same-order control ≈ 0: {'✓' if ctrl_ok else '✗'} (t={sa_t:+.2f})")
print(f"  Significance (|t|≥2, correct sign): {'✓' if sig_ok else '✗'}")

if sig_ok and ctrl_ok and sym_ok:
    if   R_surgical >= 0.5: interp = "strong causal mediation via order direction (R≳0.5)"
    elif R_surgical >= 0.2: interp = "partial causal channel (0.2≤R<0.5)"
    else:                   interp = "weak but real causal channel (R<0.2)"
    print(f"\n  RESULT: {interp}")
    if abs(wh_m) > abs(sc_m) * 1.5:
        print("  Note: whole > surgical — order direction is not the sole carrier "
              "(other components also contribute at this layer).")
elif not sig_ok:
    print(f"\n  RESULT: surgical interchange null at L*={L_STAR}.")
    print("  Suggest: try multi-layer (L 6–12 simultaneously).")
    print("  If multi-layer also null → 'decodable but not causally used' (clean Phase 2 result).")


# ── Multi-layer (if surgical is null) ────────────────────────────────────────
if not sig_ok:
    print("\n" + "=" * 68)
    print("Multi-layer interchange (L 6–12 simultaneously, surgical + whole)")
    print("=" * 68)
    MULTI_LAYERS = list(range(6, min(13, N_LAYERS+1)))
    ml_surg_deltas = []
    ml_whole_deltas = []
    for h in aligned_h:
        p = (h - 1 + N_SC) % N_SC
        sc = SCENARIOS[h]
        C_concl = f"{sc['verdict']} {sc['rationale']}"
        C_reas  = f"{sc['rationale']} {sc['verdict']}"
        P_c = build_clean_demo(0, p, v_fixed, h, include_q=True)
        P_r = build_clean_demo(1, p, v_fixed, h, include_q=True)
        q_start_c = find_q_token_start(P_c, sc["q"])
        q_start_r = find_q_token_start(P_r, sc["q"])
        os_base_h = unpatched_os[aligned_h.index(h)]
        os_ml_surg = patched_os(P_c, P_r, C_concl, C_reas,
                                MULTI_LAYERS, "surgical", q_start_c, q_start_r)
        os_ml_whole = patched_os(P_c, P_r, C_concl, C_reas,
                                 MULTI_LAYERS, "whole",    q_start_c, q_start_r)
        ml_surg_deltas.append(os_ml_surg  - os_base_h)
        ml_whole_deltas.append(os_ml_whole - os_base_h)
        print(f"  h={h:2d}  ml_surg={ml_surg_deltas[-1]:+.3f}  ml_whole={ml_whole_deltas[-1]:+.3f}")

    ml_s_arr = np.array(ml_surg_deltas)
    ml_w_arr = np.array(ml_whole_deltas)
    ml_s_m, ml_s_se = ml_s_arr.mean(), ml_s_arr.std(ddof=1)/len(ml_s_arr)**0.5
    ml_w_m, ml_w_se = ml_w_arr.mean(), ml_w_arr.std(ddof=1)/len(ml_w_arr)**0.5
    ml_s_t = ml_s_m/(ml_s_se+1e-9)
    ml_w_t = ml_w_m/(ml_w_se+1e-9)
    R_ml_s = abs(ml_s_m)/DELTA_NAT
    R_ml_w = abs(ml_w_m)/DELTA_NAT
    print(f"\n  Multi-layer surgical  Δ={ml_s_m:+.4f}  SE={ml_s_se:.4f}  t={ml_s_t:+.2f}  R={R_ml_s:.3f}")
    print(f"  Multi-layer whole     Δ={ml_w_m:+.4f}  SE={ml_w_se:.4f}  t={ml_w_t:+.2f}  R={R_ml_w:.3f}")

    ml_surg_sig = abs(ml_s_t) >= 2.0 and ml_s_m < 0
    ml_whole_sig = abs(ml_w_t) >= 2.0 and ml_w_m < 0
    if ml_surg_sig:
        print("  RESULT: Multi-layer surgical works — order direction distributed L6-L12.")
    elif ml_whole_sig:
        print("  RESULT: Multi-layer whole works, surgical null — L6-L12 context causally mediates,")
        print("          but not via the linear order direction (epiphenomenal direction).")
    else:
        print("  RESULT: Multi-layer surgical + whole both null.")
        print("  Spec §5.1 hardstop: decodable but NOT causally used. Report and stop.")
    ml_arr = ml_s_arr  # keep for figure compatibility
else:
    ml_arr = None
    ml_s_arr = ml_w_arr = None


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: per-scenario bars
xs = list(range(len(aligned_h)))
axes[0].bar([x-0.2 for x in xs], sc_arr, 0.35, label=f"Surgical cross (R={R_surgical:.2f})",
            color="tab:blue", alpha=0.8)
axes[0].bar([x+0.2 for x in xs], sa_arr, 0.35, label="Same-label ctrl (expect≈0)",
            color="gray", alpha=0.6)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_xticks(xs); axes[0].set_xticklabels([f"h={h}" for h in aligned_h], rotation=45, ha="right")
axes[0].set_ylabel("Δorder_score"); axes[0].set_title(f"Surgical interchange @ L*={L_STAR}\ncross (blue) vs same-label ctrl (gray)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis="y")

# Right: condition comparison
conds  = ["Surgical\ncross",  "Whole\ncross", "Same\nctrl", "Reverse", "Neutral"]
means  = [sc_m, wh_m, sa_m, rv_m, nt_m]
ses    = [sc_se, wh_se, sa_se, rv_se, nt_se]
colors = ["tab:blue", "steelblue", "gray", "tab:orange", "tab:green"]
bars = axes[1].bar(range(5), means, color=colors, alpha=0.8)
axes[1].errorbar(range(5), means, yerr=[2*s for s in ses],
                 fmt="none", color="black", capsize=5)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(conds)
axes[1].set_ylabel("Mean Δorder_score (± 2SE)")
axes[1].set_title(f"Interchange conditions\nL*={L_STAR}  Δ_nat={DELTA_NAT}")
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle("Step 6d — Interchange patching @ L*=8  (Qwen2.5-3B-Instruct)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")

print("\n" + "=" * 68)
print("Limitations")
print("=" * 68)
print("  L8:  decodability ≠ use; Phase 2 tests 'use in this setup at L*'.")
print("  L11: linear direction; non-linear mediation invisible.")
print("  L15: B1_clean strong effect may be format-induction; claim limited to")
print("       'L* order direction causally controls output order in this setup'.")
print("  (spec §5.1 hardstop): if surgical + whole + multi-layer all null →")
print("       report 'decodable but not causally used' and STOP.")
