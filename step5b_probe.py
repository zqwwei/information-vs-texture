"""
STEP 5b — Per-layer linear probe v2 (Phase 1, ORDER axis)

Matches step5b_data.json (single 'label': 0=conclusion-first, 1=reasons-first).

KEY CHANGES from v1 (all forced by the redesign):
  1. group = scenario_id   (not profile_id) -> probe must generalize order-pref
     to UNSEEN scenarios; also stops scenario-specific bigrams leaking across split.
  2. PRIMARY readpoint = mean over the FINAL ACCEPTED answer tokens (b_final_span).
     Rationale: both labels' final answers use the SAME words, only the ORDER
     differs -> no token-identity confound; at layer 0 the mean is ~identical
     (bag-of-words-equivalent -> ~chance), and it can only RISE above chance if
     deeper layers build an order-sensitive representation. That rise IS the signal.
     (v1's mean-over-ALL-answers pooled both orderings together and washed order out.)
  3. Lexical baseline computed unigram AND bigram (already verified ~0.500 at
     data-build time) -> drawn as the headroom line. Activation must clear it.

Core judgment (continuous): does B-form activation (final-answer readpoint) rise
ABOVE the lexical line (~0.5) across layers, and by how many std? If yes ->
the model encodes a structural preference invisible to bag-of-words. If it hugs
0.5 -> no linearly-accessible order representation at this scale.

Limitations logged at end (L1,L2,L6-close,L7-floor,L8-decodability!=use,L9-linear,L10-scale).
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2TokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42
np.random.seed(SEED)

DATA_PATH = "/Users/williams/Desktop/texture_experiment/data/step5b_data.json"
with open(DATA_PATH) as f:
    data = json.load(f)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

N_LAYERS = 13
MID = list(range(5, 10))


# ── Activation extraction ────────────────────────────────────────────────────
def span_token_mask(offsets, char_span):
    s, e = char_span
    return torch.tensor(
        [(ts >= s and te <= e and ts < te) for (ts, te) in offsets],
        dtype=torch.bool,
    )

def encode(text, char_span=None):
    """
    Returns dict of readpoints, each a list of 13 (768,) tensors:
      'span_mean' : mean over tokens inside char_span (final accepted answer).
                    If char_span is None -> mean over all tokens (A-form sanity).
      'last'      : last token of the whole sequence.
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # 13 x (1, T, 768)

    last = [hs[k][0, -1, :].clone() for k in range(N_LAYERS)]

    if char_span is not None:
        mask = span_token_mask(offsets, char_span)
        if mask.sum() == 0:
            mask = torch.ones(hs[0].shape[1], dtype=torch.bool)
        span_mean = [hs[k][0][mask].mean(dim=0).clone() for k in range(N_LAYERS)]
    else:
        span_mean = [hs[k][0].mean(dim=0).clone() for k in range(N_LAYERS)]

    return {"span_mean": span_mean, "last": last}


print(f"Encoding {len(data)} A-forms + {len(data)} B-forms through GPT-2 ...")
A_act, B_act = [], []
for i, d in enumerate(data):
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/{len(data)} ...")
    A_act.append(encode(d["a_form"], char_span=None))           # A: mean over all
    B_act.append(encode(d["b_form"], char_span=tuple(d["b_final_span"])))  # B: final answer
print("  Done.\n")


# ── Probe helpers ─────────────────────────────────────────────────────────────
def make_pipe():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])

def cv_acc(X, y, groups, n_splits=5):
    sc = cross_val_score(make_pipe(), X, y, cv=GroupKFold(n_splits), groups=groups, scoring="accuracy")
    return sc.mean(), sc.std()

def shuffle_floor(X, y, groups, n_repeat=10, n_splits=5):
    rng = np.random.default_rng(SEED); out = []
    for _ in range(n_repeat):
        out.extend(cross_val_score(make_pipe(), X, rng.permutation(y),
                                   cv=GroupKFold(n_splits), groups=groups, scoring="accuracy").tolist())
    return float(np.mean(out)), float(np.std(out))

def lex_acc(texts, y, groups, ngram, n_splits=5):
    X = CountVectorizer(ngram_range=ngram, min_df=1).fit_transform(texts).toarray().astype(np.float32)
    pipe = Pipeline([("sc", StandardScaler(with_mean=False)),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
    sc = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits), groups=groups, scoring="accuracy")
    return sc.mean(), sc.std()


y = np.array([d["label"] for d in data])
groups = np.array([d["scenario_id"] for d in data])

# Baselines (lexical already verified ~0.5 at data-build; recomputed here for the plot)
lex_uni = lex_acc([d["b_form"] for d in data], y, groups, (1, 1))
lex_bi  = lex_acc([d["b_form"] for d in data], y, groups, (1, 2))
X_b_mid = np.stack([B_act[i]["span_mean"][6].numpy() for i in range(len(data))])
sh = shuffle_floor(X_b_mid, y, groups)
print(f"Lexical(uni)={lex_uni[0]:.3f}  Lexical(1-2)={lex_bi[0]:.3f}  Shuffle={sh[0]:.3f}\n")

# Per-layer probes
res = {"B_final": [], "B_last": [], "A_mean": []}
for layer in range(N_LAYERS):
    Xb_f = np.stack([B_act[i]["span_mean"][layer].numpy() for i in range(len(data))])
    Xb_l = np.stack([B_act[i]["last"][layer].numpy()      for i in range(len(data))])
    Xa_m = np.stack([A_act[i]["span_mean"][layer].numpy() for i in range(len(data))])
    res["B_final"].append(cv_acc(Xb_f, y, groups))
    res["B_last"].append(cv_acc(Xb_l, y, groups))
    res["A_mean"].append(cv_acc(Xa_m, y, groups))


# ── Table ─────────────────────────────────────────────────────────────────────
print("=" * 88)
print("Per-layer probe accuracy (mean +/- std, GroupKFold(5) by scenario)  ORDER axis")
print(f"Lexical floor (uni)={lex_uni[0]:.3f}+/-{lex_uni[1]:.3f} | (1-2gram)={lex_bi[0]:.3f}+/-{lex_bi[1]:.3f} "
      f"| Shuffle={sh[0]:.3f}+/-{sh[1]:.3f}")
print("=" * 88)
print(f"{'layer':>5}  {'B-final (primary)':>18}  {'B-last (ctrl)':>16}  {'A-mean (sanity)':>16}")
print("-" * 64)
for k in range(N_LAYERS):
    bf, bl, am = res["B_final"][k], res["B_last"][k], res["A_mean"][k]
    mid = "*" if k in MID else " "
    print(f"{k:>5}{mid} {bf[0]:.3f}+/-{bf[1]:.3f}      {bl[0]:.3f}+/-{bl[1]:.3f}    {am[0]:.3f}+/-{am[1]:.3f}")


# ── Figure ────────────────────────────────────────────────────────────────────
xs = list(range(N_LAYERS))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axvspan(5, 9, alpha=0.08, color="gray")

def band(curve, color, ls, marker, label, lw=2):
    m = [c[0] for c in curve]; s = [c[1] for c in curve]
    ax.plot(xs, m, marker=marker, ls=ls, color=color, lw=lw, label=label)
    ax.fill_between(xs, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], alpha=0.15, color=color)

band(res["B_final"], "tab:blue",   "-",  "o", "B-form activation (final answer, PRIMARY)")
band(res["B_last"],  "steelblue",  "--", "s", "B-form activation (last token, ctrl)")
band(res["A_mean"],  "tab:orange", "-",  "^", "A-form activation (sanity, trivial)")
ax.axhline(lex_uni[0], color="tab:red", lw=2, ls="--", label=f"Lexical baseline, unigram ({lex_uni[0]:.3f})")
ax.axhline(lex_bi[0],  color="darkred", lw=1.3, ls=":", label=f"Lexical baseline, 1-2gram ({lex_bi[0]:.3f})")
ax.axhline(sh[0],      color="gray",    lw=1.3, ls=":", label=f"Shuffle floor ({sh[0]:.3f})")

ax.set_xlabel("Layer  (0=embedding, 1-12=transformer blocks)")
ax.set_ylabel("CV accuracy  (5-fold GroupKFold by scenario)")
ax.set_xticks(xs); ax.set_ylim(0.3, 1.05)
ax.set_title("ORDER axis (conclusion-first vs reasons-first)\n"
             "Lexical floor ~0.5 by construction -> any rise of the blue curve above it is model-computed")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
fig.tight_layout()
OUT_PNG = "/Users/williams/Desktop/texture_experiment/results/step5b_decode_order.png"
try:
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight"); print(f"\nsaved -> {OUT_PNG}")
except Exception:
    fig.savefig("/Users/williams/Desktop/texture_experiment/results/step5b_decode_order.png", dpi=140, bbox_inches="tight"); print("\nsaved -> step5b_decode_order.png")
plt.close()


# ── Quick read ────────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("QUICK READ  (ORDER axis; lexical floor is genuine ~0.5, so headroom is real)")
print("=" * 88)
bf_mid = np.array([res["B_final"][k][0] for k in MID])
bf_std = np.array([res["B_final"][k][1] for k in MID])
peak_layer = MID[int(np.argmax(bf_mid))]; peak = bf_mid.max(); peak_s = bf_std[int(np.argmax(bf_mid))]
l0 = res["B_final"][0][0]
above_lex = peak - lex_uni[0]
print(f"  B-form final-answer probe:  layer-0={l0:.3f}  mid-band peak={peak:.3f}+/-{peak_s:.3f} (layer {peak_layer})")
print(f"  Lexical floor (unigram):    {lex_uni[0]:.3f}")
print(f"  Rise across layers (peak - layer0) = {peak - l0:+.3f}")
print(f"  Activation - lexical = {above_lex:+.3f}  ({above_lex / max(bf_std.mean(),1e-6):.1f} std)")
if above_lex > 2 * bf_std.mean():
    print("  SIGNAL: activation clears the lexical floor by >2 std -> model encodes the")
    print("          order preference structurally (bag-of-words cannot, it is at chance).")
elif above_lex > 0:
    print("  WEAK: activation above lexical but small; report magnitude, do not over-claim.")
else:
    print("  NULL: activation hugs the lexical floor -> no linearly-accessible order")
    print("        representation at this scale. THIS is the evidence to justify a larger model.")

print("\nLimitations:")
print("  L1/L2: synthetic single-turn demonstrations; minimal 'person'.")
print("  L6 (closed): GroupKFold-by-scenario error bars; generalizes to unseen topics.")
print("  L7 (floor):  lexical baseline is genuine chance here (verified), so 'activation>lexical' is meaningful.")
print("  L8 (key):    decodability != use. Probe positive => order info linearly PRESENT;")
print("               not that GPT-2 uses it when generating. Causal patching = Phase 2.")
print("  L9:          linear only; non-linear order info would be invisible here.")
print("  L10:         GPT-2-small scale. Null here (unlike the Phase-0 null) WOULD justify a larger model.")
