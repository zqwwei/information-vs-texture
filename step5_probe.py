"""
STEP 5 — Per-layer linear probe (Phase 1, decodability)

For each preference axis × each layer (0–12) × each readpoint:
  - Logistic regression probe, GroupKFold(5) by profile_id
  - Shuffle floor (label-shuffled control)
  - Lexical baseline (CountVectorizer + LogReg on raw text)

Readpoints:
  - mean-over-answer-tokens (B-form primary; A-form uses mean-over-all)
  - last-token (control)

Core judgment (continuous, not binary):
  B-form activation probe vs B-form lexical baseline, per layer.
  If activation > lexical in mid-layers → model computed something beyond surface words.

Limitations logged at end (L1, L2, L6-close, L7-floor, L8-decodability≠use, L9-linear-only, L10-scale).
"""

import json, random
import numpy as np
import torch
import torch.nn.functional as F
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
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "/Users/williams/Desktop/texture_experiment/data/step5_data.json"
with open(DATA_PATH) as f:
    data = json.load(f)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

AXES   = ["length", "order", "decisiveness"]
N_LAYERS = 13
MID = list(range(5, 10))


# ── Activation extraction ──────────────────────────────────────────────────────

def get_asst_token_mask(text, asst_spans, n_tokens, offsets):
    """Boolean mask (len=n_tokens): True if token is inside an assistant span."""
    mask = [False] * n_tokens
    for (span_start, span_end) in asst_spans:
        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s >= span_start and tok_e <= span_end and tok_s < tok_e:
                mask[i] = True
    return mask


def encode(text, asst_spans=None):
    """
    Returns:
      last_vecs:  list of 13 tensors, each shape (768,)  — last-token readpoint
      mean_vecs:  list of 13 tensors, each shape (768,)  — mean-over-answer-tokens readpoint
                  (if asst_spans is None: mean over all tokens)
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # 13 × (1, T, 768)

    last_vecs = [hs[k][0, -1, :].clone() for k in range(N_LAYERS)]

    if asst_spans:
        n_tokens = hs[0].shape[1]
        mask = get_asst_token_mask(text, asst_spans, n_tokens, offsets)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        if mask_t.sum() == 0:
            mask_t = torch.ones(n_tokens, dtype=torch.bool)  # fallback: all tokens
        mean_vecs = [hs[k][0][mask_t].mean(dim=0).clone() for k in range(N_LAYERS)]
    else:
        mean_vecs = [hs[k][0].mean(dim=0).clone() for k in range(N_LAYERS)]

    return last_vecs, mean_vecs


print("Encoding 128 contexts through GPT-2 …")
activations = {}  # key -> {"last": list_13, "mean": list_13}
for i, d in enumerate(data):
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/128 …")
    ak = f"a_{i}"
    bk = f"b_{i}"
    a_last, a_mean = encode(d["a_form"], asst_spans=None)
    b_last, b_mean = encode(d["b_form"], asst_spans=d["b_asst_spans"])
    activations[ak] = {"last": a_last, "mean": a_mean}
    activations[bk] = {"last": b_last, "mean": b_mean}
print("  Done.\n")


# ── Probe helper ───────────────────────────────────────────────────────────────

def make_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])


def probe_layer(X, y, groups, n_splits=5):
    """Return mean ± std CV accuracy."""
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(make_pipe(), X, y, cv=cv, groups=groups,
                             scoring="accuracy")
    return scores.mean(), scores.std()


def shuffle_layer(X, y, groups, n_repeat=10, n_splits=5):
    """Shuffle-label floor: mean ± std across shuffles and folds."""
    all_scores = []
    cv = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(SEED)
    for _ in range(n_repeat):
        y_sh = rng.permutation(y)
        sc = cross_val_score(make_pipe(), X, y_sh, cv=cv, groups=groups,
                             scoring="accuracy")
        all_scores.extend(sc.tolist())
    return np.mean(all_scores), np.std(all_scores)


def lexical_probe(texts, y, groups, n_splits=5):
    """CountVectorizer (word unigrams+bigrams) + LogReg, same GroupKFold."""
    vec = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    scores = cross_val_score(make_pipe(), X, y, cv=GroupKFold(n_splits),
                             groups=groups, scoring="accuracy")
    return scores.mean(), scores.std()


# ── Main loop ─────────────────────────────────────────────────────────────────

results = {}  # axis → {layer → {readpoint → (mean, std)}}

for axis in AXES:
    print(f"Probing axis: {axis}")
    y_a = np.array([d["labels"][axis] for d in data])
    y_b = np.array([d["labels"][axis] for d in data])
    groups = np.array([d["profile_id"] for d in data])

    # Lexical baselines
    a_texts = [d["a_form"] for d in data]
    b_texts = [d["b_form"] for d in data]
    lex_a_m, lex_a_s = lexical_probe(a_texts, y_a, groups)
    lex_b_m, lex_b_s = lexical_probe(b_texts, y_b, groups)

    # Shuffle floor on B mean-pool activations (any layer gives same randomness)
    X_b_mid = np.stack([activations[f"b_{i}"]["mean"][6].numpy() for i in range(len(data))])
    sh_m, sh_s = shuffle_layer(X_b_mid, y_b, groups)

    layer_res = {}
    for layer in range(N_LAYERS):
        X_a_last = np.stack([activations[f"a_{i}"]["last"][layer].numpy() for i in range(len(data))])
        X_a_mean = np.stack([activations[f"a_{i}"]["mean"][layer].numpy() for i in range(len(data))])
        X_b_last = np.stack([activations[f"b_{i}"]["last"][layer].numpy() for i in range(len(data))])
        X_b_mean = np.stack([activations[f"b_{i}"]["mean"][layer].numpy() for i in range(len(data))])

        al_m, al_s = probe_layer(X_a_last, y_a, groups)
        am_m, am_s = probe_layer(X_a_mean, y_a, groups)
        bl_m, bl_s = probe_layer(X_b_last, y_b, groups)
        bm_m, bm_s = probe_layer(X_b_mean, y_b, groups)

        layer_res[layer] = {
            "a_last": (al_m, al_s), "a_mean": (am_m, am_s),
            "b_last": (bl_m, bl_s), "b_mean": (bm_m, bm_s),
        }

    results[axis] = {
        "layers":  layer_res,
        "lex_a":   (lex_a_m, lex_a_s),
        "lex_b":   (lex_b_m, lex_b_s),
        "shuffle": (sh_m, sh_s),
    }
    print(f"  lex_b={lex_b_m:.3f}±{lex_b_s:.3f}  shuffle={sh_m:.3f}±{sh_s:.3f}")


# ── Tables ────────────────────────────────────────────────────────────────────

print("\n" + "="*96)
print("Per-layer probe accuracy (mean ± std across 5 GroupKFold folds)")
print("="*96)
for axis in AXES:
    r = results[axis]
    print(f"\n── {axis.upper()} ──  "
          f"lex_b={r['lex_b'][0]:.3f}±{r['lex_b'][1]:.3f}  "
          f"lex_a={r['lex_a'][0]:.3f}±{r['lex_a'][1]:.3f}  "
          f"shuffle={r['shuffle'][0]:.3f}±{r['shuffle'][1]:.3f}")
    print(f"{'layer':>5}  {'B-mean(primary)':>18}  {'B-last(ctrl)':>14}  {'A-mean(sanity)':>16}  {'A-last':>8}")
    print("-" * 70)
    for k in range(N_LAYERS):
        lr = r["layers"][k]
        bm, bs = lr["b_mean"]; bl, bls = lr["b_last"]
        am, as_ = lr["a_mean"]
        al, als = lr["a_last"]
        mid = "*" if k in MID else " "
        print(f"{k:>5}{mid} {bm:.3f}±{bs:.3f}           {bl:.3f}±{bls:.3f}    {am:.3f}±{as_:.3f}    {al:.3f}±{als:.3f}")


# ── Figures ───────────────────────────────────────────────────────────────────

COLORS = {
    "b_mean": "tab:blue",   "b_last": "steelblue",
    "a_mean": "tab:orange", "lex_b": "tab:red", "shuffle": "gray",
}

for axis in AXES:
    r = results[axis]
    xs = list(range(N_LAYERS))

    bm_means = [r["layers"][k]["b_mean"][0] for k in xs]
    bm_stds  = [r["layers"][k]["b_mean"][1] for k in xs]
    bl_means = [r["layers"][k]["b_last"][0] for k in xs]
    bl_stds  = [r["layers"][k]["b_last"][1] for k in xs]
    am_means = [r["layers"][k]["a_mean"][0] for k in xs]
    am_stds  = [r["layers"][k]["a_mean"][1] for k in xs]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(5, 9, alpha=0.08, color="gray", label="_mid-band")

    ax.plot(xs, bm_means, "o-", color=COLORS["b_mean"], lw=2, label="B-form activation (mean-over-answer, primary)")
    ax.fill_between(xs,
                    [m-s for m,s in zip(bm_means, bm_stds)],
                    [m+s for m,s in zip(bm_means, bm_stds)],
                    alpha=0.2, color=COLORS["b_mean"])

    ax.plot(xs, bl_means, "s--", color=COLORS["b_last"], lw=1.5, label="B-form activation (last-token, ctrl)")
    ax.fill_between(xs,
                    [m-s for m,s in zip(bl_means, bl_stds)],
                    [m+s for m,s in zip(bl_means, bl_stds)],
                    alpha=0.15, color=COLORS["b_last"])

    ax.plot(xs, am_means, "^-", color=COLORS["a_mean"], lw=1.5, label="A-form activation (sanity, should be trivial)")
    ax.fill_between(xs,
                    [m-s for m,s in zip(am_means, am_stds)],
                    [m+s for m,s in zip(am_means, am_stds)],
                    alpha=0.12, color=COLORS["a_mean"])

    lex_b_m, lex_b_s = r["lex_b"]
    sh_m, sh_s = r["shuffle"]
    ax.axhline(lex_b_m, color=COLORS["lex_b"], lw=2, ls="--",
               label=f"B-form lexical baseline ({lex_b_m:.3f}±{lex_b_s:.3f})")
    ax.axhspan(lex_b_m - lex_b_s, lex_b_m + lex_b_s, alpha=0.10, color=COLORS["lex_b"])
    ax.axhline(sh_m, color=COLORS["shuffle"], lw=1.5, ls=":",
               label=f"Shuffle floor ({sh_m:.3f}±{sh_s:.3f})")

    ax.set_xlabel("Layer  (0=embedding, 1–12=transformer blocks)")
    ax.set_ylabel("CV accuracy  (5-fold GroupKFold by profile_id)")
    ax.set_xticks(xs)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"Preference axis: {axis.upper()}\n"
                 "Shaded band = mid-band layers 5–9  |  Dashed red = B-form lexical baseline")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"/Users/williams/Desktop/texture_experiment/results/step5_decode_{axis}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    print(f"saved -> {fname}")
    plt.close()


# ── Quick read ────────────────────────────────────────────────────────────────

print("\n" + "="*96)
print("QUICK READ — B-form activation vs B-form lexical, mid-band (layers 5–9)")
print("="*96)
for axis in AXES:
    r = results[axis]
    lex_m, lex_s = r["lex_b"]
    sh_m, sh_s   = r["shuffle"]

    bm_mid = np.array([r["layers"][k]["b_mean"][0] for k in MID])
    bm_std_mid = np.array([r["layers"][k]["b_mean"][1] for k in MID])
    peak_layer = MID[int(np.argmax(bm_mid))]
    peak_acc   = bm_mid.max()
    peak_std   = bm_std_mid[int(np.argmax(bm_mid))]
    above_lex  = peak_acc - lex_m
    above_sh   = peak_acc - sh_m

    am_peak = max(r["layers"][k]["a_mean"][0] for k in range(N_LAYERS))

    print(f"\n  {axis.upper()}:")
    print(f"    B-form activation peak: layer {peak_layer}, acc={peak_acc:.3f}±{peak_std:.3f}")
    print(f"    B-form lexical:         {lex_m:.3f}±{lex_s:.3f}")
    print(f"    Shuffle floor:          {sh_m:.3f}±{sh_s:.3f}")
    print(f"    A-form activation peak: {am_peak:.3f}  (sanity; trivial if ≥0.95)")
    print(f"    → B-form activation − lexical = {above_lex:+.3f}  ({above_lex/lex_s if lex_s>0 else float('inf'):.1f} std above lex baseline)")
    print(f"    → B-form activation − shuffle  = {above_sh:+.3f}")
    if above_lex > 2 * bm_std_mid.mean():
        print(f"    Signal: activation > lexical by >{2:.0f}σ → model contributes beyond word surface")
    elif above_lex > 0:
        print(f"    Signal: activation > lexical (positive direction, small magnitude)")
    else:
        print(f"    Signal: activation ≤ lexical → no detectable increment over word surface at this scale")

print("\nLimitations:")
print("  L1/L2 (inherited): synthetic text, non-real experience; preference density not fully controlled.")
print("  L6 (closed):  GroupKFold gives error bars; n=1 resolved.")
print("  L7 (floor):   'above shuffle' is insufficient; must beat B-form lexical baseline to claim model contribution.")
print("  L8 (key):     decodability ≠ use. Probe positive → info linearly present; does NOT mean GPT-2 uses it in generation.")
print("                Causal patching (Phase 2, instruction-tuned model) is needed to test use.")
print("  L9:           Only linear decodability tested. Non-linear signal may exist but is invisible here.")
print("  L10:          GPT-2-small scale. If B-form activation hugs lexical/shuffle, THAT is the evidence")
print("                to justify moving to a larger model — not the Phase 0 null (which was a readout artifact).")
