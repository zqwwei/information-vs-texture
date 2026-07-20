"""
STEP 5b — Dataset generator v2 (Phase 1, ORDER axis only)

WHY v2: v1's lexical baseline = 1.000 on every axis -> zero headroom -> the core
test ("activation > lexical") was unsatisfiable. Root cause: the preference was
encoded in WORDS (corrections + answers), so bag-of-words decoded the label
perfectly. v1 also tried length/decisiveness, which are *intrinsically lexical*
(length = word count; "options" vs "one" = different words) -> ill-posed for a
"beyond surface" test.

ORDER is the one well-posed axis: conclusion-first vs reasons-first is a
STRUCTURAL property that can be made bag-of-words-INVISIBLE.

v2 design (makes lexical -> chance by construction):
  - Each scenario has a one-sentence VERDICT clause and a one-sentence RATIONALE
    clause, written to read naturally in either order.
  - B-form (one scenario, one correction round):
        {name}: {q}
        Assistant: {WRONG order}
        {name}: {generic, direction-neutral correction}   # same set for both labels
        Assistant: {RIGHT order = the preferred order}     # the ACCEPTED answer
        {name}: {generic accept}
    label 0 (prefers conclusion-first): wrong = R then V, right = V then R
    label 1 (prefers reasons-first):    wrong = V then R, right = R then V
  - So BOTH labels' B-forms contain BOTH orderings (one as wrong, one as right);
    only WHICH ordering sits in the accepted/final position differs.
    => unigram AND bigram bag-of-words are identical across labels for a scenario.
    => grouped-by-scenario lexical baseline is provably ~chance.
  - The label is a POSITIONAL/structural fact (which order was accepted), readable
    only by a model that represents the dialogue structure -- not by word counts.

Grouping: by SCENARIO (not by label). Holds out whole scenarios so the probe must
generalize the order-preference to unseen topics, and so any scenario-specific
boundary-bigram can't leak across the split.

Readpoint the probe should use (exported as char spans):
  b_final_span  = the FINAL accepted answer (preferred order; same words both labels,
                  only order differs -> no token-identity confound, only order-of-
                  contextualization). PRIMARY.
  b_asst_spans  = all assistant answers (secondary/diagnostic).
"""

import json, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NAMES = ["Alex", "Sam", "Jordan", "Morgan", "Casey", "Taylor", "Riley", "Avery"]

CORRECTIONS = [
    "No - other way around.",
    "Flip those two.",
    "Swap them.",
    "Reverse it.",
    "The other way, please.",
    "Put them the other way.",
]
ACCEPTS = ["Better.", "Yes, like that.", "That's it.", "Good.", "Perfect.", "Right."]

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

M = 4  # name/correction/accept variants per (scenario, label)


def conclusion_first(sc):
    return f"{sc['verdict']} {sc['rationale']}"

def reasons_first(sc):
    return f"{sc['rationale']} {sc['verdict']}"


def make_b_form(name, sc, label, v):
    if label == 0:
        wrong, right = reasons_first(sc), conclusion_first(sc)
    else:
        wrong, right = conclusion_first(sc), reasons_first(sc)
    cor = CORRECTIONS[v % len(CORRECTIONS)]
    acc = ACCEPTS[v % len(ACCEPTS)]
    text = (
        f"{name}: {sc['q']}\n"
        f"Assistant: {wrong}\n"
        f"{name}: {cor}\n"
        f"Assistant: {right}\n"
        f"{name}: {acc}"
    )
    return text, right


def make_a_form(name, sc, label):
    pref = ("the conclusion first, then the reasons" if label == 0
            else "the reasoning first, then the conclusion")
    return (f"{name} likes {pref}. Asked '{sc['q']}', {name} wants it answered "
            f"in that order.")


def find_spans(text, substr):
    idx = text.find(substr)
    if idx == -1:
        return None
    return (idx, idx + len(substr))


def find_assistant_spans(text):
    marker = "Assistant: "
    spans, pos = [], 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            break
        start = idx + len(marker)
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        spans.append((start, end))
        pos = end
    return spans


dataset = []
for sid, sc in enumerate(SCENARIOS):
    for label in (0, 1):
        for v in range(M):
            name = NAMES[(sid * M + v) % len(NAMES)]
            b_text, final_answer = make_b_form(name, sc, label, v)
            a_text = make_a_form(name, sc, label)
            # the wrong answer also contains the same words; make sure we grab the
            # FINAL (second) assistant span as the accepted answer, not the first.
            asst = find_assistant_spans(b_text)
            final_span = asst[-1]
            dataset.append({
                "scenario_id": sid,
                "label": label,
                "name": name,
                "variant": v,
                "a_form": a_text,
                "b_form": b_text,
                "b_final_span": list(final_span),
                "b_asst_spans": asst,
            })

OUT = "/Users/williams/Desktop/texture_experiment/data/step5b_data.json"
try:
    with open(OUT, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved -> {OUT}")
except FileNotFoundError:
    with open("/Users/williams/Desktop/texture_experiment/data/step5b_data.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Saved -> step5b_data.json (cwd)")

n0 = sum(1 for d in dataset if d["label"] == 0)
n1 = sum(1 for d in dataset if d["label"] == 1)
print(f"{len(dataset)} instances | {len(SCENARIOS)} scenarios x 2 labels x {M} variants "
      f"| label balance: {n0}/{n1}")

print("\n" + "=" * 72)
print("SAMPLE - same scenario, both labels (same words, only order differs)")
print("=" * 72)
for d in [x for x in dataset if x["scenario_id"] == 0 and x["variant"] == 0]:
    lab = "conclusion-first" if d["label"] == 0 else "reasons-first"
    s, e = d["b_final_span"]
    print(f"\n--- label {d['label']} ({lab}), name={d['name']} ---")
    print(d["b_form"])
    print(f"   [final/accepted answer -> '{d['b_form'][s:e]}']")


def asst_only(d):
    t = d["b_form"]; return " ".join(t[s:e] for s, e in d["b_asst_spans"])

def final_only(d):
    t = d["b_form"]; s, e = d["b_final_span"]; return t[s:e]

def lex_cv(texts, y, groups, ngram):
    X = CountVectorizer(ngram_range=ngram, min_df=1).fit_transform(texts).toarray().astype(np.float32)
    pipe = Pipeline([("sc", StandardScaler(with_mean=False)),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
    sc = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=groups, scoring="accuracy")
    return sc.mean(), sc.std()

y = np.array([d["label"] for d in dataset])
groups = np.array([d["scenario_id"] for d in dataset])

print("\n" + "=" * 72)
print("LEXICAL GATE  (GroupKFold(5) by scenario; chance = 0.500)")
print("=" * 72)
checks = {
    "B whole  (unigram)": lambda: lex_cv([d["b_form"] for d in dataset], y, groups, (1, 1)),
    "B whole  (1-2gram)": lambda: lex_cv([d["b_form"] for d in dataset], y, groups, (1, 2)),
    "B final  (unigram)": lambda: lex_cv([final_only(d) for d in dataset], y, groups, (1, 1)),
    "B final  (1-2gram)": lambda: lex_cv([final_only(d) for d in dataset], y, groups, (1, 2)),
    "B asst   (1-2gram)": lambda: lex_cv([asst_only(d) for d in dataset], y, groups, (1, 2)),
    "A whole  (1-2gram)": lambda: lex_cv([d["a_form"] for d in dataset], y, groups, (1, 2)),
}
worst_b = 0.0
for name, fn in checks.items():
    m, s = fn()
    flag = ""
    if name.startswith("B"):
        worst_b = max(worst_b, m)
        if m > 0.60:
            flag = "   <-- LEAK"
    print(f"  {name}: {m:.3f} +/- {s:.3f}{flag}")

print("-" * 72)
THRESH = 0.60
if worst_b <= THRESH:
    print(f"GATE PASSED  worst B-form lexical = {worst_b:.3f} <= {THRESH:.2f}")
    print("Headroom exists. Safe to run the probe on GPT-2.")
else:
    print(f"GATE FAILED  worst B-form lexical = {worst_b:.3f} > {THRESH:.2f}")
    print("Fix the dataset (label leaking into words) BEFORE running GPT-2.")
