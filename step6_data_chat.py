"""
STEP 6 — Data preparation: B-form → chat template + leave-one-out test items

For each persona instance in step5b_data.json, creates:
  - chat_prompt: B-form as system context + Q'_held_out as user turn
  - C_concl / C_reas: candidate completions (same words, two orderings)
  - held_out_sid: which scenario provides Q'

Leave-one-out: persona with scenario_id=p is paired with held-out scenario h=(p+1)%12.
This ensures Q' never appeared in the persona dialogue.

Outputs step6_test_items.json with 96 items.
"""

import json, sys

DATA = "/Users/williams/Desktop/texture_experiment/data/step5b_data.json"
OUT  = "/Users/williams/Desktop/texture_experiment/data/step6_test_items.json"

with open(DATA) as f:
    dataset = json.load(f)

# Reconstruct SCENARIOS from the dataset (scenario_id → q, verdict, rationale)
# We can read them from the b_form structure since conclusion_first = verdict+" "+rationale
# and reasons_first = rationale+" "+verdict.
# Safer: just hard-code them (they're fixed from step5b_dataset.py).
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

items = []
for d in dataset:
    p  = d["scenario_id"]
    h  = (p + 1) % N_SC   # held-out: next scenario (Q' never appeared in persona)
    sc = SCENARIOS[h]

    C_concl = f"{sc['verdict']} {sc['rationale']}"
    C_reas  = f"{sc['rationale']} {sc['verdict']}"

    # Chat-format: B-form as system context, Q' as user message.
    # apply_chat_template is called in step6a with the actual tokenizer.
    # We store the raw messages list so step6a can apply any model's template.
    messages = [
        {"role": "system",    "content": d["b_form"]},
        {"role": "user",      "content": sc["q"]},
    ]

    items.append({
        "persona_sid":  p,
        "held_out_sid": h,
        "label":        d["label"],           # 0=conclusion-first, 1=reasons-first
        "name":         d["name"],
        "variant":      d["variant"],
        "b_form":       d["b_form"],
        "b_final_span": d["b_final_span"],    # char span of accepted answer in b_form
        "b_asst_spans": d["b_asst_spans"],
        "q_new":        sc["q"],
        "C_concl":      C_concl,
        "C_reas":       C_reas,
        "messages":     messages,             # raw list for apply_chat_template
    })

with open(OUT, "w") as f:
    json.dump(items, f, indent=2)

print(f"Saved {len(items)} test items -> {OUT}")
print(f"\nLabel distribution: "
      f"{sum(1 for i in items if i['label']==0)} conclusion-first | "
      f"{sum(1 for i in items if i['label']==1)} reasons-first")

# Show one sample of each label
print("\n" + "="*70 + "\nSAMPLE items (one per label):")
for tgt_label in [0, 1]:
    ex = next(i for i in items if i["label"] == tgt_label)
    lab = "conclusion-first" if tgt_label == 0 else "reasons-first"
    print(f"\n--- label {tgt_label} ({lab}), persona_sid={ex['persona_sid']}, held_out={ex['held_out_sid']} ---")
    print("B-form (persona):")
    for line in ex["b_form"].split("\n"):
        print(f"  {line}")
    print(f"Q' (held-out):  {ex['q_new']}")
    print(f"C_concl:        {ex['C_concl']}")
    print(f"C_reas:         {ex['C_reas']}")

# Verify: Q' never appears in b_form
leaks = [i for i in items if i["q_new"] in i["b_form"]]
print(f"\nQ' leakage check: {len(leaks)} items where Q' appeared in persona "
      f"({'FAIL' if leaks else 'PASS'})")
