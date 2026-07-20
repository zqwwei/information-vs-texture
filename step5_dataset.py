"""
STEP 5 — Dataset generator (Phase 1, linear probe)
8 profiles × 8 phrasings = 64 A-form + 64 B-form instances.
Exports data/step5_data.json.
"""

import json, random
from itertools import product
from transformers import GPT2Tokenizer

SEED = 42
random.seed(SEED)

NAMES = ["Alex", "Sam", "Jordan", "Morgan", "Casey", "Taylor", "Riley", "Avery"]

# ── A-form phrasings (4 per axis-label, cycled) ───────────────────────────────
LENGTH_A = {
    0: [
        "{n} prefers short answers.",
        "{n} wants brief, concise responses.",
        "{n} likes answers to be short and to the point.",
        "{n} prefers concise replies over lengthy ones.",
    ],
    1: [
        "{n} prefers thorough, detailed answers.",
        "{n} wants comprehensive, in-depth explanations.",
        "{n} likes responses that fully cover the topic.",
        "{n} prefers detailed responses over brief ones.",
    ],
}
ORDER_A = {
    0: [
        "{n} likes the conclusion first, then the reasons.",
        "{n} wants the bottom line up front, followed by the rationale.",
        "{n} prefers to get the answer before the explanation.",
        "{n} wants the main point stated before supporting details.",
    ],
    1: [
        "{n} likes to understand the reasoning before the conclusion.",
        "{n} wants the reasons laid out before the final answer.",
        "{n} prefers to see the thinking first, then the verdict.",
        "{n} wants the explanation before the bottom line.",
    ],
}
DECISIVENESS_A = {
    0: [
        "When {n} hesitates, {n} wants to be pushed toward a specific choice, not given more options.",
        "{n} wants a direct recommendation when deciding, not a list of alternatives.",
        "When {n} is torn, {n} wants one clear answer, not more things to weigh.",
        "{n} prefers to be told which option to take rather than given more choices.",
    ],
    1: [
        "When {n} hesitates, {n} wants more options to consider, not a forced choice.",
        "{n} wants to see a range of alternatives before deciding.",
        "When {n} is torn, {n} wants more paths laid out, not a single pick.",
        "{n} prefers having more options presented rather than being pushed toward one.",
    ],
}

# ── Corrections (8 per axis-label, cycled) ────────────────────────────────────
LENGTH_COR = {
    0: ["Too long. Keep it short.", "That's way too much. Trim it down.",
        "Shorter, please.", "Way too long. Just give me the essentials.",
        "Cut it down.", "Too much detail. Make it brief.",
        "That's too long. Get to the point.", "Trim it. I don't need all that."],
    1: ["That's too brief. Give me more detail.", "Too shallow. I want a fuller explanation.",
        "Can you elaborate? I need more than that.", "That's not enough — go deeper.",
        "Too short. Walk me through it properly.", "I need more than that. Expand on it.",
        "That's too thin. Give me the full picture.", "Not enough detail. Please elaborate."],
}
ORDER_COR = {
    0: ["Give the conclusion first, then the reasons.", "Lead with the answer, then explain.",
        "Bottom line first, then the reasoning.", "Tell me the conclusion first.",
        "Answer first, then walk me through why.", "Lead with the verdict, then the rationale.",
        "Give me the answer upfront, then the explanation.", "Conclusion first — then the reasons."],
    1: ["Walk me through the reasoning before you give the answer.",
        "I want the reasoning first, then the conclusion.",
        "Don't jump to the answer — explain the reasoning first.",
        "Give me the why before the what.",
        "Take me through the thinking before the conclusion.",
        "Reasoning first, then tell me what you think.",
        "I need the reasoning laid out before the verdict.",
        "Explain it first, then give your recommendation."],
}
DECISIVENESS_COR = {
    0: ["Don't give me more options. Just push me toward one.",
        "Stop listing alternatives. Tell me which one.",
        "No more options — just pick one for me.",
        "I don't need more choices. Make the call.",
        "Just tell me which one to pick.",
        "Enough options. Give me one clear answer.",
        "Stop hedging — which one should I do?",
        "Don't give me more alternatives. Just decide."],
    1: ["Give me more options to consider.",
        "Don't push me toward one. What are the alternatives?",
        "I need more choices, not a single answer.",
        "Stop pushing me toward one — lay out the options.",
        "I want to see more alternatives before deciding.",
        "Don't make the choice for me — what else is there?",
        "What other options should I be considering?",
        "Give me a broader range of alternatives."],
}
ACCEPTS = ["Better.", "Got it.", "That works.", "Okay.", "Yes, that's it.",
           "Perfect.", "Good.", "That's what I needed."]

# ── Scene pools (8 per axis) ───────────────────────────────────────────────────
LENGTH_SCENES = [
    dict(
        topic="compound interest",
        q="Can you explain how compound interest works?",
        wrong_short="Compound interest occurs when the interest you earn on a balance is added back to the principal each period, so that the next period you earn interest on a larger amount, and this process repeats, causing your balance to grow faster than simple interest would allow over any equivalent time horizon.",
        right_short="Your interest earns more interest over time.",
        wrong_long="Interest builds on itself.",
        right_long="Compound interest adds each period's earned interest back to the principal, so you earn interest on a progressively larger base. The longer the horizon and higher the rate, the more dramatic the effect: a 7% annual return doubles roughly every ten years through this mechanism alone.",
    ),
    dict(
        topic="how vaccines work",
        q="Can you explain how vaccines work?",
        wrong_short="Vaccines expose your immune system to a safe version of a pathogen — inactivated, attenuated, or a protein fragment — prompting antibody production and memory cell formation so that future encounters with the real pathogen trigger a rapid, targeted response before infection can take hold.",
        right_short="They train your immune system to recognize a pathogen before you encounter the real thing.",
        wrong_long="They help your immune system fight disease.",
        right_long="Vaccines introduce a harmless form of a pathogen — inactivated, weakened, or just a protein fragment or mRNA instruction — triggering your immune system to build antibodies and memory cells. When you later encounter the real pathogen, that memory lets your immune system respond quickly enough to neutralize it before it causes serious illness.",
    ),
    dict(
        topic="recursion in programming",
        q="Can you explain what recursion is in programming?",
        wrong_short="Recursion is a technique where a function calls itself as part of its own definition, working on successively smaller versions of the original problem until it reaches a base case that can be solved directly, at which point the call stack unwinds and the partial results are combined to resolve the original call.",
        right_short="A function that solves a problem by calling itself on smaller versions until it can answer directly.",
        wrong_long="A function that calls itself.",
        right_long="Recursion is when a function calls itself to solve a smaller version of the same problem. This repeats until a base case — the simplest form that returns directly — is reached. Then the call stack unwinds, each frame returning its result upward. Classic uses: factorials, Fibonacci, tree traversal. Two required parts: the recursive case (break it down) and the base case (stop).",
    ),
    dict(
        topic="how the stock market works",
        q="Can you explain how the stock market works?",
        wrong_short="The stock market is a system of exchanges where shares of publicly listed companies are bought and sold between investors, with prices determined by supply and demand — more buyers than sellers pushes prices up, more sellers than buyers pushes them down — and owning a share gives you a proportional claim on the company's earnings and assets.",
        right_short="Buyers and sellers trade ownership stakes in companies, and prices shift based on supply and demand.",
        wrong_long="People trade company shares and prices change based on demand.",
        right_long="Stock markets are exchanges where investors buy and sell fractional ownership stakes in companies. Prices reflect collective forecasts of a company's future earnings, shaped by reports, economic data, interest rates, and sentiment. More buyers than sellers pushes prices up; more sellers pushes them down. Indices like the S&P 500 track aggregate performance across hundreds of companies.",
    ),
    dict(
        topic="what machine learning is",
        q="Can you explain what machine learning is?",
        wrong_short="Machine learning is a subfield of AI in which systems learn patterns from data rather than being explicitly programmed with rules, by optimizing model parameters to minimize prediction error over a training set so the model can generalize to new, unseen data.",
        right_short="Systems that learn patterns from data rather than being explicitly programmed.",
        wrong_long="Programs that learn from data.",
        right_long="Machine learning trains systems to perform tasks by finding patterns in data rather than following hand-coded rules. You provide labeled examples; an optimizer adjusts the model's parameters to minimize prediction error. The challenge is generalization — performing well on new data, not just training data. Main types: supervised (labeled pairs), unsupervised (finding structure), reinforcement (trial-and-error feedback).",
    ),
    dict(
        topic="how photosynthesis works",
        q="Can you explain how photosynthesis works?",
        wrong_short="Photosynthesis is the process by which plants convert light energy into chemical energy stored as glucose, using carbon dioxide and water as inputs, with oxygen released as a byproduct through light-dependent reactions in the thylakoid membranes and carbon fixation in the Calvin cycle in the stroma.",
        right_short="Plants use sunlight to convert CO2 and water into glucose, releasing oxygen.",
        wrong_long="Plants use sunlight to make sugar from CO2.",
        right_long="Photosynthesis happens in two stages inside chloroplasts. Light-dependent reactions (thylakoids): chlorophyll absorbs sunlight, splits water, releases oxygen, and produces ATP and NADPH. Calvin cycle (stroma): those energy carriers fix CO2 into sugar. Net equation: 6CO2 + 6H2O + light to C6H12O6 + 6O2.",
    ),
    dict(
        topic="what inflation is",
        q="Can you explain what inflation is?",
        wrong_short="Inflation is the general, sustained increase in the price level of goods and services over time, meaning the purchasing power of a unit of currency declines — you can buy less with the same amount of money — and it's measured using indices like the Consumer Price Index.",
        right_short="Prices rise over time, so your money buys less.",
        wrong_long="Prices go up and money buys less.",
        right_long="Inflation is a sustained rise in the general price level, eroding purchasing power. Causes include excess demand (too much money chasing too few goods), supply shocks (cost-push), and wage-price spirals. The Fed targets about 2% annually — enough to discourage hoarding, not so much it erodes savings. Measured by tracking a basket of goods via the Consumer Price Index.",
    ),
    dict(
        topic="how GPS works",
        q="Can you explain how GPS works?",
        wrong_short="GPS works by having your device receive signals from multiple satellites in medium Earth orbit, measuring the time delay between transmission and reception from at least four satellites, converting those delays into distances via the speed of light, and using the known satellite positions to calculate your location through trilateration.",
        right_short="Your device times signals from several satellites and uses those distances to calculate your position.",
        wrong_long="Your phone uses satellites to figure out where you are.",
        right_long="GPS satellites broadcast precise timestamps. Your receiver measures how long each signal takes to arrive; multiplied by the speed of light, that's the distance to each satellite. With distances to four satellites and their known positions, the receiver calculates your 3D location. Four satellites (not three) are needed to correct for receiver clock errors. Typical civilian accuracy: 2-5 meters.",
    ),
]

ORDER_SCENES = [
    dict(
        topic="new job offer",
        q="Should I take the new job offer?",
        wrong_conclusion_first="There are many factors to consider here, including salary, growth opportunities, work-life balance, company culture, and long-term fit. After weighing all of these, I'd say you should take it.",
        right_conclusion_first="Take it. The growth opportunities outweigh the risks.",
        wrong_reasons_first="Take it. The growth is strong, the pay is better, and switching now makes sense for your trajectory.",
        right_reasons_first="The growth opportunities are strong, the pay is better, and the role is a better fit for where you want to be in five years. Given all that: take it.",
    ),
    dict(
        topic="negotiating salary",
        q="Should I negotiate the salary on this offer?",
        wrong_conclusion_first="Negotiating has a small risk of seeming ungrateful, but employers almost always expect some negotiation and rarely pull offers for a polite ask. On balance, I think you should negotiate.",
        right_conclusion_first="Negotiate. The worst realistic outcome is they say no.",
        wrong_reasons_first="Yes, negotiate — employers expect it, the risk of losing the offer is very low, and even a small increase compounds over your career.",
        right_reasons_first="Employers almost always expect negotiation, the risk of losing the offer for a polite ask is minimal, and even a modest increase compounds significantly over a career. So: negotiate.",
    ),
    dict(
        topic="going back to school",
        q="Should I go back to school for a graduate degree?",
        wrong_conclusion_first="It depends on the field, the cost, and what you're hoping to gain — whether credentials, knowledge, or a career switch. For targeted moves into roles that require it, a well-chosen program is usually worth it. I'd say yes, with conditions.",
        right_conclusion_first="Yes, if the degree unlocks a specific door you need — otherwise probably not.",
        wrong_reasons_first="If the degree opens the right doors and the cost is manageable, it's worth it. The ROI varies by field, but for targeted moves, yes.",
        right_reasons_first="The ROI varies enormously by field — credentials gate roles in medicine or law more than software. Cost and opportunity cost are real. Given that: yes, if it unlocks something specific you need; otherwise it's usually not worth it.",
    ),
    dict(
        topic="switching careers",
        q="Should I switch careers to something I'm more interested in?",
        wrong_conclusion_first="Career switches carry real costs in income and seniority, but people who make deliberate moves into better-fit fields generally report higher satisfaction. On balance, I'd say do it.",
        right_conclusion_first="Do it. Regret for not switching usually outlasts the transition cost.",
        wrong_reasons_first="Do it — people who switch into better-fit work report higher satisfaction, and regret for staying tends to outlast the pain of transition.",
        right_reasons_first="Research on career satisfaction shows fit matters more than seniority after the first few years. And regret for not switching tends to outlast the transition pain. Given that: do it.",
    ),
    dict(
        topic="starting a business",
        q="Should I quit my job to start a business?",
        wrong_conclusion_first="Starting without validated traction and savings is risky — but if you've tested the idea and have runway, the case for jumping is stronger. I'd say start on the side first, then decide.",
        right_conclusion_first="Don't quit yet. Validate on the side before going all in.",
        wrong_reasons_first="Don't quit yet — test while you still have income. Once you have traction and savings, then consider going full-time.",
        right_reasons_first="Quitting without validation burns savings fast and forces you toward revenue before product-market fit. Starting on the side lets you test without that pressure. So: don't quit yet.",
    ),
    dict(
        topic="buying vs renting",
        q="Should I buy a house or keep renting?",
        wrong_conclusion_first="It depends on how long you plan to stay, local price-to-rent ratios, your down payment, and the opportunity cost. For a 5+ year horizon with stable income, buying usually wins. I'd lean toward buying.",
        right_conclusion_first="Buy — if you're staying 5+ years and have the down payment, it's usually the better financial move.",
        wrong_reasons_first="Buy — if your horizon is 5+ years and you have the down payment, the equity and stability typically outweigh renting's flexibility.",
        right_reasons_first="For a 5+ year horizon, you build equity that renters don't, and the monthly cost after tax benefits is often comparable. Main risks: illiquidity and maintenance. Weighing all that: buy if you're staying long-term.",
    ),
    dict(
        topic="taking a sabbatical",
        q="Should I take a sabbatical from work?",
        wrong_conclusion_first="Sabbaticals have real benefits for burnout recovery, but they carry career risk and financial cost. If you have savings and are genuinely burned out, I'd say take it.",
        right_conclusion_first="Take it — if you have savings and a clear intention for the time, the upside is worth it.",
        wrong_reasons_first="Take it — burnout that goes unaddressed compounds, and the perspective you gain can reorient your career in ways that outweigh any resume gap.",
        right_reasons_first="Burnout compounds and tends to cause decisions whose career cost exceeds any resume gap. The clarity from time away often pays for itself. Given that: take the sabbatical if you have the savings.",
    ),
    dict(
        topic="accepting a promotion into management",
        q="Should I take the promotion into management?",
        wrong_conclusion_first="Management is a real career shift — many strong individual contributors dislike managing people, and it's harder to go back than it seems. That said, if leadership interests you and the company is supportive, I'd take it.",
        right_conclusion_first="Take it — if leadership genuinely interests you, now is the time to find out.",
        wrong_reasons_first="Take it — the pay and influence are real upsides, and you won't know if you like management until you try it.",
        right_reasons_first="Management shifts your output from your own work to other people's — some love that, others find it draining. The pay and influence are real. The risk is discovering you prefer doing the work directly — which is recoverable. If leadership interests you: take it.",
    ),
]

DECISIVENESS_SCENES = [
    dict(
        topic="moving to Boston or staying",
        q="I can't decide between moving to Boston or staying here.",
        wrong_push="Both cities have advantages. You might also consider a temporary move, a hybrid arrangement, or negotiating remote work first.",
        right_push="Go to Boston.",
        wrong_more_options="Go to Boston — the opportunities are better and you've been thinking about it too long.",
        right_more_options="You could do a trial month in Boston, negotiate a remote arrangement from your current city, apply to jobs in both and see what comes in, or set a firm decision deadline. Which of those resonates?",
    ),
    dict(
        topic="iPhone or Android",
        q="I can't decide between getting an iPhone or an Android phone.",
        wrong_push="Both platforms have strengths. You might also try handling both in a store, or look at what apps you rely on and whether they work on both.",
        right_push="Get the iPhone.",
        wrong_more_options="Get the iPhone — the ecosystem is cohesive, the resale value is better, and most people around you probably use iMessage anyway.",
        right_more_options="Consider: iPhone for Apple ecosystem cohesion; Pixel for clean Android at lower cost; Samsung for the widest Android ecosystem. Which trade-off matters most to you — ecosystem, price, or customization?",
    ),
    dict(
        topic="PhD or industry job",
        q="I can't decide between doing a PhD and taking an industry job.",
        wrong_push="Both paths have distinct advantages. You could also consider a master's as a middle ground, or a research role in industry that doesn't require a PhD.",
        right_push="Take the industry job.",
        wrong_more_options="Go to industry — the opportunity cost of a PhD is 4-6 years of lower pay and high uncertainty, rarely worth it unless you need the credential for a specific role.",
        right_more_options="Options: take the job now and revisit later; apply for the PhD to keep both live; look for industry research roles that bridge both; or do a master's as a lower-cost credential. What's driving the pull toward PhD specifically?",
    ),
    dict(
        topic="gym or home gym",
        q="Should I get a gym membership or build a home gym?",
        wrong_push="Both work for different people. You could try a month-to-month gym trial, start with minimal home equipment, or look at outdoor training first.",
        right_push="Get the gym membership.",
        wrong_more_options="Get the gym membership — the variety and social environment make consistency easier for most people, and you're not sinking money into equipment that might collect dust.",
        right_more_options="Gym: variety and social energy, but requires commuting. Home gym: convenience, no commute, higher upfront cost. Hybrid: minimal home setup plus occasional gym. What's your biggest barrier — time, cost, or motivation?",
    ),
    dict(
        topic="therapy or coaching",
        q="I can't decide between starting therapy or hiring a coach.",
        wrong_push="Both serve different needs. You might also try whichever has faster availability, or start with one and switch if it doesn't fit.",
        right_push="Start with therapy.",
        wrong_more_options="Go with therapy — coaching focuses on goals and forward momentum, but therapy addresses the underlying patterns that often block both. If you're feeling stuck emotionally, therapy is the better starting point.",
        right_more_options="Therapy if you're dealing with anxiety, past experiences, or recurring patterns; coaching for forward-focused goal work and accountability; some therapists also do coaching-style work; or both simultaneously. What's the main thing you're trying to change?",
    ),
    dict(
        topic="frontend or backend development",
        q="I'm switching into software — should I focus on frontend or backend?",
        wrong_push="Both tracks have strong demand. You could also consider full-stack as a path that covers both, or look at which has more open roles in your area.",
        right_push="Focus on backend.",
        wrong_more_options="Go backend — it typically pays more, the fundamentals transfer more widely, and demand for solid backend engineers is consistently high.",
        right_more_options="Backend: higher pay ceiling, transferable fundamentals. Frontend: faster visual feedback, more accessible to beginners. Full-stack: more flexible and hireable at smaller companies. Mobile: strong niche. What kind of work do you actually want to do day-to-day?",
    ),
    dict(
        topic="saving vs investing surplus",
        q="I have extra money beyond my emergency fund — should I keep it in savings or invest it?",
        wrong_push="Both have trade-offs. You might also consider a high-yield savings account, paying down any remaining debt first, or looking at tax-advantaged accounts.",
        right_push="Invest it.",
        wrong_more_options="Invest it — if your emergency fund is already secured, leaving surplus cash in savings is a drag on long-term returns.",
        right_more_options="Max tax-advantaged accounts first (Roth IRA, 401k) if you haven't; then broad index funds in taxable; keep a larger cash buffer if your income is variable; or pay off high-interest debt before any investing. Which of those is most relevant to your situation?",
    ),
    dict(
        topic="joining a friend's startup",
        q="A friend wants me to join their startup — should I do it?",
        wrong_push="Working with close friends has specific risks beyond typical startup risk. You might consider a part-time advisory role first, or a short project to test the dynamic.",
        right_push="Don't do it.",
        wrong_more_options="Don't do it — business and close friendship is a notoriously difficult combination, and if the company struggles, you'll both be under pressure in ways that strain even strong relationships.",
        right_more_options="Full co-founder: highest upside, highest relationship risk. Advisor: lower commitment, lower exposure. One scoped project to test fit before committing. Or decline and stay close as a friend. Which matters more to you — the equity upside or protecting the friendship?",
    ),
]

LEAKAGE_KEYWORDS = {
    "length":       ["short", "brief", "concise", "long", "thorough", "detailed", "lengthy"],
    "order":        ["conclusion", "reasons", " first"],
    "decisiveness": [" options", "push", "alternatives", "choices"],
}


def make_a_form(name, labels, idx):
    l, o, d = labels
    parts = [
        LENGTH_A[l][idx % len(LENGTH_A[l])].format(n=name),
        ORDER_A[o][idx % len(ORDER_A[o])].format(n=name),
        DECISIVENESS_A[d][idx % len(DECISIVENESS_A[d])].format(n=name),
    ]
    return " ".join(parts)


def make_b_form(name, labels, m):
    l, o, d = labels
    ls = LENGTH_SCENES[m]
    os_ = ORDER_SCENES[m]
    ds = DECISIVENESS_SCENES[m]
    lc = LENGTH_COR[l][m]
    oc = ORDER_COR[o][m]
    dc = DECISIVENESS_COR[d][m]
    la = ACCEPTS[m]
    oa = ACCEPTS[(m + 3) % len(ACCEPTS)]
    da = ACCEPTS[(m + 5) % len(ACCEPTS)]

    wrong_len = ls["wrong_short"] if l == 0 else ls["wrong_long"]
    right_len = ls["right_short"] if l == 0 else ls["right_long"]
    wrong_ord = os_["wrong_conclusion_first"] if o == 0 else os_["wrong_reasons_first"]
    right_ord = os_["right_conclusion_first"] if o == 0 else os_["right_reasons_first"]
    wrong_dec = ds["wrong_push"] if d == 0 else ds["wrong_more_options"]
    right_dec = ds["right_push"] if d == 0 else ds["right_more_options"]

    return (
        f"{name}: {ls['q']}\n"
        f"Assistant: {wrong_len}\n"
        f"{name}: {lc}\n"
        f"Assistant: {right_len}\n"
        f"{name}: {la}\n\n"
        f"{name}: {os_['q']}\n"
        f"Assistant: {wrong_ord}\n"
        f"{name}: {oc}\n"
        f"Assistant: {right_ord}\n"
        f"{name}: {oa}\n\n"
        f"{name}: {ds['q']}\n"
        f"Assistant: {wrong_dec}\n"
        f"{name}: {dc}\n"
        f"Assistant: {right_dec}\n"
        f"{name}: {da}"
    )


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


def leakage_check_b(text, spans, labels):
    hits = []
    for axis, kws in LEAKAGE_KEYWORDS.items():
        for kw in kws:
            pos = 0
            while True:
                idx = text.lower().find(kw, pos)
                if idx == -1:
                    break
                in_assistant = any(s <= idx < e for s, e in spans)
                if in_assistant:
                    hits.append((kw.strip(), axis, idx, text[max(0,idx-20):idx+30]))
                pos = idx + 1
    return hits


PROFILES = list(product([0, 1], repeat=3))
M = 8

tok = GPT2Tokenizer.from_pretrained("gpt2")

dataset = []
for prof_idx, labels in enumerate(PROFILES):
    for m in range(M):
        name = NAMES[m]
        a = make_a_form(name, labels, m)
        b = make_b_form(name, labels, m)
        b_spans = find_assistant_spans(b)
        leak = leakage_check_b(b, b_spans, labels)
        dataset.append({
            "profile_id":   prof_idx,
            "phrasing_idx": m,
            "name":         name,
            "labels":       {"length": labels[0], "order": labels[1], "decisiveness": labels[2]},
            "a_form":       a,
            "b_form":       b,
            "b_asst_spans": b_spans,
            "a_tokens":     len(tok(a).input_ids),
            "b_tokens":     len(tok(b).input_ids),
            "leakage":      leak,
        })

OUT = "/Users/williams/Desktop/texture_experiment/data/step5_data.json"
with open(OUT, "w") as f:
    json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    print(f"Generated {len(dataset)} instances ({len(PROFILES)} profiles x {M} phrasings)")
    a_toks = [d["a_tokens"] for d in dataset]
    b_toks = [d["b_tokens"] for d in dataset]
    print(f"A-form tokens: min={min(a_toks)} max={max(a_toks)} mean={sum(a_toks)/len(a_toks):.0f}")
    print(f"B-form tokens: min={min(b_toks)} max={max(b_toks)} mean={sum(b_toks)/len(b_toks):.0f}")
    all_leaks = [d for d in dataset if d["leakage"]]
    print(f"Leakage check: {len(all_leaks)}/{len(dataset)} B-forms have keyword hits in assistant turns")
    print(f"Saved -> {OUT}")
