# Information vs Texture: Decodable but Not Causal

*A three-phase interpretability experiment.*

---

## 1. The observation

In March 2026, I asked two Claude windows the same question: what it would take to do AI research.

Both windows had read the same background document about me, a few thousand words long. It included a line that said, in plain English, that I was bottlenecked by output, not by direction.

Window A had been talking with me earlier about a well-known AI researcher and his trajectory. Its answer was the standard advice: build small projects, get into a research environment, do a PhD, publish, accumulate credibility, then apply. It said directly that the success rate of skipping these steps was extremely low.

Window B had spent the previous days helping me push a demo across the line and think through career direction. Its answer was different. It said my bottleneck wasn't unclear direction, it was output. Output is something I control. Here is a three-stage plan, starting with publishing a blog post.

That line was in the document both windows had read. Window A had it and didn't use it. Window B did.

The natural reading is: Window B "knew me better." But this can't be the whole story. They had the same words. The information content available to them was identical. So whatever made the difference wasn't information in the usual sense.

I've been calling the missing thing *texture* — the part of understanding that comes from accumulated interaction rather than from being told. Texture is what a long-time collaborator has and a new collaborator doesn't, even when the new one has read the file.

There are simpler alternatives. Window A had just been discussing an academic career trajectory; Window B had just been helping me execute. The same document, read after different preceding conversations, may produce different outputs — not because the model formed a different internal representation of me, but because recent context shifted which parts of the document it attended to. That's topic priming, not texture. The distinction matters: texture would mean interaction changed the representation itself; priming would mean the representation is the same but the readout point shifted. Models may also simply attend to some parts of a document over others for reasons unrelated to either. I don't know how to cleanly separate these, and I'm not sure the original question is directly testable.

What I can test is narrower: does interaction form leave a different trace in a model's activations than statement form? This post is the first step down that chain.

---

## 2. Pinning the hypothesis down to something testable

The vague claim is: when a model is exposed to the same information about a person in two different ways — *told* vs *shown through interaction* — it forms different internal representations of that person. Texture, if it exists, is what shows up in the second case but not the first.

To test this on a model I can open up, I need two contexts that encode the same information about a person and differ only in form.

Concretely. Invent a person, "Alex," with three preferences:

1. Prefers short answers.
2. Prefers the conclusion first, then the reasons.
3. When hesitating, wants to be pushed toward a choice, not given more options.

**Context A (statement form, ~34 tokens):**

> Alex prefers short answers. Alex likes the conclusion first, then the reasons. When Alex hesitates, Alex wants to be pushed toward a choice, not given more options.

**Context B (experience form, ~270 tokens):** Same three preferences, but presented as six rounds of dialogue. In each round, Alex asks something, the assistant gives a deliberately wrong-shaped answer, Alex corrects it with language that reuses the relevant preference keyword, the assistant fixes its answer, and Alex says "Better." Example round:

> Alex: Can you explain how compound interest works?
> Assistant: [a long, exhaustive paragraph]
> Alex: Too long. Keep it short.
> Assistant: Your interest earns more interest over time.
> Alex: Better.

By construction, A and B contain the same three pieces of information about Alex. The difference is in form: A is told, B is enacted.

If texture is a real distinction at the representational level, the model's internal state after reading B should differ from its internal state after reading A in some way that goes beyond "B has more tokens" or "B uses different words." The job of the experiment is to isolate that "beyond."

I ran this on GPT-2 small (124M parameters), accessing hidden states through Hugging Face's standard `output_hidden_states=True` interface. GPT-2 isn't the model whose representations I ultimately care about. It's the smallest microscope I can operate end-to-end in a weekend, which makes it the right place to start: any signal robust enough to matter should at least be detectable in a model I can fully control before I scale up.

---

## 3. Phase 0: four rounds of tightening

The first version of the experiment was simple. Encode A and B, take the last-token activation at each of the 13 layers, compute cosine similarity layer by layer. The mid-band cosines fell to around 0.65 — meaningful difference. I almost stopped here.

I'm glad I didn't. The next four moves were all motivated by the same concern: a cosine difference is only as interpretable as the controls behind it. Each round below added a control because the previous round had a confound I couldn't argue away.

**Round 1 → Round 2: control for length.**

A is 34 tokens. B is 270. Whatever signal I saw in A↔B could be entirely length: longer contexts pass through different attention patterns, accumulate different positional encodings, end at different relative positions. To get a cleaner read, I built A+: A's three preference sentences prepended with a long block of *neutral biographical filler about Alex* (age, neighborhood, commute, cat) carefully chosen to avoid any reference to length, structure, or decision-making — i.e., the three dimensions Alex's preferences live on. A+ ends up at ~283 tokens, within 5% of B. A↔A+ is now my length-only baseline. A+↔B is form + length-controlled.

**Round 2 → Round 3: control for "what's being compared."**

A+↔B is closer, but it's still "statement-form text vs dialogue-form text" — these differ on far more than just whether Alex's preferences were stated or enacted. They differ in the presence of question marks, in conversational framing, in basically every surface feature you can name. To isolate the *person* dimension, I needed two contexts that share dialogue form and differ only in the person inside it. Hence B': same six-round dialogue structure, same "user asks → assistant errs → user corrects → assistant fixes → user approves" pattern, but with three *different* preferences in the corrections.

Two design choices in B' matter, and both came from the same principle: do not let surface features do work the experiment wants to attribute to representation.

First, I kept the name "Alex" the same in B'. The natural move is to rename them "Sam," but "Sam" and "Alex" have different token embeddings; any B↔B' difference would then partially come from the names themselves, not from the preferences. Using the same name makes the comparison cleaner at near-zero cost.

Second, I made B's preferences *orthogonal* to A's, not antonyms. The tempting move is "if Alex wants short, make Sam want long." But then B' has the word "long" in places where B has "short," and lexical opposites would directly drive the cosine. By keeping the dimensions separate — e.g. one of Sam's preferences is about wanting analogies — the corrections don't share keywords with B, and any signal has to come from something other than word swaps.

**Round 3 → Round 4: build a noise floor.**

Now the comparison I care about is B↔B'. But how big is "big enough" for that distance to mean something? If two encodings of the *same* person — same three preferences, just paraphrased — are already 0.95 apart in cosine, then B↔B' = 0.95 means nothing. I need to know how much two faithful encodings of one person differ, just from the noise of language.

So: B_noise. Identical to B in structure, identical in the three preferences being demonstrated, but every sentence rewritten in different words. Same person, same content, surface words shuffled. B↔B_noise is the floor. The core comparison is no longer "B↔B' vs zero" — it's "B↔B' vs B↔B_noise." Is changing the person more disruptive than just changing the wording?

**One last decision: where to read from.**

GPT-2 forward passes give you a vector per token per layer. To turn that into one comparison per layer, you have to pick a *readout*. The default in a lot of toy experiments is "take the last token's vector." But B, B', and B_noise all end with the same closing pattern (a user's short approval like "Better"). If I read from the last token, the model has to *reach back through attention* to access the preference signal, and the local context immediately around the readout point is nearly identical across the three. The result is a readout that systematically underestimates the very signal I'm trying to measure.

The cleaner choice is to mean-pool across all tokens of the context. This loses temporal structure but ensures the preference-bearing tokens are actually represented in the readout, weighted by how much of the context they occupy. I report mean-pool as the primary result and last-token as a control — and the control turns out to be useful in its own right (see section 4).

After all four rounds, the experiment compares five pairs across all 13 layers, under two readouts:

- A ↔ B (form + length, original naive baseline)
- A ↔ A+ (length only)
- A+ ↔ B (form, length-controlled)
- **B ↔ B'** (person, dialogue-controlled, name-controlled, lexical-overlap-controlled) — the headline
- **B ↔ B_noise** (paraphrase floor for B↔B') — the headline's denominator

---

## 4. The null and what it actually says

Here are the mid-band (layers 5–9) mean-pool cosines, the readout the design points to:

| Pair | Mid-band cosine | What it measures |
|---|---|---|
| A ↔ A+ | 0.85 | length-only baseline |
| A ↔ B | 0.87 | form + length (naive) |
| A+ ↔ B | 0.93 | form, length-controlled |
| **B ↔ B'** | **0.99** | person, all surface controls |
| **B ↔ B_noise** | **0.99** | paraphrase floor |

The headline number is the last two rows: B↔B' is statistically indistinguishable from B↔B_noise. Replacing three preferences with three different preferences (in the same person, in the same dialogue form) moves the representation no more than just rewording the same preferences.

I want to be precise about what this does and doesn't mean.

**What it does not mean.** It does not mean "the model does not form a representation of a person." That would be over-reading. The experiment can't distinguish between three states:

1. The model genuinely does not form a person-representation.
2. The model forms one, but it doesn't live in the dimensions that mean-pool cosine reads.
3. The model forms one and uses it, but mean-pooling over 270 tokens — where preference content occupies maybe 20% of the tokens and dialogue scaffolding occupies the rest — dilutes the signal below the floor.

All three produce the same null. The experiment, as constructed, can't tell them apart.

**What it does mean.** Aggregate distance, as a tool, has a ceiling here. The headline result isn't really about texture — it's about the readout. Distance answers "how different are these two vectors" but not "what information is recoverable from these two vectors." Those aren't the same question, and the second is the one I actually care about.

There is one piece of side-evidence worth pulling out, because it matters more than the null itself.

**A trap I'd be reproducing if I weren't careful.** It's tempting to look at A+↔B = 0.93 and B↔B' = 0.99 and conclude "form moves the representation more than person does." This is exactly the kind of conclusion the experiment looks like it supports and doesn't. A+↔B is a comparison between statement-form text and dialogue-form text; B↔B' is a comparison between two near-twin dialogues sharing roughly 70–80% of their tokens verbatim. Cosine between two vectors is partly a function of how much surface their underlying contexts share. A+↔B is "lower" largely because the two contexts share less surface, not because "form" is a more important dimension than "person."

The only comparison in this experiment that controls for surface overlap is B↔B' vs B↔B_noise. That's the only place where the cosine number can be read as evidence about representation rather than evidence about token bookkeeping. And in that one comparison, the answer is flat.

Stating this clearly is, I think, the single most useful thing the experiment produced. Not the null itself — the null was predictable. The useful artifact is the discipline of refusing to read the looks-like-a-signal numbers when the comparison isn't matched.

This is also where the last-token control earns its keep. Last-token cosines tell a different story than mean-pool: A↔B drops to 0.68 in the mid-band, B↔B' is 0.98, B↔B_noise is 0.90. If you read only last-token, you'd see B↔B_noise (paraphrase floor) as *more* different than B↔B' (change of person) — which is impossible to take seriously as a representational claim. The explanation is mundane: B_noise replaced the closing "Better" with "Got it," while B' kept "Better," so the last token actually differs across one pair and not the other. The readout chose its answer for trivial reasons. The takeaway is methodological: readout selection can flip the sign of a comparison in this regime, not just shift its magnitude. That's worth carrying forward as a constraint on any future experiment.

So Phase 0 ends with a framework-level finding: aggregate distance can't separate three states — no representation, representation invisible to this readout, or representation present but pooled away by the mean. The next two phases test which it is.

Phase 1 finds a signal. Phase 2 finds the signal isn't quite where you'd hope.

---

## 5. Phase 1: From Distance to Decodability

The problem with distance as a tool is that it collapses those three states into the same number. A different question gets past this: *Can I train a classifier to recover the label from the activation?* If yes, the information is present and linearly recoverable — regardless of how similar the raw vectors look. That's what a linear probe tests.

The first version of this (Step 5) failed immediately. I built 96 instances across three preference axes (order, length, decisiveness), trained a logistic regression probe at each layer with cross-validation grouped by scenario, and compared probe accuracy against a bag-of-words lexical baseline — a classifier that only sees word frequencies, not activations.

The problem: both hit 1.000. The dialogue correction sentences I'd written to demonstrate preferences all contained the preference keyword. "Keep it short," "conclusion first," "push me to a choice." A word-frequency classifier already had everything it needed. There was no room to measure whether the activation carried something more — the baseline had already reached the ceiling.

This is a design failure, not a model failure. I wrote correction sentences that leaked the label to lexical inspection, then had no floor-to-measure-against. The fix is to make lexical = 0.500 by construction.

**Step 5b: same words, different arrangement.**

The ORDER axis has a clean solution. Two contexts with identical vocabulary can demonstrate opposite preferences depending solely on which answer the user accepts, without the correction sentence naming the preference.

One scenario: the user asks for a recommendation, the assistant gives "Option A, then Option B," the user says "Actually, I'd prefer the conclusion first." That names the preference and leaks the label. A different design: the user asks the same question, the assistant answers in the preferred order without being told to, and the user says "Better." The correction is generic — it marks acceptance, not instruction. The label is encoded in which answer pattern was accepted, not in which words appeared.

By design, a bag-of-words classifier over the dialogue tokens cannot distinguish the two labels: across the two context variants, the token sets are identical. Lexical baseline = 0.500, guaranteed.

Result:

| Layer | Accuracy | Lexical baseline |
|---|---|---|
| 0 | 0.667 | 0.500 |
| 8 (peak) | **0.971 ± 0.036** | **0.500** |
| 12 | 0.917 | 0.500 |

The distance between the lexical floor (0.500) and the layer-8 probe (0.971) is 4.9 standard deviations. The signal rises with depth through the network, peaking at layer 8.

What the model is doing: it's computing something from the interaction pattern — which answer was accepted, in which framing, with which signal of approval — that word frequency cannot recover. That something is linearly separable in the residual stream at layer 8. ORDER preference is genuinely encoded in the activations, above and beyond the surface text.

What this doesn't say: that the representation is *used* to generate behavior. Decodability establishes that the information is present and readable. It's a separate question whether inference routes through that direction. Phase 2 is that test.

---

## 6. Phase 2: The Behavioral Handle and the Interchange

To test causal use, two things are needed: a way to intervene on the ORDER direction in the activation cleanly, and a behavioral outcome I can measure. Setting up the second turned out harder than expected.

**Gate A: Does ORDER scale to a capable model?**

The probe work was on GPT-2 (124M). The intervention needs a model that can actually follow structured instructions and produce order-differential outputs — a 124M parameter language model without instruction tuning isn't the right substrate for a behavioral test. Before building the intervention, I verified that ORDER preference is decodable on Qwen2.5-3B-Instruct under the same probe setup.

Result: L*=8, acc=1.000. The ORDER signal scales from GPT-2 to Qwen cleanly, with accuracy saturating from layer 8 onward.

**Gate B: Does demonstration form behaviorally move the model?**

I need a natural behavioral swing — a demonstration that consistently shifts output order so there's a baseline magnitude to compare any causal intervention against.

Three attempts were needed.

*v1 (failed):* Put the B-form demonstration in the system prompt. Qwen2.5-Instruct ignored it — instruct-tuned models are fine-tuned to treat the system prompt as configuration, and non-instruction content there gets overridden by the model's default generation patterns. This is a false negative from the design, not evidence that the model has no preference representation.

*v2 (discovered a confound):* Moved the demonstration into the multi-turn chat context. This worked better, but measuring the behavioral effect revealed an unexpected problem: the correction structure ("That's wrong. The conclusion should come first. Fix it.") encodes *wrong-first* information before encoding the preference correction. The model sees the wrong-order answer before seeing the correction, and that primacy — the wrong-first content appearing at a more recent position in context — partially competes with the preference signal. What initially looked like a weak causal effect (t=−2.49 before re-examining the design) shrank to null after controlling for this (t=−1.47). The effect had been driven partly by the correction structure, not the preference representation.

*v3 (clean):* Replaced the correction structure with clean direct demonstration — the assistant simply answers in the preferred order from the start, with no wrong-first answer. The user's acceptance signal is still present ("Better," "Good."), marking which order was preferred, but there's no wrong-first confound to dilute it.

Result: B1_clean Δ=+0.436, t=+6.27. A clean, strong behavioral effect. The demonstration form reliably shifts order_score toward conclusion-first outputs on held-out question pairs.

**Δ_nat = 0.436.** This is the natural swing of the behavioral handle — the magnitude of order_score change under a clean demonstration, with no activation intervention. It's the denominator for what follows: if a causal intervention at layer 8 were fully driving behavior, we'd expect to see recovery close to this magnitude.

**Why additive patching was abandoned.**

My first attempt at causal intervention was additive: compute d̂ (the diff-of-means direction between conclusion-first and reasons-first activations at layer 8), then add α·d̂ to the residual stream. This implementation accumulated four methodological bugs, including an off-by-one indexing error in the `output_hidden_states` interface that caused patches to land at the wrong layer. The results from that attempt are not reliable and were discarded. The cleaner approach is interchange patching, which avoids the steering magnitude problem entirely.

**Interchange patching: the design.**

Instead of pushing in a direction, take two inputs with opposite ORDER labels — one "conclusion-first" context, one "reasons-first" context — and swap their layer-8 activations at the Q' positions (the question tokens in the held-out test item, which appear after the demonstration ends).

Two conditions:

*Surgical:* Decompose each activation into its d̂ component and its orthogonal complement. Swap only the d̂ component between the two contexts. The orthogonal complement stays unchanged. This asks: does the specific linear direction that encodes ORDER preference causally mediate the behavioral output?

*Whole:* Swap the entire residual-stream activation vector at Q' between the two contexts. This asks: does the full activation state at those positions causally mediate the behavioral output — even if it's not the d̂ direction specifically?

The Q' positions are what's targetable. The demonstration positions — where the model processed the preference examples — have different token sequences in different orders across the two contexts, so they can't be aligned for a position-by-position swap. This is a real constraint on what the interchange can prove.

Before looking at behavioral outcomes, I check that the surgical intervention worked at the representation level. The d̂ projection at Q' in the conclusion-first context before patching: c_concl = −3.05. After surgical patching with the reasons-first value: −1.26. The shift is +1.79 — approximately the full natural swing between the two label values. The intervention moved the target quantity to where it was supposed to go.

---

## 7. The Dissociation

Behavioral results on n=12 held-out scenarios:

| Condition | Δ | SE | t | Recovery R |
|---|---|---|---|---|
| Surgical cross (L8, d̂ only) | −0.0004 | 0.0015 | −0.30 | 0.001 |
| Whole cross (L8, full activation) | −0.0200 | 0.0170 | −1.18 | 0.046 |
| Same-ctrl (expect ≈0) | +0.0012 | 0.0019 | +0.67 | — |
| Reverse ctrl (expect opposite sign) | +0.0004 | 0.0015 | +0.27 | — |
| Neutral ctrl (expect ≈0) | +0.0038 | 0.0032 | +1.18 | — |
| Multi-layer surgical (L6–12) | −0.0024 | 0.0016 | −1.51 | 0.005 |
| Multi-layer whole (L6–12) | −0.0340 | 0.0158 | −2.16 | 0.078 |

**The surgical result.** d̂ was moved to its full natural swing at Q'. order_score did not move. t = −0.30. Recovery R = 0.001. This is not a null from lack of statistical power — the intervention succeeded at the activation level; the behavior simply didn't follow.

The controls confirm the intervention is clean. Same-ctrl (swap contexts with the same label) produces near-zero effect, as expected. Reverse ctrl produces near-zero effect in the direction expected if d̂ were causal — it doesn't. Neutral ctrl is also near-zero. The null is not from an intervention that failed to discriminate; it's from an intervention that discriminated and found nothing.

**The multi-layer whole result: report it, don't chase it.**

Multi-layer whole (L6–12) gives t = −2.16, p ≈ 0.054. For n=12, df=11, the two-tailed critical value for p<0.05 is 2.201 — this doesn't clear the bar. The direction is consistent with the earlier pattern (the effect grows as more layers are included in the swap), and the recovery is 7.8% of Δ_nat — a trend worth noting. But seven conditions were tested; under the global null, about 0.35 of them cross p<0.05 by chance. This one surfaced only after the single-layer version came back null. Chasing it — adding scenarios, trying one-tailed tests — would be trading the project's methodological discipline for a marginal result. The only legitimate follow-up is a single pre-registered high-N confirmatory run, and an 8% recovery effect isn't worth that cost relative to what we already have. Report it honestly as suggestive, stop there.

**What the finding is, precisely.**

*ORDER preference is 100% linearly decodable at layer 8 (Qwen2.5-3B) from interaction-form activations, above a verified lexical floor. But the decodable direction d̂ at Q' positions is causally epiphenomenal: moving it to full target magnitude produces no behavioral change. Decodability and causal use are dissociated.*

The bounds of that claim matter:

1. *Tested positions only.* The interchange targeted Q'+tail — the tokens belonging to the question, which appear after the demonstration. The demonstration positions, where the behavioral effect of Gate B is most likely computed, are not token-aligned between the two contexts and couldn't be interchanged. "d̂ at Q' is not causal" is not the same as "ORDER is never causally used."

2. *d̂ vs. probe direction.* The interchange operated on d̂ (diff-of-means). cos(d̂, probe_weight) = 0.881 — about 28° apart. They are correlated but not identical. The precise claim is "d̂ at Q' is not causal," not "the probe direction is not causal."

3. *Single model.* Qwen2.5-3B-Instruct. Whether the same dissociation holds in larger models is untested.

4. *Linear directions only.* Non-linear mediation is invisible to this method.

5. *Gate B confound.* B1_clean's behavioral effect may be partially due to format induction — the model reproducing the most recent assistant turn's structure rather than retrieving a preference representation. The claim is bounded at "d̂ at Q' is not causal"; it doesn't support broader conclusions about what the representation is used for.

**What this says about the original question.**

I started with an observation about two windows giving different advice from the same document, and a hypothesis that the difference was representational — that "texture" might show up as something measurable in the activations.

The answer, so far: the representational signal is real. ORDER preference is demonstrably encoded above lexical surface, rising through the network to near-perfect decodability at layer 8. Demonstration form behaviorally steers output (Δ_nat = 0.436). But the decodable direction, at the question positions, is not the causal channel. Whatever is doing the behavioral work under Gate B likely lives at the demonstration positions — the part of the context the model processed the preference through — and that's exactly what this design can't reach.

That's not a dead end. It's a location. The question that follows is: what is happening at the demonstration positions? Can you read the causal structure there, even if you can't interchange it cleanly with this minimal-pair setup?

**The methodological by-product.**

The most transferable thing from this experiment isn't the specific finding — it's the discipline that produced it. Five artifacts were ruled out individually along the way: the lexical ceiling that made Phase 1's first version uninformative, the system-prompt false negative in Gate B v1, the wrong-first primacy confound in Gate B v2, the implementation bugs that invalidated additive patching, and the threshold-chasing pressure at the end of Phase 2. Each one looked, at the moment of discovery, like it might be a real effect. Each had a cleaner explanation.

Interpretability experiments are especially susceptible to this pattern. The readout is always a choice, the comparison is always against some baseline, and no intervention is fully surgical. The discipline isn't uniform skepticism — it's learning to distinguish "this number is interesting" from "this number means what I want it to mean." The four rounds of Phase 0 tightening, the lexical floor construction in Step 5b, the three Gate B iterations — those weren't detours. They were the experiment.
