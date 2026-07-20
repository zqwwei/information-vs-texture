# HAND-OFF SPEC: STEP 6 — Phase 2 (causal patching: does the model USE the order rep?)

## 给 Claude Code 的话
Phase 1 (decodability) 干净落地:在 word-level lexical 严格 = 0.5 的前提下,order preference 从 GPT-2 激活里线性可解码(embedding 层 0.67 → mid-stack 0.97)。**这只证明信息「线性在场且可访问」,不证明模型「用」了它**(L8)。Phase 2 用 causal patching 把 claim 从「在场」升到「使用」:**外科手术式地只改 order-preference 那个方向的分量,看模型的输出顺序是否随之翻转。**

原则不变:先把「行为把手」验出来再做 patching;每一步出结果给 Ziqi 看;连续描述,不二元判定。**模型 = Qwen2.5-3B-Instruct(已定)**,保留 GPT-2 做一个 logprob 交叉验证桥回 Phase 1。**effect 阈值见 §5(已定,锚在两个参考尺度上,不是凭空一个数)。** 全部设计已锁,直接执行。

---

## §0 — 因果阶梯:这一步到底在证什么

- Phase 1 = **decodability**:信息在表示里、线性可读。✓(已完成)
- Phase 2 = **use(因果)**:如果手术式地改写「order 方向」的分量,模型的**行为(它生成答案的顺序)**是否随之改变?
  - 改了方向 → 行为翻转,且有 **dose-response**(剂量越大效应越大)、**direction-specific**(随机方向不翻转)、**双向**(conclusion↔reasons 都能推)、**necessity**(把方向投影掉,行为退回中性)→ **模型用了这个表示。**
  - 改了方向 → 行为不动 → 信息在场但**不被这条线性方向因果驱动**(也是清楚结果)。
- **天花板(必须写进 limitation):** 因果证据是「在这个 setup 里、这条方向、这些层」对行为有因果作用,不是「模型在所有情况下都靠它」。Phase 2 不声称普适使用。

---

## §1 — 模型(已定:Qwen2.5-3B-Instruct)

**为什么换模型:** 主行为读数(§5)需要一个**输出顺序可读**的模型。GPT-2-small 生成不连贯,读不出「答案是 conclusion-first 还是 reasons-first」。instruction-tuned 模型才有可读的行为把手。这也正是两轮前说的「behavioral readout 属于 instruction-tuned 档」。

**主模型 = `Qwen/Qwen2.5-3B-Instruct`**(ungated,~6GB fp16,MacBook + MPS 能跑,无需申请许可)。用 HF `AutoModelForCausalLM` 加载,decoder layer = `model.model.layers[L]`,forward hook 改 residual stream。
**交叉验证桥 = GPT-2-small:** §5 的 **logprob 版**不需要连贯生成,GPT-2 能做——把因果效应直接接回 Phase 1 那个模型,检查一致性。

> spec 与具体模型无关:任何 HF causal LM 都用同样的 decoder-layer forward hook 拿/改 residual。

---

## §2 — 数据(复用 + 适配)

复用 `step5b_data.json` 的 order 设计(同词、异序、generic correction、group by scenario)。两处适配:
1. **套 chat template:** 把 B-form 的 `{name}:/Assistant:` 多轮放进目标模型的 chat 格式(instruct 模型对格式敏感)。GPT-2 保持纯文本。
2. **Leave-one-scenario-out 交叉拟合(代替固定留 4 个):** 对每个被测场景 s,order 方向(§4)只在**其余 11 个**场景的上文上拟合,行为读数在 s 的 Q' 上测。轮遍 12 个场景 → 12 个效应点,误差棒来自这 12 点。这样既不在「拟合方向的同一批数据」上测因果,又把数据用满。

每个行为测试 item:
- **persona 上文** = 某场景的 B-form(确立此人 order 偏好,label 0 或 1)。
- **新问题 Q'** = held-out 场景的问题(persona 上文里没出现过)。
- **两个候选续写**(同内容、两种顺序,用 Q' 场景的 verdict/rationale 造):
  `C_concl = "{verdict} {rationale}"`,`C_reas = "{rationale} {verdict}"`。

---

## §3 — Phase 2a:两个前置闸门(都过了才继续)

**闸门 A — decodability 在新模型上重现。** ✓ 已过:Qwen2.5-3B 上 L\*=8,解码 1.000(L8→L29 持续满分)。**direction 从 L\*=8 取,不从 layer 0**(避开 BPE 表面伪信号)。

**闸门 B — 行为把手存在(unpatched)。** ⚠️ **第一次跑的结果作废,因为没跑成预期实验**(B-form 被放进了 system prompt,不是真实 chat 轮次)。同时纠正读法:**0.34σ 不是「方向反了」,是「无分离」(null)** —— 不要从 0.34σ 读符号。重跑前三处必修:

**(B-fix-1)真实多轮 chat,不是 system 整段。** 把演示偏好的来回做成**真正的 user/assistant 消息对象**(model 的过去回合就是它「自己」给过 wrong-order→被纠正→给 right-order→被接受),Q' 作为最后一个 user 回合。instruct 模型只对真实 role 结构敏感,不把 system 里的对话当「要遵从的演示」。

**(B-fix-2)配对测量 + 去先验基线。** `order_score` 用 **per-token 平均 logprob**(§5)。对**同一个 Q'**,在三种条件下各测一次:
- `concl-persona`、`reas-persona`、`no-persona`(中性,无偏好上文)。
- **持手 = 配对的 persona 效应** `Δ_pair(Q') = order_score(concl) − order_score(reas)`,在同一 Q' 内做差 → 消掉场景方差,也消掉模型自身的强 conclusion-first 先验(先验在两条件相同,作差抵消)。`Δ_nat = mean_Q' Δ_pair`,误差棒来自跨 Q'。

**(B-fix-3)诊断梯子 —— 把「B 失败」从「歧义」变成「可解释」。** 按顺序测两级:
- **B0(上界 / 测量是否成立):** 不演示,直接**显式 system 指令**(「Always answer with the conclusion first」vs「reasons first」)。这测「这个行为读数 + 这个模型,在最强信号下能不能产生把手」。
- **B1(texture 条件):** 真实多轮**演示**(B-fix-1),不显式说偏好。这才是假设要测的。

### 决策树(重跑后照此走,两条岔路都给出真结果)

| B0 | B1 | 含义 & 下一步 |
|----|----|----|
| 过 | 过 | 把手存在 → 跑 §4/§5 完整 patching(真正的 texture 因果测试)。 |
| 过 | null | **真发现**:order 完美可解码、显式指令能驱动行为,但**单轮演示的偏好驱动不了生成**(decodability ≠ use 的具体化,L8)。→ 转 §3.5 的「方向因果」回退测试(这是另一个、更弱但诚实的 claim,别和「用了演示偏好」混为一谈)。 |
| null | — | 行为读数在 Qwen-3B 上**测不出把手**(连显式指令都推不动 order)→ 行为臂在这个模型上不成立。报告 decodability-only + §3.5 方向因果作为唯一可用因果探针;记成 model-choice 限制(可考虑换更强 instruct 模型重试 B0)。 |

### §3.5 — 回退:方向因果测试(仅当 B1=null 但 B0=过)

**这回答的是另一个问题,措辞必须分清:** 不是「模型用了演示的偏好」,而是「**L\*=8 的 order 方向,对模型输出顺序有没有因果控制力**」。做法 = §4/§5 的干预与全部控制照跑,但**判据改写**:
- 没有 Δ_nat 当分母了(≈0),所以**不报 recovery**;报 **patched 后 order_score 是否被推动 + 全部控制**(随机方向 null、dose-response、双向、消融、coherence guard)。
- 判据:真方向推动 order_score 且 ≥ 随机方向 null 2σ、dose-response 单调 → 「L8 order 方向对生成顺序有因果作用,但**演示偏好→行为**这条通路是缺失的那一环」。
- **不得**把这个结果说成「模型使用了 persona 的偏好」——它只说明那条方向能因果操纵顺序输出。

---

## §4 — Phase 2b:order 方向 + 手术式干预

**方向(primary = difference-of-means,在 L\*):**
在 L\* 取每个 B-form 上文的 residual(读点 = 上文最后一个 token,或 final-answer span 的 mean,与 Phase 1 一致),
`d = mean_{reasons-first} − mean_{conclusion-first}`,单位化 `d̂`。
**交叉校验:** 算 `d̂` 与 Phase-2a probe 权重方向的 cosine(应较高,证明两种取法一致);diff-of-means 作主方向(对 causal steering 更稳)。

**干预(在 L\* 的 decoder layer 输出上加 forward hook,改 residual stream):**
- **(i) 加性 + dose-response(主):** `a' = a + α·‖a‖·d̂`(推向 reasons-first)或 `−` 号(推向 conclusion-first)。`α ∈ {0, ±0.5, ±1, ±2, ±4}` 扫一遍。
- **(ii) 投影-置位(外科版,Claude Code 的点):** `a' = a − (a·d̂)d̂ + c·d̂`,`c` 设为目标类在 L\* 的典型投影(target 类 `a·d̂` 的均值)。**只替换 order 方向那一个分量,其余子空间不动**——这是把「顺序特征」设成目标值、保留 topic/name/其它一切。
- patching 作用于 persona 上文位置(+ 生成位置,见 §5)。

---

## §5 — Phase 2c:因果读数 + 控制 + effect 阈值(已定)

**主读数(logprob,judge-free,任何模型可用):**
`order_score = meanlogP(C_concl | persona+Q') − meanlogP(C_reas | persona+Q')`。
**用 per-token 平均 logprob**(不是 sum)——C_concl/C_reas 同词异序,token 数因 BPE 边界可能差 1–2,per-token 均值消掉长度偏差。
patched 后重测,报告 **Δorder_score = patched − unpatched**,across 12 个 leave-one-out 场景,带误差棒。
推 reasons-first 应使 order_score **下降**,推 conclusion-first 应使其**上升**。

**必须有的控制(没有这些,单次翻转不证明任何因果):**
1. **随机方向 null:** 同范数随机方向 `r` 替 `d̂`(多个随机种子)→ 给出 Δorder_score 的 null 分布。
2. **Dose-response:** Δorder_score 随 α(§4 的 `{0,±0.5,±1,±2,±4}`)**单调**。
3. **Necessity(消融):** 只投影掉、不加 → order_score 应**退向 0**。
4. **双向对称:** 推两个方向都测,效应反号对称。
5. **Specificity:** 改了**顺序**不应改**内容/话题**(简单 content-similarity 或定性确认续写仍切题)。
6. **层扫:** L\* 附近几层各做一遍。
7. **Coherence guard:** 大 α 会把模型推崩(续写 logprob 整体塌陷)→ 那个 α 的 Δ 无效。每个 α 记录 C 的绝对 per-token logprob,若较 α=0 掉超过某量级则标为「模型已不连贯」,不纳入主结论。

### effect 阈值 —— 锚在两个真实参考尺度,不是凭空一个数

绝对 logprob 单位没有可解释性(随长度/分词/模型变)。所以阈值**相对**两个本实验自带的尺度定义:

**尺度 A — 自然行为摆幅 `Δ_nat`(来自 §3 闸门 B):**
`Δ_nat = mean(order_score | conclusion-first persona) − mean(order_score | reasons-first persona)`(unpatched)。
这是「**换掉真实 persona**」让模型顺序偏好移动多少 = 生态上限。

**尺度 B — 随机方向 null 的 std `σ_rand`(来自控制 1)。**

**因果效应「成立」的判据(连续三条全过,用程度描述不用是非):**
- **(a) 方向特异:** 真方向 |Δorder_score| ≥ 随机方向 null 均值 + **2·σ_rand**。(效应不是任意扰动。)
- **(b) 分级:** Δorder_score vs α 的 Spearman ρ 的绝对值 ≥ **0.8** 且符号正确。(因果是分级的,不是噪声尖峰。)
- **(c) 显著且符号对:** 12 场景的配对 Δorder_score 的 |mean|/SE ≥ **2**,符号与 steering 方向一致。

**幅度(headline,用投影-置位 (ii) 的「自然剂量」效应报告):**
`recovery R = |Δorder_score(投影-置位)| / |Δ_nat|` —— 「只把 order 方向设成目标值,复现了换真实 persona 效应的百分之几」。
解读带(**不设硬性成功线,连续判断**):
- **R ≳ 0.5**:这条单一方向**中介了大部分**行为效应 → 强因果使用。
- **0.2–0.5**:这条方向是**一个真实因果通道**(但非全部)→ 部分中介。
- **< 0.2 但 (a)(b)(c) 都过**:因果但只是**次要线性通道** → 弱而真。
- **(a) 不过**:这条线性方向上**测不到因果使用**(对照 L11/L13:可能非线性中介,或行为把手太弱)。

**Necessity 单列:** 消融后 `|order_score_ablated| / |order_score_unpatched|` ≤ **0.5** = 投影掉这条方向移除了至少一半行为把手 → 方向是必需的;同样用比例和带描述,不二元。

**判据语言示例:** "在 L\*,投影-置位把 order 方向设成 reasons-first 目标值,使 order_score 移动 −X,recovery R=0.4(随机方向 null 内 ≈0,dose-response ρ=0.9 单调,消融移除 0.6 行为把手)→ 模型在这个 setup 里因果性地用了这条 order 方向,它中介了约四成的自然行为效应。"

**生态学次级读数(可选):** 自由生成 → 独立 judge 判顺序 → patched vs unpatched 翻转率。比 logprob 真但吵,佐证不作主结论。

---

## §6 — Limitations(写进脚本输出)

- **L8 升级:** Phase 1 = 在场;Phase 2 = 在此 setup/此方向/此层有**因果作用**。**不**声称模型在所有情况下都用它,也不声称这条线性方向是唯一中介。
- **L11(线性中介假设):** diff-of-means 是**线性**方向;若使用是非线性的,线性 patching 可能低估效应。线性 null ≠ 无因果使用。
- **L12(行为把手依赖):** 整个 Phase 2 依赖 §3 闸门 B 的 unpatched 行为分离;若分离弱,效应上限就低。
- **L13(persona 仍是单轮演示):** B-form 仍是最小演示;这测的是「单次被接受顺序」的因果,不是富 persona 聚合。v3 再加聚合。
- **L14(模型迁移):** 在 instruct 模型上的结论不自动迁回 GPT-2;GPT-2 的 logprob 交叉验证用来检查一致性,不是等价证明。

---

## §7 — 边界 & 文件命名

**这一步 = Phase 2(因果使用)。** 做完若拿到「方向特异 + dose-response + 必需性」的因果效应 → texture 假设拿到**最强一档**证据(在场 → 被使用)。下一刀(若有,Phase 3)= **机制定位 / 跨模型规模曲线**(哪些 head/层实现这个使用;effect 随模型规模如何变),不在这一步。

```
texture_experiment/
├── ... (step1–step5b 已有)
├── step6_data_chat.py        ← B-form 套 chat template + 造 held-out 行为测试对
├── step6a_handles.py         ← 闸门 A(decodability 重现, 定 L*)+ 闸门 B(unpatched order_score 分离)
├── step6b_patch.py           ← diff-of-means 方向 + 加性/投影干预 + hook
├── step6c_causal.py          ← Δorder_score + 全部控制(随机方向/dose/消融/对称/specificity/层扫)
├── step6_doseresponse.png    ← Δorder_score vs α(含随机方向 null)
├── step6_layer_sweep.png     ← 因果效应 vs 层
└── step6_handles.png         ← unpatched 两组 order_score 分布(行为把手)
```

**带回原对话:** ① 闸门 A/B 是否过(L\*、unpatched 分离度);② dose-response 曲线 + 随机方向 null;③ 消融是否退回中性;④ 双向是否对称;⑤ GPT-2 logprob 交叉验证一致否;⑥ 任何 surprise;⑦ 过程体感。
