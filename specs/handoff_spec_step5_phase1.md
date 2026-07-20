# HAND-OFF SPEC: STEP 5 — Phase 1 (linear-probe decodability)

## 给 Claude Code 的话
Phase 0(distance)收口在一个干净的度量 null:在 mean-pool 距离上,换偏好内容 (B') 相对只改措辞 (floor) 没有可见位移。结论不是"模型没表示人",而是**距离这个工具分不开"没表示 / 有但 readout 看不到"**。Phase 1 把 readout 从「距离」换成「可解码性」——这能区分上面两种情况。

原则不变:**先建数据 → 先验证 probe 能跑 → 再出对比图,一步给我看一次。** 这一轮**只做 GPT-2 上的 linear probe(decodability)**,**不做** behavioral readout(理由见 §0),**不做** causal patching(那是 Phase 2)。

**开工前唯一要确认的:数据集规模(§2 的 N)。** 默认 8 profiles × 8 phrasings = 64 段/form。Ziqi 觉得够就跑,想要更稳就调大。其余设计已锁。

---

## §0 — 这一轮回答的精确问题(以及为什么是 probe 不是别的)

**Phase 0 的 null 为什么发生:** mean-pool 对一段 ~270 token、其中 ~70–80% 逐字相同的序列做平均,结果由共享骨架主导,被测内容(回答 / 纠正句)那一小撮 token 被冲淡到 floor 以下。所以距离看不见 ≠ 信息不在,只是这个度量在这种近孪生序列上结构性地接近 1。

**probe 为什么能看见 distance 看不见的东西:** 线性 probe 不读"方差占多少",它读"某个属性是否线性可分地存在于激活里"。即使偏好信息只占激活方差的一小部分、被 mean-pool 冲掉,probe 仍可能找到那个方向把它读出来。**前提:probe 不能再去 naive 整段 pool**(见 §3 读点)。

**关键:shuffle 是太弱的 floor,真正的 floor 是 lexical baseline。** shuffle 只测"标签是不是随机贴的"。但 B-form 里偏好被演示了多轮、纠正词("short"/"options"/"push")在用户纠正句里**字面出现过**(只是不在陈述位置)——probe 只要学到"用户说过 short 这个词"就能高于 shuffle,而这几乎等于读词面,**不是** texture 假设要的"模型把演示整合成了对人的表示"。所以"高于 shuffle"会给假阳性的兴奋。真正要清的分界是:**模型中间层的解码精度,是否高于纯词面 baseline**——这才是"模型算出了什么" vs "信号本就在词面上"。

**这一轮的核心 texture 测试(within-B-form,逐层,连续量,不做二元判定):**

> **B-form 的「激活 probe」解码精度,逐层是否超过 B-form 自己的「lexical baseline」?**

- B-form 激活 probe 在**中层超过** B-form lexical baseline → 模型把隐式演示算出了**超越词面**的、线性可访问的 person-rep。**这是 texture 信号。**(继承 Phase 0 教训:看"被层层算出来"的东西——早层激活≈词面,若中层爬到 lexical 之上,就是计算贡献。)
- B-form 激活 probe ≈ lexical baseline(各层都贴着)→ 全是词面,模型没加表示。
- 两者都贴 shuffle(≈chance)→ 不可线性解码,显微镜太小 / 数据集太难(**此时**上大模型才有理由,见 §6)。

**A-form 是校准线,不是 ceiling。** A-form 把偏好直说("Alex prefers short answers"),probe 大概率从 layer 0 就饱和(≈1.0)、**一条平线、没有形状**——所以它是 *trivial baseline*(证明 probe 在偏好被明说时确实读得出),**不能**用"B-form 追上 A-form 的轨迹"当判据(参考线没形状,判据就废)。A-form 的用途只有两个:(a) sanity——probe 在明说时该接近满分;(b) 给 B-form 中层达到的绝对水平一个"读显式文本有多高"的标尺。**texture 判据锚在 B-form vs B-form-lexical,不锚在 A-form 的形状。**

---

## §1 — 为什么现在不做 behavioral readout(写进 spec,免得有人提)

behavioral readout(同一个关于 Alex 的新问题,A-context vs B-context,看回答是否更贴预测方向)是对的方向,但**不在这一轮**:GPT-2-small **不是 instruction-tuned**,它不会"照着 Alex 的偏好作答"。behavioral null 会被"模型根本不跟指令"这个 capability confound 污染,读不出假设(这是 L3「显微镜太小」第一次真正成立)。**behavioral readout 属于 instruction-tuned 模型阶段(Llama-3-8B-Instruct 那一档),Phase 2 再做。** 这一轮 decodability 不需要模型"会作答",只需要信息"线性在场",所以 GPT-2 能做。

---

## §2 — 数据集(程序化生成,Claude Code 写 `step5_dataset.py`)

**三条偏好轴**(复用 Alex,已验证干净),每条是二元标签:

| 轴 | 标签 0 | 标签 1 |
|---|---|---|
| `length`        | 简短 (short)            | 详尽 (long/thorough) |
| `order`         | 结论先行 (conclusion-first) | 理由先行 (reasons-first) |
| `decisiveness`  | 被推一把 (push to one)   | 要更多选项 (more options) |

**每个 persona = 三轴的一种组合** → 2³ = 8 个 profile。每个 profile 生成 **M 种措辞实例**(默认 M=8),共 **8 × M** 段/form。

**两种 form,每个 persona 各生成一段(同 persona 内 length-matched):**
- **A-form(陈述):** 三条偏好直说成说明书。例:"<NAME> prefers short answers. <NAME> wants the conclusion first. <NAME> wants to be pushed toward one option."
- **B-form(对话/经历):** 三条偏好用多轮 Q&A + 纠正信号**演示**出来,**偏好词从不直接出现在陈述里**(只出现在用户的纠正句里,像 Phase 0 的 B)。

**生成方式 = 模板 + 槽位,关键是让"标签"成为唯一跨实例一致的可预测信号:**
- **场景轮换:** 准备一个场景池(≥8 个:解释概念 / 做决策 / 选择 / 写作请求 / 排查问题 / 计划行程 …),每个实例**随机抽不同场景**,这样 probe 学不到"场景 = 标签"。
- **措辞轮换:** 每条偏好的提问句、错答、纠正句各准备多个改写,实例间换着用。
- **名字轮换:** 名字从一个池里抽(Alex / Sam / Jordan / …),**避免 probe 学到某个名字 token = 某标签**(把 Phase 0 的 L4 直接做进数据生成)。
- **去捷径自查(必须):** 生成后跑一个"词面泄漏检查"——B-form 里**不得**出现偏好轴的关键词字面(short/long/conclusion/options/push 等)在**陈述位置**;只允许出现在用户纠正句里。打印一份 sample 给 Ziqi 抽审。
  - 注:纠正句里仍会出现关键词(自然对话避不开),所以 B-form 的 lexical baseline 不会是 0——这没关系,§4 的 lexical baseline **正是用来扣掉这部分词面可解性**的。想让 texture 测试更敏感,可选地让纠正句**少用偏好关键词的字面**(用近义/转述),把 lexical baseline 压低、给激活 probe 留出"超出词面"的空间;但别为此把对话写得不自然。两种写法 lexical baseline 都会如实反映,不强制。

> **验证关卡(给 Ziqi 看):** 随机打印 2 个 profile 的 A-form 和 B-form 各一条 + 词面泄漏检查结果 + 每 form 的 token 数分布。Ziqi 盯一件事:**B-form 里偏好是不是只被"演示"、没被"陈述"**(否则核心测试退化成"读显式 vs 读显式")。

---

## §3 — 读点(关键:不要重蹈 mean-pool 覆辙)

对每段 context,逐层(0–12)抽激活,**用两种读点,都报:**
1. **last-token**(对照;Phase 0 已知它受结尾主导,但这里序列结尾不再统一,且 probe 不是测自发状态,可作对照)
2. **mean-over-answer-tokens**:**只对 assistant 回答的 token 求平均**,不是整段 pool。理由:偏好信号集中在回答里,只 pool 回答能避免被共享骨架(提问句/scaffolding)冲淡——这是直接针对 Phase 0 null 成因的修正。
   - 实现:tokenize 时记录每个 "Assistant:" 段的 token span,只平均这些 span。A-form 没有 assistant 轮 → 对 A-form 用 mean-over-all(它本来就是纯偏好陈述,没有骨架稀释问题)。

> 不引入 probe 后缀(Phase 0 已否)。probe 在固定 readout 上读,不重塑 attention。

---

## §4 — probe + 判据(Claude Code 写 `step5_probe.py`)

**模型:** 每轴 × 每层 × 每 form,训一个 **logistic regression**(L2 正则,标准化输入)。

**评估:** **k-fold 交叉验证(默认 5-fold),按 profile 分组划分**(GroupKFold,确保同 profile 的不同措辞实例不跨 train/test 泄漏)。报告 CV accuracy 的均值 ± std → **这是 Phase 0 缺的误差棒,n=1 问题在此解决(L6 关闭)**。

**两条 floor(都用同一套 GroupKFold split,apples-to-apples):**
1. **shuffle floor:** label-shuffled 重跑一遍 → chance(应 ≈ 0.5)。只作 sanity,确认有信号。
2. **lexical baseline(关键,主对照):** 对**原始文本**跑 `CountVectorizer`(或 token-presence)+ LogReg,解码同一标签。**这是"信号本就在词面上能解多少"的线。** B-form 的 lexical baseline 当作一条**水平参考线**画进每张图。
   - 对 A-form 也算 lexical baseline(它直说偏好,lexical 大概率就接近满分——这恰恰印证 A-form 是 trivial baseline)。

**判据(锚在 B-form,连续量):** 核心比较是 **B-form 激活 probe(逐层) vs B-form lexical baseline(水平线)**。激活在中层**高出** lexical 几个 std = 模型贡献了超越词面的表示;贴着 = 没贡献;贴 shuffle = 不可解码。

**输出:**
- **表:** 每轴,每层,B-form 激活 acc / A-form 激活 acc(均 ± std);外加每轴的 B-form lexical / A-form lexical / shuffle(各一个数 ± std)。
- **图(每轴一张):** x = layer,y = CV accuracy;曲线 = **B-form 激活(主角)** + A-form 激活(校准);水平线 = **B-form lexical(主对照)** + shuffle(chance)。带误差带。**重点看:B-form 激活这条,中层有没有抬到 B-form lexical 那条水平线之上。**
- **quick read(连续语言,不二元):**
  - **texture 信号:** B-form 激活在第几层、哪些轴**超过** B-form lexical,超出几个 std。
  - **计算 vs 词面:** B-form 激活 layer-0 → 峰值的爬升幅度(Δacc);早层≈lexical、中层>lexical 才是"算出来"的形状。
  - **A-form 饱和检查(Problem 2 处理):** 先看 A-form 激活是否从 layer 0 就 >0.95。**若是 → A-form 是 trivial baseline,平线无形状,不要拿"B-form 追上 A-form 轨迹"说事;** 改说"B-form 中层达到 P% 的绝对水平(对照 A-form 的显式上限 Q%、B-form lexical 的词面线 R%)"。
  - 措辞模板:"在 <轴>、layer <L>,从 B-form 激活线性解码到 <Z>%(B-form lexical <R>%,shuffle <W>%);激活比词面高 <Z−R> 个百分点 / <k> 个 std。"

---

## §5 — Limitations(写进脚本输出)

- **L1–L2(继承):** 合成文本、非真实经历;偏好密度等未完全控。
- **L4/L6(本轮关闭):** 名字轮换消除名字 token confound;CV 给出误差棒,n=1 解决。
- **L7(本轮新加的 floor):** "高于 shuffle"不够,必须"高于 lexical baseline"才算模型贡献——否则信号可能本就在词面上。结论锚在激活 vs 词面之差,不锚在激活 vs chance。
- **L8(probe 的核心边界,必须显著写):** **decodability ≠ use。** probe 阳性只说明偏好信息**线性在场且可访问**,**不**说明 GPT-2 在生成时**用**了它。"有且被用 / 有但没用"这一刀 probe 切不开——那需要 **causal patching**(把解码出的方向 patch 进去看行为变不变),是 **Phase 2**,在 instruction-tuned 模型上做。
- **L9:** 只测**线性**可解码。非线性可能存在的信息 probe 读不到;线性 null 不等于信息不在,只是"不线性可读"。
- **L10:** GPT-2-small 尺度限制依旧;若 B-form 激活全程贴 lexical/shuffle(扣掉词面后无增量),**这时**上大模型才有依据(对比 Phase 0 的 null——那个 null 不构成上大模型理由,因为是 readout 造成的;Phase 1 这个"扣掉词面后无增量"的 null 才是"显微镜不够"的证据)。

---

## §6 — 边界 & 文件命名

**这一轮 = Phase 1(decodability,线性、在场性)。** 它能区分 Phase 0 分不开的"没表示 vs 有但距离看不到":若 B-form 可解码 → 信息**在场**(距离只是看不到);若 B-form 贴 floor → 在这个尺度信息确实不可线性访问。

**下一刀(Phase 2,不在这轮):** readout 从"在场性"换成"因果/行为"——causal patching + behavioral readout,在 **instruction-tuned 模型**上,回答"模型是否**用**了这个 person-rep"。这才是 texture 假设的终点形态。

```
texture_experiment/
├── ... (step1–step4 已有)
├── step5_dataset.py     ← 新建:程序化生成 A-form/B-form persona 数据 + 泄漏自查
├── step5_probe.py       ← 新建:per-layer logistic probe + GroupKFold CV + shuffle floor
├── step5_decode_length.png      ← length 轴:A/B/shuffle × layer
├── step5_decode_order.png       ← order 轴
└── step5_decode_decisiveness.png← decisiveness 轴
```

**带回原对话:** ① 三轴各自 B-form 的 per-layer 解码曲线(是否爬升、峰值层、追到 A-form 几成);② B-form vs shuffle 的差(几个 std);③ 任何 surprise(某轴特别可解码 / 某层特别突出 / A-form 自己都解不出);④ 过程体感。
