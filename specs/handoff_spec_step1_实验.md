# HAND-OFF SPEC: 信息与质地 — Interpretability 实验

## 给 Claude Code 的话
帮我在本地执行一个 interpretability 实验。我是有工程背景的 SDE,但第一次跑 transformer 实验。
执行原则:**一步跑通再进下一步,不要一次写完整个 pipeline。** 每步跑出结果给我看,我确认了再继续。
我的瓶颈是"想太多不动手",所以你的任务是帮我快速产生真实结果,不是帮我把方案想得更完美。

---

## 背景(一句话)
假设:LLM 在「只有信息」vs「信息+真实交互历史」两种 context 下,内部表示不同。
那个差异如果存在,就是"质地"(对一个人的理解)的计算对应物。
本实验用 GPT-2 当显微镜,看这个差异在 activation 层面存不存在、长什么样。

---

## 环境
```bash
pip install torch transformers
# 可选(后面画图/算距离用): pip install numpy matplotlib scikit-learn
```

---

## STEP 1 — 显微镜开机(死代码,直接跑)
目标:确认能加载 GPT-2、抽出中间层 activation、看到形状。还不涉及 A/B 数据。

```python
import torch
from transformers import GPT2Model, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

text = "He hesitated before deciding."
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

hs = out.hidden_states   # 13 层:1 embedding + 12 transformer
print("层数:", len(hs))
print("第6层形状:", tuple(hs[6].shape))   # 期望 [1, n_tokens, 768]
```
**通过标准:** 打印出 `[1, n_tokens, 768]` 这样的形状。看到了 → 进 STEP 2。

---

## STEP 2 — 构造 A/B context(这一步的核心决定是我的,不是你的)

实验的关键在数据设计。我需要先定义三样东西,Claude Code 请**先问我、不要替我假设**:

1. **要"理解"的对象是什么?**
   建议:一个虚构的人,比如 "Alex"。给他一组明确的偏好(例:回答喜欢简短、讨厌长篇大论、犹豫时希望被推一把)。

2. **Context A(只有信息)长什么样?**
   把 Alex 的偏好写成静态描述,像一份说明书。
   例:"Alex prefers short answers. Alex dislikes long explanations. ..."

3. **Context B(信息 + 交互历史)长什么样?**
   同样的偏好,但以**真实来回对话**的形式呈现,带 accept/reject 信号。
   例:多轮 "Alex asked X → assistant gave long answer → Alex: 'too long' → assistant shortened → Alex: 'better'"。
   **关键控制变量:** A 和 B 必须编码**同样的偏好信息**,唯一的区别是「陈述」vs「经历」。否则差异可能来自信息量不同,而不是形式不同。

> 这一步是实验成败的核心。Claude Code 请把 A/B 两段文本的草稿写出来给我审,我来判断它们是否真的只差"形式"这一个变量。

---

## STEP 3 — 对比 activation
两个 context 各跑一遍,抽同一层(先试第 6 层和最后一层),对比表示差异:

- 取每个 context **最后一个 token** 的 activation 向量(它"读完"了整个 context)。
- 算 A 向量和 B 向量的:① cosine similarity ② L2 距离。
- 对所有 12 层都算一遍,看差异在哪一层最大(画一条 "层 vs 差异" 的曲线)。

**通过标准:** 得到一条曲线,无论差异大还是小,能说清"这说明了什么"。

---

## STEP 4 — 解读(我和 Claude Code 一起做)
三种可能结果,都要能解释:
- **差异明显:** 形式(经历 vs 陈述)真的改变了内部表示 → 质地假设得到初步支持,值得做更大模型。
- **差异微弱:** 可能 GPT-2 太小、或 A/B 没控制好、或假设本身需要修正 → 分析是哪个。
- **没差异:** 也是真结果 → 说明这个现象要么不存在于这个尺度,要么实验设计要改。

---

## 我会反馈回来的东西
跑完后我会把以下带回原对话:
1. 哪一步卡住了(如果有)。
2. STEP 3 的那条曲线 / 数字。
3. **最重要的:跑这个实验的时候,我是觉得兴奋停不下来,还是觉得烦想逃。**
   (这是在测我适不适合 research engineer 这条路,比实验结果本身还重要。)
