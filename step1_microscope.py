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
print("token 数:", inputs.input_ids.shape[1])
print("最后一层最后一个 token 的前 5 个值:", hs[-1][0, -1, :5].tolist())
