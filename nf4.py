# %%
# bitsandbytes: create_normal_map
import torch
import numpy as np
from scipy.stats import norm
offset = 0.9677083
print("----------------P step---------------------")
step_pos = torch.linspace(offset, 0.5, 17)[:-1] #概率区间
step_neg = torch.linspace(offset, 0.5, 16)[:-1]
print("NF step 9: ",step_pos)
print("NF step 8: ",step_neg)

print("----------------norm---------------------")
v1 = norm.ppf(step_pos).tolist()              #得到概率对应的正态分布的分数位
print("NF+:", [f'{num:.4f}' for num in v1])
v2 = [0]
v3 = (-norm.ppf(step_neg)).tolist()
print("NF-:", [f'{num:.4f}' for num in v3])

print("----------------nf4-pre---------------------")
v = v1 + v2 + v3
print("NF4:", [f'{num:.4f}' for num in v])

print("----------------nf4 归一化---------------------")
values = torch.Tensor(v)
values = values.sort().values
values /= values.max()
print("NF4/max:", [f'{num:.4f}' for num in values])
# %%
print(values)
# %%
print(values.shape)
# %%
