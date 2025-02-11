# %%
import torch
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.model import ExLlamaV2, ExLlamaV2Config
from exllamav2.attn import ExLlamaV2Attention
# %%
# def test_linear_layer():
# 1. 创建一个简单的配置
config = ExLlamaV2Config()
config.model_dir = "./models/Meta-Llama-3-8B/int4/"
config.prepare()
config.max_batch_size = 1
config.arch_compat_overrides()

model = ExLlamaV2(config)
model.load(lazy = True)
# %%
# print model
model.load()
# %%
for module in model.modules:
    if isinstance(module, ExLlamaV2Attention):
        q_proj = module.q_proj
        break
        # print(module.linear)
# %%
# 2. 创建一个线性层实例
linear = q_proj
q_handle = q_proj.q_handle
# %%
# 3. 准备输入数据
batch_size = 16
seq_length = 4096
input_data = torch.ones(batch_size, seq_length, 4096, dtype=torch.float16).cuda()

# 4. 前向传播
output = linear.forward(input_data)

# 5. 验证输出形状
expected_shape = (batch_size, seq_length, 4096)
assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

print("Test passed successfully!")
print(output.mean())
batch_size = 16
seq_length = 4096
input_data = torch.ones(batch_size, seq_length, 4096, dtype=torch.float16).cuda()

# 4. 前向传播
output = linear.forward(input_data)

# 5. 验证输出形状
expected_shape = (batch_size, seq_length, 4096)
assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

print("Test passed successfully!")
print(output.mean())
batch_size = 16
seq_length = 4096
input_data = torch.ones(batch_size, seq_length, 4096, dtype=torch.float16).cuda()

# 4. 前向传播
output = linear.forward(input_data)

# 5. 验证输出形状
expected_shape = (batch_size, seq_length, 4096)
assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

print("Test passed successfully!")
print(output.mean())
batch_size = 16
seq_length = 4096
input_data = torch.ones(batch_size, seq_length, 4096, dtype=torch.float16).cuda()

# 4. 前向传播
output = linear.forward(input_data)

# 5. 验证输出形状
expected_shape = (batch_size, seq_length, 4096)
assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

print("Test passed successfully!")
print(output.mean())
# %%
