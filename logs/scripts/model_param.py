GPT_2_6B = {
    "model_name": "GPT_2-6B",
    "num_layers": 32,
    "parameters": {
        # 词表大小
        "vocab_size": 51200,
        # 隐藏层大小
        "hidden_size": 2560,
        # 前馈网络大小，通常是 hidden_size 的 4 倍
        "ff_size": 10240,
        # 计算每层参数大小
        "parameters_per_layer_bytes": [],
        # 总参数量字节数
        "total_parameters_bytes": 0
    }
}

# 计算每层的参数量
embedding_params = GPT_2_6B["parameters"]["vocab_size"] * GPT_2_6B["parameters"]["hidden_size"]

for layer_idx in range(GPT_2_6B["num_layers"]):
    # 自注意力部分: 4 * hidden_size^2
    attention_params = 3 * GPT_2_6B["parameters"]["hidden_size"] ** 2
    # 前馈网络部分: hidden_size * ff_size + ff_size * hidden_size
    feedforward_params = (GPT_2_6B["parameters"]["hidden_size"] * GPT_2_6B["parameters"]["ff_size"] +
                          GPT_2_6B["parameters"]["ff_size"] * GPT_2_6B["parameters"]["hidden_size"])
    # LayerNorm 部分: 每层两个参数 (gamma 和 beta)
    layernorm_params = 2 * GPT_2_6B["parameters"]["hidden_size"]
    # 多头注意力之后的线性层: hidden_size * hidden_size
    linear_after_attention_params = GPT_2_6B["parameters"]["hidden_size"] * GPT_2_6B["parameters"]["hidden_size"]
    
    # 每层的参数总量
    total_layer_params = (attention_params + feedforward_params + layernorm_params + linear_after_attention_params)
    # 将每层参数量以字节为单位加到 parameters_per_layer_bytes
    GPT_2_6B["parameters"]["parameters_per_layer_bytes"].append(total_layer_params * 4)  # 每个参数占4字节

# 总参数量是嵌入层和所有 Transformer 层的参数总和
total_params = embedding_params + sum(GPT_2_6B["parameters"]["parameters_per_layer_bytes"])
GPT_2_6B["parameters"]["total_parameters_bytes"] = total_params

# 打印结果
print("Total Parameters (bytes):", GPT_2_6B["parameters"]["total_parameters_bytes"])
print("Parameters per layer (bytes):", GPT_2_6B["parameters"]["parameters_per_layer_bytes"])
