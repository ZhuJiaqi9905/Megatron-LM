# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## "algo" stands for tensor parallel partition algorithm
model_prof_configs = {
    "gpt": {
        "dtype": "fp16",
        "model_size": ["350M", "1_3B", "2_6B", "6_7B",],
        "mbs": [
            1, 
            2, 
            # 3,
            # 4, 
            # 5,
            # 6,
            # 7,
            # 8,
            # 16,
        ], 
        "seq_len": [
            # 512,
            # 1024,
            # 1536,
            # 2048,
            # 2560,
            # 3072,
            # 3584,
            # 4096,
            8192,
            16 * 1024,
            32 * 1024,
            # 64 * 1024,
            ], 
        "seq_len_kv": [ # used when enable --prof-core-attention
            # 512,
            # 1024,
            # 1536,
            # 2048,
            # 2560,
            # 3072,
            # 3584,
            4096
            ],
        "ops": [
                "LanguageModelEmbedding",
                "TELayerNormPostProcess",
                "TELayerNormSelfAttentionDropout", 
                "TELayerNormMlpDropout", 
                # "PostProcess",
                # "TELayerNorm",
                # "LocalLayerNormSelfAttentionDropout", 
                # "LocalLayerNormMlpDropout", 
                ]
    }
}

# model_size: (num_layers, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (1,  1024, 1024*4, 16, 1024//16, 51200, "fp16"),
    "1-3B": (1,  2048, 2048*4, 32, 2048//32, 51200, "fp16"),
    "2-6B": (1,  2560, 2560*4, 32, 2560//32, 51200, "fp16"),
    "6-7B": (1, 4096, 4096*4, 32, 4096//32, 51200, "fp16"),
    "13B": (1,  5120, 5120*4, 40, 5120//40, 51200, "fp16"),
    "scale-layer": (1, 512, 512*4, 8, 512//8, 51200, "fp16")
}

