import pandas as pd 





if __name__ == "__main__":
    
    file_name = "tp_1_20241128_060054.csv"
    df = pd.read_csv(file_name)
    filtered_df = df[(df['model_size'] == "6_7B") &
              (df['sequence_length'] == 2048) &
              (df['micro_batch_size'] == 8)
              ]

    desired_columns = [
    "op", "fwd_compute", "bwd_compute", "weight", 
     "fwd_allocated", "fwd_reserved", "bwd_reserved"
    ]
    # 指定 op 的展示顺序
    op_order = [
        "LanguageModelEmbedding", 
        "LocalLayerNormSelfAttentionDropout", 
        "LocalLayerNormMlpDropout", 
        "LocalLayerNorm", 
        "PostProcess"
    ]
    filtered_df = filtered_df[desired_columns]
    filtered_df["op"] = pd.Categorical(filtered_df["op"], categories=op_order, ordered=True)
    sorted_df = filtered_df.sort_values(by="op")
    print(sorted_df)