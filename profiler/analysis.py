import pandas as pd 


def predict_stage_time(df, model_name, model_size, mbs: int, seq_len: int):
    transformer_data = df[df['op'].isin(['TELayerNormSelfAttentionDropout', 'TELayerNormMlpDropout'])]

    # 按模型参数分组并计算forward和backward的时间之和
    transformer_grouped = transformer_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length'], as_index=False).agg({
        'fwd_compute': 'sum',
        'bwd_compute': 'sum'
    })

    transformer_fwd_compute = transformer_grouped[
        (transformer_grouped['model_name'] == model_name) &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length'] == seq_len)
        ]['fwd_compute'].iloc[0]

    transformer_bwd_compute = transformer_grouped[
        (transformer_grouped['model_name'] == model_name) &
        (transformer_grouped['model_size'] == model_size) &
        (transformer_grouped['micro_batch_size'] == mbs) &
        (transformer_grouped['sequence_length'] == seq_len)
        ]['bwd_compute'].iloc[0]
  
    pre_post_process_data = df[df['op'].isin(['PostProcess', 'LanguageModelEmbedding'])]

    pre_post_grouped = pre_post_process_data.groupby(['model_name', 'model_size', 'micro_batch_size', 'sequence_length'], as_index=False).agg({
        'fwd_compute': 'sum',
        'bwd_compute': 'sum'
    })
    pre_post_process_fwd_compute = pre_post_grouped[
        (pre_post_grouped['model_name'] == model_name) &
        (pre_post_grouped['model_size'] == model_size) &
        (pre_post_grouped['micro_batch_size'] == mbs) &
        (pre_post_grouped['sequence_length'] == seq_len)
        ]['fwd_compute'].iloc[0]
    
    pre_post_process_bwd_compute = pre_post_grouped[
        (pre_post_grouped['model_name'] == model_name) &
        (pre_post_grouped['model_size'] == model_size) &
        (pre_post_grouped['micro_batch_size'] == mbs) &
        (pre_post_grouped['sequence_length'] == seq_len)
        ]['bwd_compute'].iloc[0]
    
    
    fwd_compute = transformer_fwd_compute * 24 + pre_post_process_fwd_compute
    bwd_compute = transformer_bwd_compute * 24 + pre_post_process_bwd_compute
    print(f"fwd_compute: {fwd_compute}. bwd_compute: {bwd_compute}")
    
    # 重命名列
    # grouped.rename(columns={
    #     'fwd_compute': 'transformer_layer_fwd_time',
    #     'bwd_compute': 'transformer_layer_bwd_time'
    # }, inplace=True)


if __name__ == "__main__":
    
    file_name = "tp_1_20241129_022804.csv"
    df = pd.read_csv(file_name)
    
    predict_stage_time(df, "gpt", "350M", 2, 2048)
    
    
    # filtered_df = df[(df['model_size'] == "350M") &
    #           (df['sequence_length'] == 2048) &
    #           (df['micro_batch_size'] == 4)
    #           ]












    # desired_columns = [
    # "op", "fwd_compute", "bwd_compute", "weight", 
    #  "fwd_allocated", "fwd_reserved", "bwd_reserved"
    # ]
    # # 指定 op 的展示顺序
    # op_order = [
    #     "LanguageModelEmbedding", 
    #     "LocalLayerNormSelfAttentionDropout", 
    #     "LocalLayerNormMlpDropout", 
    #     "LocalLayerNorm", 
    #     "PostProcess"
    # ]
    # filtered_df = filtered_df[desired_columns]
    # filtered_df["op"] = pd.Categorical(filtered_df["op"], categories=op_order, ordered=True)
    # sorted_df = filtered_df.sort_values(by="op")
    # print(sorted_df)