DATA_PATH=/mnt/varuna/dataset

python3 tools/preprocess_data.py \
    --input $DATA_PATH/oscar-1GB.jsonl \
    --output-prefix $DATA_PATH/gpt-dataset-simplewiki/meg-gpt2 \
    --vocab $DATA_PATH/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $DATA_PATH/gpt2-merges.txt \
    --append-eod \
    --workers 32