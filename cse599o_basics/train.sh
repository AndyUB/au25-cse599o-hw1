CKPT_DIR=../ckpt
DATA_DIR=../data

mkdir -p $CKPT_DIR

uv run main.py --train_data $DATA_DIR/TinyStoriesV2-GPT4-train.txt \
    --val_data $DATA_DIR/TinyStoriesV2-GPT4-valid.txt \
    --checkpoint_dir $CKPT_DIR