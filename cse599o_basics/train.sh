DATA_DIR=../data

mkdir -p $CKPT_DIR

# lrs="1e-3 5e-4 1e-4"
lrs="5e-4 1e-4"

export PYTHONUNBUFFERED=1

for lr in $lrs; do
    LOG_DIR=../logs/lr_$lr
    CKPT_DIR=../ckpt/lr_$lr
    mkdir -p $LOG_DIR
    uv run main.py --train_data $DATA_DIR/TinyStoriesV2-GPT4-train.txt \
        --val_data $DATA_DIR/TinyStoriesV2-GPT4-valid.txt \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CKPT_DIR \
        --lr $lr > $LOG_DIR/train.log 2>&1
done