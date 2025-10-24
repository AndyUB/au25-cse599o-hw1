DATA_DIR=../data

export PYTHONUNBUFFERED=1

# All single lrs tried
# lrs="1e-2 5e-3 3e-3 2e-3 1.5e-3 1e-3 5e-4 1e-4"
# Good lrs
# lrs="1.5e-3 1e-3 5e-4"
# Best lr
lrs="1e-3"

for lr in $lrs; do
    LOG_DIR=../logs/lr_$lr
    CKPT_DIR=../ckpt/lr_$lr
    mkdir -p $LOG_DIR
    mkdir -p $CKPT_DIR
    uv run train.py --train_data $DATA_DIR/TinyStoriesV2-GPT4-train.txt \
        --val_data $DATA_DIR/TinyStoriesV2-GPT4-valid.txt \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CKPT_DIR \
        --lr $lr > $LOG_DIR/train.log 2>&1
done

# All lr pairs tried
# lr_min_maxs="1e-3_5e-4 3e-3_1e-3 1.5e-3_5e-4 1.5e-3_1e-3 2e-3_5e-5 1.5e-3_5e-5 1e-3_1e-4 1.5e-3_1e-4 1.5e-3_1e-5"
# Best lr pair
lr_min_maxs="1.5e-3_5e-5"
# Also tried warmup_iters=100, not as good
warmup_iters=500
# Also tried cosine_iters=4500 and 4000
cosine_iters=5000

for lr_min_max in $lr_min_maxs; do
    max=$(echo $lr_min_max | cut -d'_' -f1)
    min=$(echo $lr_min_max | cut -d'_' -f2)
    LOG_DIR=../logs/warmup_${warmup_iters}_cosine_${cosine_iters}_lr_$lr_min_max
    CKPT_DIR=../ckpt/warmup_${warmup_iters}_cosine_${cosine_iters}_lr_$lr_min_max
    mkdir -p $LOG_DIR
    mkdir -p $CKPT_DIR

    # Resume from checkpoint
    # resume_ckpt=../ckpt/...
    # seed=42
    # uv run train.py --train_data $DATA_DIR/TinyStoriesV2-GPT4-train.txt \
    #     --val_data $DATA_DIR/TinyStoriesV2-GPT4-valid.txt \
    #     --log_dir $LOG_DIR \
    #     --seed $seed \
    #     --checkpoint_resume_path $resume_ckpt \
    #     --checkpoint_dir $CKPT_DIR \
    #     --enable_lr_schedule \
    #     --warmup_iters $warmup_iters \
    #     --cosine_iters $cosine_iters \
    #     --lr_min $min \
    #     --lr_max $max > $LOG_DIR/train.log 2>&1

    uv run train.py --train_data $DATA_DIR/TinyStoriesV2-GPT4-train.txt \
        --val_data $DATA_DIR/TinyStoriesV2-GPT4-valid.txt \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CKPT_DIR \
        --enable_lr_schedule \
        --warmup_iters $warmup_iters \
        --cosine_iters $cosine_iters \
        --lr_min $min \
        --lr_max $max > $LOG_DIR/train.log 2>&1
done
