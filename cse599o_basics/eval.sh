PYTHONUNBUFFERED=1 uv run eval.py \
    --checkpoint_path ../ckpt/10/resume_warmup_500_cosine_5000_lr_1.5e-3_5e-5/ckpt5000.pt \
    --val_data ../data/TinyStoriesV2-GPT4-valid.txt
