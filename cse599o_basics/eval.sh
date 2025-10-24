PYTHONUNBUFFERED=1 uv run eval.py \
    --checkpoint_path ../ckpt/best.pt \
    --val_data ../data/TinyStoriesV2-GPT4-valid.txt
