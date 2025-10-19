CKPT=../ckpt/ckpt4000.pt
PROMPT="Once upon a time"
uv run inference.py \
    --model_checkpoint $CKPT \
    --prompt "$PROMPT"
