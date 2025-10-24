CKPT=../ckpt/best.pt
PROMPT="Once upon a time"

export PYTHONUNBUFFERED=1
max_tokens=512
temperature_p_list="1_1 1_0.9 1_0.8 1_0.01 1000_1 0.8_1 0.01_1 0.9_0.8"

LOG_DIR=../logs/predict
mkdir -p $LOG_DIR

for temp_p in $temperature_p_list; do
    temperature=$(echo $temp_p | cut -d'_' -f1)
    top_p=$(echo $temp_p | cut -d'_' -f2)
    LOG_FILE=$LOG_DIR/${temperature}temp_${top_p}p.log
    uv run inference.py \
        --model_checkpoint $CKPT \
        --prompt "$PROMPT" \
        --max_tokens $max_tokens \
        --temperature $temperature \
        --top_p $top_p > $LOG_FILE 2>&1
done
