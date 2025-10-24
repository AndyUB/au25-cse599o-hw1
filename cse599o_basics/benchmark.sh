export PYTHONUNBUFFERED=1

BENCHMARK_DIR=../bench
mkdir -p $BENCHMARK_DIR

benchmark_context_lengths="128 256 512 1024 2048"

for context_length in $benchmark_context_lengths; do
    uv run benchmark.py \
        --output_file $BENCHMARK_DIR/small_$context_length.csv \
        --d_model 768 \
        --d_ff 3072 \
        --num_layers 12 \
        --num_heads 12 \
        --context_length $context_length > $BENCHMARK_DIR/small_$context_length.log 2>&1

    uv run benchmark.py \
        --output_file $BENCHMARK_DIR/large_$context_length.csv \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --context_length $context_length > $BENCHMARK_DIR/large_$context_length.log 2>&1
done

breakdown_context_lengths="128 256 512 1024 2048"

BREAKDOWN_DIR=$BENCHMARK_DIR/breakdown
mkdir -p $BREAKDOWN_DIR

for context_length in $breakdown_context_lengths; do
    breakdown_dir=$BREAKDOWN_DIR/${context_length}context
    mkdir -p $breakdown_dir
    uv run ffn_attention_breakdown.py \
        --output_file $breakdown_dir/breakdown_small.csv \
        --d_model 768 \
        --d_ff 3072 \
        --num_heads 12 \
        --context_length $context_length > $breakdown_dir/breakdown_small.log 2>&1

    uv run ffn_attention_breakdown.py \
        --output_file $breakdown_dir/breakdown_large.csv \
        --d_model 1280 \
        --d_ff 5120 \
        --num_heads 20 \
        --context_length $context_length > $breakdown_dir/breakdown_large.log 2>&1
done
