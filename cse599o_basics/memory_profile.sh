export PYTHONUNBUFFERED=1

MEMORY_PROFILE_DIR=../bench/memory_profile
mkdir -p $MEMORY_PROFILE_DIR

# remove logs and csvs from previous runs if --clean is passed
if [[ "$1" == "--clean" ]]; then
    rm -rf $MEMORY_PROFILE_DIR/*
fi

context_lengths="128 256 512"

for context_length in $context_lengths; do
    profile_dir=$MEMORY_PROFILE_DIR/inference
    mkdir -p $profile_dir
    output_dir_small=$profile_dir/small_${context_length}context
    mkdir -p $output_dir_small
    output_dir_large=$profile_dir/large_${context_length}context
    mkdir -p $output_dir_large

    # uv run python memory_profile.py \
    #     --output_dir $output_dir_small \
    #     --d_model 768 \
    #     --d_ff 3072 \
    #     --num_layers 12 \
    #     --num_heads 12 \
    #     --skip_backward \
    #     --context_length $context_length > $profile_dir/small_${context_length}context.log 2>&1

    # # if context_length is over 512, skip large model due to OOM
    # if [ $context_length -gt 512 ]; then
    #     continue
    # fi

    uv run python memory_profile.py \
        --output_dir $output_dir_large \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --skip_backward \
        --context_length $context_length > $profile_dir/large_${context_length}context.log 2>&1
done

for context_length in $context_lengths; do
    profile_dir=$MEMORY_PROFILE_DIR/training
    mkdir -p $profile_dir
    output_dir_small=$profile_dir/small_${context_length}context
    mkdir -p $output_dir_small
    output_dir_large=$profile_dir/large_${context_length}context
    mkdir -p $output_dir_large

    # uv run python memory_profile.py \
    #     --output_dir $output_dir_small \
    #     --d_model 768 \
    #     --d_ff 3072 \
    #     --num_layers 12 \
    #     --num_heads 12 \
    #     --context_length $context_length > $profile_dir/small_${context_length}context.log 2>&1

    # # if context_length is over 256, skip large model due to OOM
    # if [ $context_length -gt 256 ]; then
    #     continue
    # fi

    uv run python memory_profile.py \
        --output_dir $output_dir_large \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --context_length $context_length > $profile_dir/large_${context_length}context.log 2>&1
done
