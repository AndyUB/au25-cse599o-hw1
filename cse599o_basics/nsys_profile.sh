export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-13.0/bin:$PATH
nsys status -e

NSYS_PROFILE_DIR=../bench/nsys_profile
mkdir -p $NSYS_PROFILE_DIR

# remove logs and csvs from previous runs if --clean is passed
if [[ "$1" == "--clean" ]]; then
    rm -rf $NSYS_PROFILE_DIR/*
fi

context_lengths="128 256 512 1024"

for context_length in $context_lengths; do
    profile_dir=$NSYS_PROFILE_DIR/inference
    mkdir -p $profile_dir
    uv run nsys profile -o $profile_dir/small_${context_length}context \
        python nsys_profile.py \
        --d_model 768 \
        --d_ff 3072 \
        --num_layers 12 \
        --num_heads 12 \
        --skip_backward \
        --context_length $context_length > $profile_dir/small_${context_length}context.log 2>&1

    uv run nsys profile -o $profile_dir/large_${context_length}context \
        python nsys_profile.py \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --skip_backward \
        --context_length $context_length > $profile_dir/large_${context_length}context.log 2>&1
done

for context_length in $context_lengths; do
    profile_dir=$NSYS_PROFILE_DIR/fwd_bwd
    mkdir -p $profile_dir
    uv run nsys profile -o $profile_dir/small_${context_length}context \
        python nsys_profile.py \
        --d_model 768 \
        --d_ff 3072 \
        --num_layers 12 \
        --num_heads 12 \
        --skip_optim \
        --context_length $context_length > $profile_dir/small_${context_length}context.log 2>&1

    uv run nsys profile -o $profile_dir/large_${context_length}context \
        python nsys_profile.py \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --skip_optim \
        --context_length $context_length > $profile_dir/large_${context_length}context.log 2>&1
done

for context_length in $context_lengths; do
    profile_dir=$NSYS_PROFILE_DIR/training
    mkdir -p $profile_dir
    uv run nsys profile -o $profile_dir/small_${context_length}context \
        python nsys_profile.py \
        --d_model 768 \
        --d_ff 3072 \
        --num_layers 12 \
        --num_heads 12 \
        --context_length $context_length > $profile_dir/small_${context_length}context.log 2>&1

    uv run nsys profile -o $profile_dir/large_${context_length}context \
        python nsys_profile.py \
        --d_model 1280 \
        --d_ff 5120 \
        --num_layers 36 \
        --num_heads 20 \
        --context_length $context_length > $profile_dir/large_${context_length}context.log 2>&1
done
