import torch
import einx
import torch.cuda.nvtx as nvtx
from argparse import ArgumentParser

from transformer import softmax, Transformer
from benchmark import init_model, gen_batch
from util import AdamW, cross_entropy_loss


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.size(-1)
    with nvtx.range("computing attention scores"):
        attention_scores: torch.Tensor = einx.dot(
            "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k", query, key
        ) / (d_k**0.5)

    if mask is not None:
        attention_scores.masked_fill_(mask == 0, float("-inf"))

    with nvtx.range("computing softmax"):
        probs = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einx.dot(
            "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v", probs, value
        )
    return output


def warmup_step(
    model: Transformer,
    optimizer: AdamW | None,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    skip_backward: bool = False,
    skip_optim: bool = False,
) -> None:
    if skip_backward:
        assert skip_optim
    if not skip_optim:
        assert optimizer is not None
        optimizer.zero_grad()

    logits = model(input_ids)

    if skip_backward:
        return

    loss = cross_entropy_loss(logits, target_ids)
    loss.backward()

    if skip_optim:
        return

    optimizer.step()


def profile_step(
    model: Transformer,
    optimizer: AdamW | None,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    skip_backward: bool = False,
    skip_optim: bool = False,
) -> None:
    if skip_backward:
        assert skip_optim
    if not skip_optim:
        assert optimizer is not None
        optimizer.zero_grad()

    with nvtx.range("forward"):
        logits = model(input_ids)

    if skip_backward:
        return

    with nvtx.range("loss computation"):
        loss = cross_entropy_loss(logits, target_ids)

    with nvtx.range("backward"):
        loss.backward()

    if skip_optim:
        return

    with nvtx.range("optimizer step"):
        optimizer.step()


def nsys_main():
    parser = ArgumentParser()
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--skip_optim", action="store_true")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    skip_backward = args.skip_backward
    skip_optim = args.skip_optim
    if skip_backward:
        skip_optim = True
    print(f"skip_backward={skip_backward},skip_optim={skip_optim}")

    assert torch.cuda.is_available(), "NSYS profiling requires a CUDA-capable device."
    device = torch.device("cuda")
    model = init_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        device=device,
        attn_fn=annotated_scaled_dot_product_attention,
    )
    if skip_optim:
        optimizer = None
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    input_ids, target_ids = gen_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )
    for _ in range(args.warmup_steps):
        warmup_step(
            model,
            optimizer,
            input_ids,
            target_ids,
            skip_backward=skip_backward,
            skip_optim=skip_optim,
        )
    for _ in range(args.benchmark_steps):
        profile_step(
            model,
            optimizer,
            input_ids,
            target_ids,
            skip_backward=skip_backward,
            skip_optim=skip_optim,
        )


if __name__ == "__main__":
    nsys_main()
