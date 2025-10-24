import torch
from argparse import ArgumentParser

from transformer import Transformer
from benchmark import init_model, gen_batch
from util import AdamW, cross_entropy_loss


def step(
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

    if skip_backward:
        with torch.no_grad():
            logits = model(input_ids)
    else:
        logits = model(input_ids)

    if skip_backward:
        return

    loss = cross_entropy_loss(logits, target_ids)
    loss.backward()

    if skip_optim:
        return

    optimizer.step()


def memory_main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
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
    parser.add_argument("--record_all", action="store_true")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    output_dir = args.output_dir
    skip_backward = args.skip_backward
    skip_optim = args.skip_optim
    if skip_backward:
        skip_optim = True
    print(f"skip_backward={skip_backward},skip_optim={skip_optim}")

    assert torch.cuda.is_available(), "Memory profiling requires a CUDA GPU"
    device = torch.device("cuda")

    if args.record_all:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    model = init_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        device=device,
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
        step(
            model,
            optimizer,
            input_ids,
            target_ids,
            skip_backward=skip_backward,
            skip_optim=skip_optim,
        )

    if args.record_all:
        for i in range(args.benchmark_steps):
            step(
                model,
                optimizer,
                input_ids,
                target_ids,
                skip_backward=skip_backward,
                skip_optim=skip_optim,
            )
        torch.cuda.memory._dump_snapshot(f"{output_dir}/memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return

    for i in range(args.benchmark_steps):
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        torch.cuda.synchronize()
        step(
            model,
            optimizer,
            input_ids,
            target_ids,
            skip_backward=skip_backward,
            skip_optim=skip_optim,
        )
        torch.cuda.synchronize()

        torch.cuda.memory._dump_snapshot(f"{output_dir}/memory_snapshot_{i}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    memory_main()
