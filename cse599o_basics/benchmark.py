import torch
import numpy as np
from argparse import ArgumentParser
from dataclasses import dataclass
from timeit import default_timer

from transformer import Transformer
from training import cross_entropy_loss

VOCAB_SIZE = 50257
BATCH_SIZE = 4


def init_model(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    context_length: int,
    device: torch.device,
) -> Transformer:
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        context_length=context_length,
        theta=10000,
        device=device,
    )
    return model


def gen_batch(
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    target_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    return input_ids, target_ids


@dataclass
class TimeBreakdown:
    fwd_time: float | None
    bwd_time: float | None
    loss_time: float | None


def benchmark_step(
    model: Transformer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    time_forward: bool = True,
    time_backward: bool = True,
    time_loss: bool = True,
) -> tuple[TimeBreakdown, float]:
    fwd_start_ev = None
    fwd_end_ev = None
    bwd_start_ev = None
    bwd_end_ev = None
    loss_start_ev = None
    loss_end_ev = None
    if time_forward:
        fwd_start_ev = torch.cuda.Event(enable_timing=True)
        fwd_end_ev = torch.cuda.Event(enable_timing=True)
    if time_backward:
        bwd_start_ev = torch.cuda.Event(enable_timing=True)
        bwd_end_ev = torch.cuda.Event(enable_timing=True)
    if time_loss:
        loss_start_ev = torch.cuda.Event(enable_timing=True)
        loss_end_ev = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_time = default_timer()

    if time_forward:
        fwd_start_ev.record()
    logits = model(input_ids)
    if time_forward:
        fwd_end_ev.record()

    if time_loss:
        loss_start_ev.record()
    loss = cross_entropy_loss(logits, target_ids)
    if time_loss:
        loss_end_ev.record()

    if time_backward:
        bwd_start_ev.record()
    loss.backward()
    if time_backward:
        bwd_end_ev.record()

    torch.cuda.synchronize()
    end_time = default_timer()
    total_time = end_time - start_time  # seconds

    fwd_time = None
    bwd_time = None
    loss_time = None
    if time_forward:
        fwd_time = fwd_start_ev.elapsed_time(fwd_end_ev)  # milliseconds
    if time_backward:
        bwd_time = bwd_start_ev.elapsed_time(bwd_end_ev)  # milliseconds
    if time_loss:
        loss_time = loss_start_ev.elapsed_time(loss_end_ev)  # milliseconds

    return TimeBreakdown(fwd_time, bwd_time, loss_time), total_time * 1000


def benchmark(
    model: Transformer,
    context_length: int,
    device: torch.device,
    warmup_steps: int,
    benchmark_steps: int,
    time_forward: bool = True,
    time_backward: bool = True,
    time_loss: bool = True,
) -> tuple[
    list[tuple[TimeBreakdown, float]],
    tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float],
    ],
]:
    times = []
    fwd_times = []
    bwd_times = []
    loss_times = []
    total_times = []

    for step in range(warmup_steps + benchmark_steps):
        input_ids, target_ids = gen_batch(
            batch_size=BATCH_SIZE,
            context_length=context_length,
            device=device,
        )
        time_breakdown, total_time = benchmark_step(
            model,
            input_ids,
            target_ids,
            time_forward=time_forward,
            time_backward=time_backward,
            time_loss=time_loss,
        )
        if step >= warmup_steps:
            times.append((time_breakdown, total_time))
            fwd_times.append(time_breakdown.fwd_time)
            bwd_times.append(time_breakdown.bwd_time)
            loss_times.append(time_breakdown.loss_time)
            total_times.append(total_time)

    fwd_stats = None
    bwd_stats = None
    loss_stats = None
    if time_forward:
        fwd_stats = (np.mean(fwd_times), np.std(fwd_times))
    if time_backward:
        bwd_stats = (np.mean(bwd_times), np.std(bwd_times))
    if time_loss:
        loss_stats = (np.mean(loss_times), np.std(loss_times))
    total_stats = (np.mean(total_times), np.std(total_times))

    return times, (fwd_stats, bwd_stats, loss_stats, total_stats)


def log_benchmark_results(times: list[tuple[TimeBreakdown, float]], output_file: str):
    with open(output_file, "w") as f:
        f.write("step,fwd,bwd,loss,total\n")
        for step, (time_breakdown, total_time) in enumerate(times):
            fwd_time = (
                f"{time_breakdown.fwd_time:.4f}"
                if time_breakdown.fwd_time is not None
                else "N/A"
            )
            bwd_time = (
                f"{time_breakdown.bwd_time:.4f}"
                if time_breakdown.bwd_time is not None
                else "N/A"
            )
            loss_time = (
                f"{time_breakdown.loss_time:.4f}"
                if time_breakdown.loss_time is not None
                else "N/A"
            )
            f.write(f"{step+1},{fwd_time},{bwd_time},{loss_time},{total_time:.4f}\n")


def benchmark_main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument("--disable_fwd_timing", action="store_true")
    parser.add_argument("--disable_bwd_timing", action="store_true")
    parser.add_argument("--disable_loss_timing", action="store_true")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        device=device,
    )
    breakdown, stats = benchmark(
        model,
        context_length=args.context_length,
        device=device,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        time_forward=not args.disable_fwd_timing,
        time_backward=not args.disable_bwd_timing,
        time_loss=not args.disable_loss_timing,
    )
    log_benchmark_results(breakdown, args.output_file)

    fwd_stats, bwd_stats, loss_stats, total_stats = stats
    print("Benchmark Results:")
    if not args.disable_fwd_timing:
        fwd_mean, fwd_std = fwd_stats
        print(f"Fwd: mean={fwd_mean:.4f} ms, std={fwd_std:.4f} ms")
    if not args.disable_bwd_timing:
        bwd_mean, bwd_std = bwd_stats
        print(f"Bwd: mean={bwd_mean:.4f} ms, std={bwd_std:.4f} ms")
    if not args.disable_loss_timing:
        loss_mean, loss_std = loss_stats
        print(f"Loss: mean={loss_mean:.4f} ms, std={loss_std:.4f} ms")
    total_mean, total_std = total_stats
    print(f"Total: mean={total_mean:.4f} ms, std={total_std:.4f} ms")


if __name__ == "__main__":
    benchmark_main()
