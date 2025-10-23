import torch
import numpy as np
from argparse import ArgumentParser
from dataclasses import dataclass
from timeit import default_timer

from transformer import Transformer
from util import cross_entropy_loss, AdamW

VOCAB_SIZE = 50257
BATCH_SIZE = 4


def init_model(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    context_length: int,
    device: torch.device,
    **kwargs,
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
        **kwargs,
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
    optim_time: float | None


def benchmark_step(
    model: Transformer,
    optimizer: AdamW,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    time_forward: bool = True,
    time_backward: bool = True,
    time_loss: bool = True,
    time_optim: bool = True,
    skip_loss: bool = False,
    skip_backward: bool = False,
    skip_optim: bool = False,
) -> tuple[TimeBreakdown, float]:
    fwd_start_ev = None
    fwd_end_ev = None
    bwd_start_ev = None
    bwd_end_ev = None
    loss_start_ev = None
    loss_end_ev = None
    optim_start_ev = None
    optim_end_ev = None

    if skip_loss:
        assert not time_loss
        assert skip_backward
    if skip_backward:
        assert not time_backward
        assert skip_optim
    if skip_optim:
        assert not time_optim

    if time_forward:
        fwd_start_ev = torch.cuda.Event(enable_timing=True)
        fwd_end_ev = torch.cuda.Event(enable_timing=True)
    if time_backward:
        bwd_start_ev = torch.cuda.Event(enable_timing=True)
        bwd_end_ev = torch.cuda.Event(enable_timing=True)
    if time_loss:
        loss_start_ev = torch.cuda.Event(enable_timing=True)
        loss_end_ev = torch.cuda.Event(enable_timing=True)
    if time_optim:
        optim_start_ev = torch.cuda.Event(enable_timing=True)
        optim_end_ev = torch.cuda.Event(enable_timing=True)

    optimizer.zero_grad()
    torch.cuda.synchronize()
    start_time = default_timer()

    if time_forward:
        fwd_start_ev.record()
    logits = model(input_ids)
    if time_forward:
        fwd_end_ev.record()

    loss = None
    if not skip_loss:
        if time_loss:
            loss_start_ev.record()
        loss = cross_entropy_loss(logits, target_ids)
        if time_loss:
            loss_end_ev.record()

    if not skip_backward:
        if time_backward:
            bwd_start_ev.record()
        loss.backward()
        if time_backward:
            bwd_end_ev.record()

    if not skip_optim:
        if time_optim:
            optim_start_ev.record()
        optimizer.step()
        if time_optim:
            optim_end_ev.record()

    torch.cuda.synchronize()
    end_time = default_timer()
    total_time = end_time - start_time  # seconds

    fwd_time = None
    bwd_time = None
    loss_time = None
    optim_time = None
    if time_forward:
        fwd_time = fwd_start_ev.elapsed_time(fwd_end_ev)  # milliseconds
    if time_backward:
        bwd_time = bwd_start_ev.elapsed_time(bwd_end_ev)  # milliseconds
    if time_loss:
        loss_time = loss_start_ev.elapsed_time(loss_end_ev)  # milliseconds
    if time_optim:
        optim_time = optim_start_ev.elapsed_time(optim_end_ev)  # milliseconds

    return TimeBreakdown(fwd_time, bwd_time, loss_time, optim_time), total_time * 1000


def benchmark(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    context_length: int,
    device: torch.device,
    warmup_steps: int,
    benchmark_steps: int,
    time_forward: bool = True,
    time_backward: bool = True,
    time_loss: bool = True,
    time_optim: bool = True,
    skip_loss: bool = False,
    skip_backward: bool = False,
    skip_optim: bool = False,
) -> tuple[
    list[tuple[TimeBreakdown, float]],
    tuple[
        tuple[float, float] | None,
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
    optim_times = []
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
            time_optim=time_optim,
            skip_loss=skip_loss,
            skip_backward=skip_backward,
            skip_optim=skip_optim,
        )
        if step >= warmup_steps:
            times.append((time_breakdown, total_time))
            fwd_times.append(time_breakdown.fwd_time)
            bwd_times.append(time_breakdown.bwd_time)
            loss_times.append(time_breakdown.loss_time)
            optim_times.append(time_breakdown.optim_time)
            total_times.append(total_time)

    fwd_stats = None
    bwd_stats = None
    loss_stats = None
    optim_stats = None
    if time_forward:
        fwd_stats = (np.mean(fwd_times), np.std(fwd_times))
    if time_backward:
        bwd_stats = (np.mean(bwd_times), np.std(bwd_times))
    if time_loss:
        loss_stats = (np.mean(loss_times), np.std(loss_times))
    if time_optim:
        optim_stats = (np.mean(optim_times), np.std(optim_times))
    total_stats = (np.mean(total_times), np.std(total_times))

    return times, (fwd_stats, bwd_stats, loss_stats, optim_stats, total_stats)


def log_benchmark_results(times: list[tuple[TimeBreakdown, float]], output_file: str):
    with open(output_file, "w") as f:
        f.write("step,fwd,bwd,loss,optim,total\n")
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
            optim_time = (
                f"{time_breakdown.optim_time:.4f}"
                if time_breakdown.optim_time is not None
                else "N/A"
            )
            f.write(f"{step+1},{fwd_time},{bwd_time},{loss_time},{optim_time},{total_time:.4f}\n")


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
    parser.add_argument("--disable_optim_timing", action="store_true")
    parser.add_argument("--skip_loss", action="store_true")
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--skip_optim", action="store_true")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    time_forward = not args.disable_fwd_timing
    time_loss = not args.disable_loss_timing
    time_backward = not args.disable_bwd_timing
    time_optim = not args.disable_optim_timing
    skip_loss = args.skip_loss
    skip_backward = args.skip_backward
    skip_optim = args.skip_optim
    if skip_loss:
        time_loss = False
        skip_backward = True
    if skip_backward:
        time_backward = False
        skip_optim = True
    if skip_optim:
        time_optim = False
    print(
        f"Timing settings: fwd={time_forward},bwd={time_backward},"
        f"loss={time_loss},optim={time_optim}"
    )
    print(
        f"Training step settings: skip_loss={skip_loss},"
        f"skip_backward={skip_backward},skip_optim={skip_optim}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    breakdown, stats = benchmark(
        model,
        optimizer,
        context_length=args.context_length,
        device=device,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        time_forward=time_forward,
        time_backward=time_backward,
        time_loss=time_loss,
        time_optim=time_optim,
        skip_loss=skip_loss,
        skip_backward=skip_backward,
        skip_optim=skip_optim,
    )
    log_benchmark_results(breakdown, args.output_file)

    fwd_stats, bwd_stats, loss_stats, total_stats = stats
    print("Benchmark Results:")
    if time_forward:
        fwd_mean, fwd_std = fwd_stats
        print(f"Fwd: mean={fwd_mean:.4f} ms, std={fwd_std:.4f} ms")
    if time_backward:
        bwd_mean, bwd_std = bwd_stats
        print(f"Bwd: mean={bwd_mean:.4f} ms, std={bwd_std:.4f} ms")
    if time_loss:
        loss_mean, loss_std = loss_stats
        print(f"Loss: mean={loss_mean:.4f} ms, std={loss_std:.4f} ms")
    if time_optim:
        optim_mean, optim_std = optim_stats
        print(f"Optim: mean={optim_mean:.4f} ms, std={optim_std:.4f} ms")
    total_mean, total_std = total_stats
    print(f"Total: mean={total_mean:.4f} ms, std={total_std:.4f} ms")


if __name__ == "__main__":
    benchmark_main()
