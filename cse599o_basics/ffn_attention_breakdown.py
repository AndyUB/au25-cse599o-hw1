import torch
import numpy as np
from dataclasses import dataclass
from argparse import ArgumentParser

from transformer import TransformerBlock


@dataclass
class Stats:
    avg: float
    std: float
    pct_avg: float
    pct_std: float


class TimingStats:
    def __init__(self):
        self.times: list[tuple[float, float]] = []

    def stats(self) -> tuple[Stats, Stats]:
        attn_times = [t[0] for t in self.times]
        ffn_times = [t[1] for t in self.times]
        total_times = [t[0] + t[1] for t in self.times]

        attn_avg, attn_std = np.mean(attn_times), np.std(attn_times)
        ffn_avg, ffn_std = np.mean(ffn_times), np.std(ffn_times)

        attn_pcts = [
            attn_time / total_time
            for attn_time, total_time in zip(attn_times, total_times)
        ]
        ffn_pcts = [1 - attn_pct for attn_pct in attn_pcts]
        attn_pct_avg, attn_pct_std = np.mean(attn_pcts), np.std(attn_pcts)
        ffn_pct_avg, ffn_pct_std = np.mean(ffn_pcts), np.std(ffn_pcts)

        return Stats(
            avg=attn_avg,
            std=attn_std,
            pct_avg=attn_pct_avg,
            pct_std=attn_pct_std,
        ), Stats(
            avg=ffn_avg,
            std=ffn_std,
            pct_avg=ffn_pct_avg,
            pct_std=ffn_pct_std,
        )

    def reset(self):
        self.times = []


class FlopStats:
    def __init__(self):
        self.flops: dict[torch.Size, tuple[int, int]] = {}
        self.flop_list: list[tuple[int, int]] = []

    def stats(self) -> tuple[Stats, Stats]:
        attn_flops = [f[0] for f in self.flop_list]
        ffn_flops = [f[1] for f in self.flop_list]
        total_flops = [f[0] + f[1] for f in self.flop_list]

        attn_avg, attn_std = np.mean(attn_flops), np.std(attn_flops)
        ffn_avg, ffn_std = np.mean(ffn_flops), np.std(ffn_flops)

        attn_pcts = [
            attn_flop / total_flop
            for attn_flop, total_flop in zip(attn_flops, total_flops)
        ]
        ffn_pcts = [1 - attn_pct for attn_pct in attn_pcts]
        attn_pct_avg, attn_pct_std = np.mean(attn_pcts), np.std(attn_pcts)
        ffn_pct_avg, ffn_pct_std = np.mean(ffn_pcts), np.std(ffn_pcts)

        return Stats(
            avg=attn_avg,
            std=attn_std,
            pct_avg=attn_pct_avg,
            pct_std=attn_pct_std,
        ), Stats(
            avg=ffn_avg,
            std=ffn_std,
            pct_avg=ffn_pct_avg,
            pct_std=ffn_pct_std,
        )

    def reset(self):
        self.flop_list = []
        self.flops = {}


def matmul_flops(a: int, b: int, c: int) -> int:
    return 2 * a * b * c


class TransformerBlockWithTiming(TransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_stats: TimingStats = TimingStats()
        self.flop_stats: FlopStats = FlopStats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size() not in self.flop_stats.flops:
            flop_attn, flop_ffn = self.compute_flops(x)
            self.flop_stats.flops[x.size()] = (flop_attn, flop_ffn)
        self.flop_stats.flop_list.append(self.flop_stats.flops[x.size()])

        attn_start_ev = torch.cuda.Event(enable_timing=True)
        attn_end_ev = torch.cuda.Event(enable_timing=True)
        ffn_start_ev = torch.cuda.Event(enable_timing=True)
        ffn_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        attn_start_ev.record()
        n1 = self.ln1(x)
        attn_output = self.attn(n1)
        y = x + attn_output
        attn_end_ev.record()

        ffn_start_ev.record()
        n2 = self.ln2(y)
        z = self.ffn(n2)
        output = y + z
        ffn_end_ev.record()

        torch.cuda.synchronize()

        attn_time = attn_start_ev.elapsed_time(attn_end_ev)
        ffn_time = ffn_start_ev.elapsed_time(ffn_end_ev)
        self.time_stats.times.append((attn_time, ffn_time))

        return output

    def compute_flops(self, x: torch.Tensor) -> tuple[int, int]:
        *batch_dims, seq_length, d_model = x.size()
        total_batch_dim = np.prod(batch_dims)
        num_heads = self.attn.num_heads
        d_head = d_model // num_heads
        d_ff = self.ffn.w1.weight.shape[0]

        # {q_proj, k_proj, v_proj, output_proj}.weight (d_model, d_model)
        proj_flops = 4 * total_batch_dim * matmul_flops(seq_length, d_model, d_model)
        # {q, k, v} (*batch_dims, num_heads, seq_length, d_head)
        qk_flops = (
            total_batch_dim * num_heads * matmul_flops(seq_length, d_head, seq_length)
        )
        # probs (*batch_dims, num_heads, seq_length, seq_length)
        probv_flops = (
            total_batch_dim * num_heads * matmul_flops(seq_length, seq_length, d_head)
        )
        attn_flops = proj_flops + qk_flops + probv_flops

        # input to ffn (*batch_dims, seq_length, d_model)
        # {w1, w3}.weight (d_ff, d_model)
        # w2.weight (d_model, d_ff)
        # *batch_dims, (seq_length, d_model) * (d_model, d_ff) -> (seq_length, d_ff)
        # *batch_dims, (seq_length, d_ff) * (d_ff, d_model) -> (seq_length, d_model)
        # *batch_dims, (seq_length, d_model) * (d_model, d_ff) -> (seq_length, d_ff)
        ffn_flops = 3 * total_batch_dim * matmul_flops(seq_length, d_model, d_ff)

        print(
            f"Input size: {x.size()}, "
            f"num_heads: {num_heads}, d_head: {d_head}, d_ff: {d_ff}, "
            f"Attn FLOPs: {attn_flops}, projs FLOPs: {proj_flops}, "
            f"QK FLOPs: {qk_flops}, ProbsV FLOPs: {probv_flops}, "
            f"FFN FLOPs: {ffn_flops}"
        )

        return attn_flops, ffn_flops


def compare_stats(flops: Stats, time: Stats, output_file: str, component: str):
    with open(output_file, "a") as f:
        f.write(
            f"{component},{flops.avg / 1e6:<.2f}({flops.std / 1e6:<.2f}),"
            f"{time.avg:.2f}({time.std:.2f}),"
            f"{100 * flops.pct_avg:.2f}({100 * flops.pct_std:.2f}),"
            f"{100 * time.pct_avg:.2f}({100 * time.pct_std:.2f}),"
            f"{100 * (flops.pct_avg - time.pct_avg):.2f}\n"
        )


def breakdown_main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    args = parser.parse_args()
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = TransformerBlockWithTiming(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        device=device,
    )
    for step in range(args.warmup_steps + args.benchmark_steps):
        if step == args.warmup_steps:
            block.time_stats.reset()
            block.flop_stats.reset()
        x = torch.randn(
            args.batch_size,
            args.context_length,
            args.d_model,
            device=device,
        )
        _ = block(x)

    time_attn, time_ffn = block.time_stats.stats()
    flop_attn, flop_ffn = block.flop_stats.stats()
    with open(args.output_file, "w") as f:
        f.write("component,MFLOPs,time,flops_pct,time_pct,pct_diff\n")
    compare_stats(
        flops=flop_attn,
        time=time_attn,
        output_file=args.output_file,
        component="attention",
    )
    compare_stats(
        flops=flop_ffn,
        time=time_ffn,
        output_file=args.output_file,
        component="ffn",
    )


if __name__ == "__main__":
    breakdown_main()
