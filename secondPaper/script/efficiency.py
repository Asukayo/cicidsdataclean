#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import argparse
import csv
import gc
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import torch


@dataclass
class ModelSpec:
    display_name: str
    module: str
    class_name: str
    kwargs_fn: Callable[[argparse.Namespace, torch.device], Dict[str, Any]]


def get_model_specs():
    return [
        ModelSpec(
            "FreqDAR",
            "secondPaper.models.FullModel",
            "FullModel",
            lambda args, device: dict(
                input_dim=args.input_dim,
                window_size=args.window_size,
                freq_infer_segments=args.freq_infer_segments,
            ),
        ),
        ModelSpec(
            "OmniAnomaly",
            "secondPaper.models.OmniAnomaly",
            "OmniAnomaly",
            lambda args, device: dict(
                feats=args.input_dim,
                device=device,
                n_hidden=args.omni_hidden,
                n_latent=args.omni_latent,
            ),
        ),
        ModelSpec(
            "LSTM-AE",
            "secondPaper.models.LSTMAE",
            "LSTMAE",
            lambda args, device: dict(
                input_dim=args.input_dim,
                hidden_dim=args.lstm_hidden,
                latent_dim=args.lstm_latent,
            ),
        ),
        ModelSpec(
            "USAD",
            "secondPaper.models.USAD",
            "USAD",
            lambda args, device: dict(
                seq_len=args.window_size,
                input_dim=args.input_dim,
                hidden_dim=args.usad_hidden,
                latent_dim=args.usad_latent,
                output_activation="identity",
                use_official_width=False,
            ),
        ),
        ModelSpec(
            "MemAE",
            "secondPaper.models.MemAE_Adapted",
            "MemAE",
            lambda args, device: dict(
                input_dim=args.input_dim,
                hidden_dim=args.memae_hidden,
                latent_dim=args.memae_latent,
                mem_dim=args.memae_memory,
            ),
        ),
        ModelSpec(
            "TranAD",
            "secondPaper.models.TranAD",
            "TranAD",
            lambda args, device: dict(
                input_dim=args.input_dim,
                window_size=args.window_size,
                n_heads=args.n_heads,
                output_activation="identity",
            ),
        ),
        ModelSpec(
            "DTAAD",
            "secondPaper.models.DTAAD",
            "DTAAD",
            lambda args, device: dict(
                input_dim=args.input_dim,
                window_size=args.window_size,
                n_heads=args.n_heads,
                output_activation="identity",
            ),
        ),
        ModelSpec(
            "TransDe",
            "secondPaper.models.TransDe",
            "TransDe",
            lambda args, device: dict(
                win_size=args.window_size,
                input_dim=args.input_dim,
                d_model=args.transde_d_model,
                n_heads=1,
                patch_size=tuple(args.transde_patch_sizes),
            ),
        ),
        ModelSpec(
            "STFT-TCAN",
            "secondPaper.models.STFTTCAN_Adapted",
            "STFTTCAN",
            lambda args, device: dict(
                input_dim=args.input_dim,
                seq_len=args.window_size,
                d_model=args.stft_d_model,
            ),
        ),
    ]


def add_project_root_to_path(project_root):
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def clear_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def import_model(spec, args, device):
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.class_name)
    model = cls(**spec.kwargs_fn(args, device))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("{} is not torch.nn.Module".format(spec.display_name))
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_infer(model, x):
    if hasattr(model, "compute_anomaly_score"):
        return model.compute_anomaly_score(x)
    if hasattr(model, "score"):
        return model.score(x)
    return model(x)


def fmt_params(n):
    return "{:.2f}M".format(n / 1e6)


def fmt_throughput(v):
    if v >= 1e6:
        return "{:.2f}M".format(v / 1e6)
    if v >= 1e3:
        return "{:.2f}K".format(v / 1e3)
    return "{:.2f}".format(v)


def benchmark_one(spec, args, device):
    clear_cuda(device)
    model = import_model(spec, args, device).to(device)
    model.eval()

    x = torch.randn(args.batch_size, args.window_size, args.input_dim, device=device)
    params = count_params(model)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model_infer(model, x)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.repeat):
            _ = model_infer(model, x)
    sync(device)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0 / float(args.repeat)
    throughput = (args.batch_size * args.window_size) / (latency_ms / 1000.0)

    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        mem = float("nan")

    del model, x
    clear_cuda(device)

    return {
        "Method": spec.display_name,
        "Params": fmt_params(params),
        "Latency(ms)": "{:.2f}".format(latency_ms),
        "Throughput(flows/s)": fmt_throughput(throughput),
        "GPU Mem(MB)": "N/A" if math.isnan(mem) else "{:.1f}".format(mem),
    }


def make_markdown(rows):
    headers = ["Method", "Params", "Latency(ms)", "Throughput(flows/s)", "GPU Mem(MB)"]
    widths = {}
    for h in headers:
        widths[h] = max([len(h)] + [len(str(r[h])) for r in rows])

    def line(values):
        return "| " + " | ".join(str(v).ljust(widths[h]) for v, h in zip(values, headers)) + " |"

    out = [line(headers)]
    out.append("| " + " | ".join("-" * widths[h] for h in headers) + " |")
    for r in rows:
        out.append(line([r[h] for h in headers]))
    return "\n".join(out)


def make_latex(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{COMPUTATIONAL EFFICIENCY COMPARISON}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & Params & Latency(ms) & Throughput(flows/s) & GPU Mem(MB) \\",
        r"\hline",
    ]
    for r in rows:
        lines.append("{} & {} & {} & {} & {} \\\\".format(
            r["Method"], r["Params"], r["Latency(ms)"],
            r["Throughput(flows/s)"], r["GPU Mem(MB)"]
        ))
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def save_outputs(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headers = ["Method", "Params", "Latency(ms)", "Throughput(flows/s)", "GPU Mem(MB)"]

    with (out / "efficiency_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r[h] for h in headers})

    (out / "efficiency_table.md").write_text(make_markdown(rows), encoding="utf-8")
    (out / "efficiency_table.tex").write_text(make_latex(rows), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--input_dim", type=int, default=68)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="./efficiency_results")

    p.add_argument("--omni_hidden", type=int, default=128)
    p.add_argument("--omni_latent", type=int, default=32)

    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_latent", type=int, default=64)

    p.add_argument("--usad_hidden", type=int, default=64)
    p.add_argument("--usad_latent", type=int, default=16)

    p.add_argument("--memae_hidden", type=int, default=64)
    p.add_argument("--memae_latent", type=int, default=64)
    p.add_argument("--memae_memory", type=int, default=64)

    p.add_argument("--n_heads", type=int, default=4)

    p.add_argument("--transde_d_model", type=int, default=128)
    p.add_argument("--transde_patch_sizes", type=int, nargs="+", default=[5, 10, 20])

    p.add_argument("--stft_d_model", type=int, default=64)
    p.add_argument("--freq_infer_segments", type=int, default=2)

    p.add_argument("--include", type=str, nargs="*", default=None)
    p.add_argument("--exclude", type=str, nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    add_project_root_to_path(args.project_root)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    include = set(args.include) if args.include else None
    exclude = set(args.exclude or [])

    print("=" * 80)
    print("Computational efficiency benchmark")
    print("device      : {}".format(device))
    print("batch_size  : {}".format(args.batch_size))
    print("window_size : {}".format(args.window_size))
    print("input_dim   : {}".format(args.input_dim))
    print("warmup      : {}".format(args.warmup))
    print("repeat      : {}".format(args.repeat))
    print("=" * 80)

    rows = []
    for spec in get_model_specs():
        if include is not None and spec.display_name not in include:
            continue
        if spec.display_name in exclude:
            continue

        print("\n[RUN] {}".format(spec.display_name))
        try:
            row = benchmark_one(spec, args, device)
            rows.append(row)
            print("  Params={}, Latency={} ms, Throughput={} flows/s, GPU Mem={} MB".format(
                row["Params"], row["Latency(ms)"], row["Throughput(flows/s)"], row["GPU Mem(MB)"]
            ))
        except Exception as e:
            print("  [SKIP] {}: {}: {}".format(spec.display_name, type(e).__name__, e))

    if not rows:
        raise RuntimeError("No model was successfully benchmarked.")

    print("\nTable:")
    print(make_markdown(rows))
    save_outputs(rows, args.out_dir)
    print("\nSaved to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
