#!/usr/bin/env python3
"""
Sweep FlexLLMGen OPT-6.7B prompt length until the first OOM/crash.

Run this file from inside the FlexLLMGen directory:
    python3 run_flexllmgen_opt67b_max_length.py

Baseline policy:
    --percent 100 0 100 0 100 0
    --gpu-batch-size 1
    --num-gpu-batches 1
    --gen-len 512

This keeps model weights, KV cache, and activations on GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


MODEL_NAME = "facebook/opt-6.7b"
DEFAULT_PERCENT = ["100", "0", "100", "0", "100", "0"]

FLEXLLMGEN_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = FLEXLLMGEN_ROOT / "output_data"
LOG_PATH = OUTPUT_DIR / "baseline_opt6.7b_max_length_log.txt"
CSV_PATH = OUTPUT_DIR / "baseline_opt6.7b_max_length_data.csv"
PLOT_PATH = OUTPUT_DIR / "baseline_opt6.7b_vram_breakdown.png"
RAW_LOG_PATH = OUTPUT_DIR / "baseline_opt6.7b_flexllmgen_raw.log"


@dataclass
class RunMetrics:
    prompt_len: int
    gen_len: int
    batch_size: int
    status: str
    total_throughput: Optional[float] = None
    model_weight_gb: Optional[float] = None
    kv_cache_gb: Optional[float] = None
    activation_gb: Optional[float] = None
    peak_gpu_mem_gb: Optional[float] = None
    return_code: int = 0
    command: str = ""
    error_hint: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the max no-offload FlexLLMGen OPT-6.7B prompt length."
    )
    parser.add_argument("--start-prompt-len", type=int, default=512)
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--gen-len", type=int, default=512)
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir")
    parser.add_argument("--max-prompt-len", type=int, default=None)
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run python -m flexllmgen.flex_opt.",
    )
    return parser.parse_args()


def ensure_output_files() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "timestamp",
                    "prompt_len",
                    "gen_len",
                    "batch_size",
                    "status",
                    "total_throughput_token_s",
                    "model_weight_gb",
                    "kv_cache_gb",
                    "activation_gb",
                    "stacked_vram_gb",
                    "peak_gpu_mem_gb",
                    "return_code",
                    "error_hint",
                ]
            )

    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# FlexLLMGen OPT-6.7B no-offload max prompt length sweep\n"
            "# percent = 100 0 100 0 100 0 "
            "(weight/cache/activation all on GPU)\n"
            "# SUCCESS rows contain prompt_len and total_throughput_token_s.\n"
            "# CSV mirror: baseline_opt6.7b_max_length_data.csv\n\n",
            encoding="utf-8",
        )


def build_command(args: argparse.Namespace, prompt_len: int) -> list[str]:
    return [
        args.python,
        "-u",
        "-m",
        "flexllmgen.flex_opt",
        "--model",
        args.model,
        "--path",
        args.path,
        "--offload-dir",
        args.offload_dir,
        "--prompt-len",
        str(prompt_len),
        "--gen-len",
        str(args.gen_len),
        "--gpu-batch-size",
        str(args.gpu_batch_size),
        "--num-gpu-batches",
        str(args.num_gpu_batches),
        "--percent",
        *DEFAULT_PERCENT,
        "--log-file",
        str(RAW_LOG_PATH),
        "--verbose",
        "1",
    ]


def run_command(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(FLEXLLMGEN_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)

    return proc.wait(), "".join(captured)


def last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(value)
    except ValueError:
        return None


def parse_success_metrics(
    prompt_len: int,
    gen_len: int,
    batch_size: int,
    return_code: int,
    cmd: list[str],
    output: str,
) -> RunMetrics:
    size_matches = re.findall(
        r"model size:\s*([0-9.]+)\s*GB(?:,|\t)\s*"
        r"cache size:\s*([0-9.]+)\s*GB(?:,|\t)\s*"
        r"hidden size\s*\((?:p|prefill)\):\s*([0-9.]+)\s*GB",
        output,
        flags=re.IGNORECASE,
    )

    model_weight_gb = kv_cache_gb = activation_gb = None
    if size_matches:
        model_weight_gb, kv_cache_gb, activation_gb = [
            float(x) for x in size_matches[-1]
        ]

    peak_gpu_mem_gb = last_float(r"peak gpu mem:\s*([0-9.]+)\s*GB", output)
    total_throughput = last_float(
        r"total throughput:\s*([0-9.]+)\s*token/s", output
    )

    if total_throughput is None:
        return RunMetrics(
            prompt_len=prompt_len,
            gen_len=gen_len,
            batch_size=batch_size,
            status="PARSE_ERROR",
            return_code=return_code,
            command=" ".join(cmd),
            error_hint="completed but total throughput was not found",
        )

    return RunMetrics(
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch_size=batch_size,
        status="SUCCESS",
        total_throughput=total_throughput,
        model_weight_gb=model_weight_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        peak_gpu_mem_gb=peak_gpu_mem_gb,
        return_code=return_code,
        command=" ".join(cmd),
    )


def parse_failure_metrics(
    prompt_len: int,
    gen_len: int,
    batch_size: int,
    return_code: int,
    cmd: list[str],
    output: str,
) -> RunMetrics:
    lower_output = output.lower()
    if "cuda out of memory" in lower_output or "out of memory" in lower_output:
        status = "OOM"
        error_hint = "CUDA out of memory detected"
    else:
        status = "FAILED"
        tail = " ".join(output.strip().splitlines()[-3:])
        error_hint = tail[:300] if tail else "non-zero exit code"

    return RunMetrics(
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch_size=batch_size,
        status=status,
        return_code=return_code,
        command=" ".join(cmd),
        error_hint=error_hint,
    )


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def append_metrics(metrics: RunMetrics) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    stacked_vram = None
    if (
        metrics.model_weight_gb is not None
        and metrics.kv_cache_gb is not None
        and metrics.activation_gb is not None
    ):
        stacked_vram = (
            metrics.model_weight_gb + metrics.kv_cache_gb + metrics.activation_gb
        )

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                timestamp,
                metrics.prompt_len,
                metrics.gen_len,
                metrics.batch_size,
                metrics.status,
                metrics.total_throughput,
                metrics.model_weight_gb,
                metrics.kv_cache_gb,
                metrics.activation_gb,
                stacked_vram,
                metrics.peak_gpu_mem_gb,
                metrics.return_code,
                metrics.error_hint,
            ]
        )

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}] prompt_len={metrics.prompt_len}, "
            f"gen_len={metrics.gen_len}, batch_size={metrics.batch_size}, "
            f"status={metrics.status}, "
            f"total_throughput_token_s={fmt(metrics.total_throughput)}, "
            f"model_weight_gb={fmt(metrics.model_weight_gb)}, "
            f"kv_cache_gb={fmt(metrics.kv_cache_gb)}, "
            f"activation_gb={fmt(metrics.activation_gb)}, "
            f"stacked_vram_gb={fmt(stacked_vram)}, "
            f"peak_gpu_mem_gb={fmt(metrics.peak_gpu_mem_gb)}, "
            f"return_code={metrics.return_code}, "
            f"error_hint={metrics.error_hint}\n"
        )
        f.write(f"command={metrics.command}\n\n")


def plot_results() -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        print(f"Skip plotting because a plotting dependency is missing: {exc}")
        return

    if not CSV_PATH.exists():
        return

    df = pd.read_csv(CSV_PATH)
    success = df[df["status"] == "SUCCESS"].copy()
    if success.empty:
        print("No successful rows yet; skip plot.")
        return

    for col in [
        "prompt_len",
        "model_weight_gb",
        "kv_cache_gb",
        "activation_gb",
        "peak_gpu_mem_gb",
    ]:
        success[col] = pd.to_numeric(success[col], errors="coerce")

    success = success.dropna(
        subset=["prompt_len", "model_weight_gb", "kv_cache_gb", "activation_gb"]
    )
    if success.empty:
        print("Successful rows do not contain complete VRAM data; skip plot.")
        return

    success = (
        success.sort_values("timestamp")
        .groupby("prompt_len", as_index=False)
        .tail(1)
        .sort_values("prompt_len")
    )

    contexts = success["prompt_len"].astype(int).astype(str).to_list()
    x = np.arange(len(contexts))
    model = success["model_weight_gb"].to_numpy(dtype=float)
    kv = success["kv_cache_gb"].to_numpy(dtype=float)
    act = success["activation_gb"].to_numpy(dtype=float)
    peak = success["peak_gpu_mem_gb"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    ax.bar(x, model, label="Model Weight", color="#2f4858", edgecolor="white")
    ax.bar(x, kv, bottom=model, label="KV Cache", color="#2f80ed", edgecolor="white")
    ax.bar(
        x,
        act,
        bottom=model + kv,
        label="Activation",
        color="#f2994a",
        edgecolor="white",
    )

    if not np.isnan(peak).all():
        ax.plot(
            x,
            peak,
            color="#111111",
            marker="o",
            linewidth=1.8,
            label="FlexLLMGen Peak GPU Mem",
        )

    ax.axhline(16.0, color="#d62728", linestyle="--", linewidth=1.6, label="16GB VRAM")
    ax.set_title("OPT-6.7B No-Offload VRAM Breakdown")
    ax.set_xlabel("Context Length / Prompt Length (tokens)")
    ax.set_ylabel("VRAM Usage (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {PLOT_PATH}")


def main() -> int:
    args = parse_args()
    ensure_output_files()

    total_batch_size = args.gpu_batch_size * args.num_gpu_batches
    prompt_len = args.start_prompt_len

    print(f"Output log: {LOG_PATH}")
    print(f"Output CSV: {CSV_PATH}")
    print(f"Output plot: {PLOT_PATH}")
    print(f"Model: {args.model}")
    print(f"Percent: {' '.join(DEFAULT_PERCENT)}")
    print(f"Batch size: {total_batch_size}, gen_len: {args.gen_len}")

    while True:
        if args.max_prompt_len is not None and prompt_len > args.max_prompt_len:
            print(f"Reached --max-prompt-len={args.max_prompt_len}; stop.")
            break

        print("\n" + "=" * 88)
        print(
            f"Testing prompt_len={prompt_len}, "
            f"gen_len={args.gen_len}, batch_size={total_batch_size}"
        )

        cmd = build_command(args, prompt_len)
        print("Command:", " ".join(cmd))
        return_code, output = run_command(cmd)

        if return_code == 0:
            metrics = parse_success_metrics(
                prompt_len, args.gen_len, total_batch_size, return_code, cmd, output
            )
            append_metrics(metrics)

            if metrics.status != "SUCCESS":
                print("Maximum length reached")
                print(f"Stopped at prompt_len={prompt_len}: {metrics.error_hint}")
                break

            print(
                f"SUCCESS prompt_len={prompt_len}, "
                f"total throughput={metrics.total_throughput:.3f} token/s"
            )
            plot_results()
            prompt_len += args.step
            continue

        metrics = parse_failure_metrics(
            prompt_len, args.gen_len, total_batch_size, return_code, cmd, output
        )
        append_metrics(metrics)
        print("Maximum length reached")
        print(
            f"Stopped at prompt_len={prompt_len}, status={metrics.status}, "
            f"return_code={return_code}"
        )
        plot_results()
        break

    plot_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
