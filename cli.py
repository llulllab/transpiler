#!/usr/bin/env python3
"""
Sonic Pi → SuperCollider NRT transpiler – command-line interface.

Usage:
    python cli.py input.rb output.wav [options]

Options:
    --sonic-pi-root PATH     Path to the sonic-pi repository (default: ../sonic-pi)
    --iters N                live_loop unroll count (default: 8)
    --sample-rate HZ         Output sample rate (default: 44100)
    --tail SECS              Silence after last event in seconds (default: 2.0)
    --seed N                 RNG seed for deterministic output (default: 42)
    --scd PATH               Write generated .scd to this path (default: <output>.scd)
    --run                    Execute the generated .scd with sclang after generation
    --sclang PATH            Path to sclang executable (default: sclang)
    --no-run                 Do not run sclang (default)

Examples:
    python cli.py my_beat.rb output.wav
    python cli.py my_beat.rb output.wav --iters 16 --run
    python cli.py my_beat.rb output.wav --sonic-pi-root C:/dev/sonic-pi --run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from transpiler import transpile


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Transpile Sonic Pi DSL to SuperCollider NRT script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    parser.add_argument("input",  help="Sonic Pi source file (.rb)")
    parser.add_argument("output", help="Output WAV file path")

    parser.add_argument(
        "--sonic-pi-root", default=None,
        help="Path to the sonic-pi repository root",
    )
    parser.add_argument(
        "--iters", type=int, default=8,
        help="live_loop unroll count (default: 8)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100,
        help="Output sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--tail", type=float, default=2.0,
        help="Silence after last event in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for deterministic random calls (default: 42)",
    )
    parser.add_argument(
        "--scd", default=None,
        help="Path for the generated .scd file (default: replaces .rb extension)",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run the generated .scd with sclang after generation",
    )
    parser.add_argument(
        "--sclang", default="sclang",
        help="Path to the sclang executable (default: sclang)",
    )

    args = parser.parse_args(argv)

    # ── Resolve paths ────────────────────────────────────────────────────

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_wav = str(Path(args.output).resolve())

    # Determine sonic-pi root
    sp_root = args.sonic_pi_root
    if sp_root is None:
        # Try common locations relative to this script
        candidates = [
            Path(__file__).parent.parent / "sonic-pi",
            Path(__file__).parent / ".." / "sonic-pi",
            Path.cwd() / ".." / "sonic-pi",
            Path.cwd(),
        ]
        for c in candidates:
            if (c / "etc" / "synthdefs" / "compiled").exists():
                sp_root = str(c.resolve())
                break
        if sp_root is None:
            print(
                "Warning: could not auto-detect sonic-pi root. "
                "Pass --sonic-pi-root to avoid missing synthdefs.",
                file=sys.stderr,
            )
            sp_root = "."

    # Determine .scd output path
    scd_path = args.scd
    if scd_path is None:
        scd_path = str(input_path.with_suffix('.scd'))

    # ── Read source ──────────────────────────────────────────────────────

    source = input_path.read_text(encoding='utf-8')

    # ── Transpile ────────────────────────────────────────────────────────

    print(f"Transpiling {input_path.name} -> {Path(scd_path).name} ...")

    try:
        scd_code = transpile(
            source=source,
            output_wav=output_wav,
            sonic_pi_root=sp_root,
            live_loop_iters=args.iters,
            sample_rate=args.sample_rate,
            tail_time=args.tail,
            rng_seed=args.seed,
        )
    except Exception as exc:
        print(f"Transpiler error: {exc}", file=sys.stderr)
        raise

    Path(scd_path).write_text(scd_code, encoding='utf-8')
    print(f"Written: {scd_path}")

    # Print a quick summary
    from transpiler.parser import parse
    from transpiler.evaluator import evaluate
    events = evaluate(
        parse(source),
        sonic_pi_root=sp_root,
        live_loop_iters=args.iters,
        rng_seed=args.seed,
    )
    n_synth  = sum(1 for e in events if e.kind == 'synth')
    n_sample = sum(1 for e in events if e.kind == 'sample')
    n_fx     = sum(1 for e in events if e.kind == 'fx_open')
    total_dur = max((e.time + e.duration() for e in events), default=0.0)
    print(
        f"Events: {n_synth} synth, {n_sample} sample, {n_fx} FX  |  "
        f"Duration: {total_dur + args.tail:.2f}s"
    )

    # ── Optionally run sclang ────────────────────────────────────────────

    if args.run:
        print(f"\nRunning: {args.sclang} {scd_path}")
        try:
            result = subprocess.run(
                [args.sclang, scd_path],
                check=True,
            )
        except FileNotFoundError:
            print(
                f"Error: sclang not found at '{args.sclang}'. "
                "Install SuperCollider or pass --sclang /path/to/sclang.",
                file=sys.stderr,
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"sclang exited with code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)
        print(f"Output written to: {output_wav}")


if __name__ == "__main__":
    main()
