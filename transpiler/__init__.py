"""
Sonic Pi → SuperCollider NRT transpiler.

Quick-start:
    from transpiler import transpile

    scd = transpile(
        source='play :C4, release: 2',
        output_wav='/tmp/out.wav',
        sonic_pi_root='/path/to/sonic-pi',
    )
    with open('out.scd', 'w') as f:
        f.write(scd)
"""
from .parser import parse
from .evaluator import evaluate
from .codegen import generate
import os


def transpile(
    source: str,
    output_wav: str,
    sonic_pi_root: str = ".",
    live_loop_iters: int = 8,
    sample_rate: int = 44100,
    tail_time: float = 2.0,
    rng_seed: int | None = 42,
) -> str:
    """
    High-level entry point: Sonic Pi DSL source → SC NRT .scd script.

    Args:
        source:           Sonic Pi code as a string.
        output_wav:       Absolute path for the rendered WAV file.
        sonic_pi_root:    Path to the sonic-pi repository root.
        live_loop_iters:  How many times to unroll live_loops.
        sample_rate:      Output sample rate.
        tail_time:        Seconds of silence appended after the last event.
        rng_seed:         Seed for deterministic random calls.

    Returns:
        A SuperCollider .scd script as a string.
    """
    synthdef_dir = os.path.join(sonic_pi_root, "etc", "synthdefs", "compiled")

    ast = parse(source)
    events = evaluate(
        ast,
        sonic_pi_root=sonic_pi_root,
        live_loop_iters=live_loop_iters,
        rng_seed=rng_seed,
    )
    return generate(
        events=events,
        output_wav=output_wav,
        synthdef_dir=synthdef_dir,
        sample_rate=sample_rate,
        tail_time=tail_time,
    )


__all__ = ['transpile', 'parse', 'evaluate', 'generate']
