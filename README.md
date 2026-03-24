# Sonic Pi -> SuperCollider NRT Transpiler

Converts Sonic Pi DSL code into SuperCollider Non-Real-Time (NRT) render scripts (`.scd`), enabling stable offline audio rendering without running the full Sonic Pi application.

## Quick Start

```bash
python cli.py examples/simple_beat.rb output.wav --run
```

Or from Python:

```python
from transpiler import transpile

scd = transpile(
    source=open("my_beat.rb").read(),
    output_wav="/tmp/output.wav",
    sonic_pi_root="/path/to/sonic-pi",
)
open("my_beat.scd", "w").write(scd)
# Then: sclang my_beat.scd
```

## CLI Options

```
python cli.py input.rb output.wav [options]

--sonic-pi-root PATH   Path to sonic-pi repo (auto-detected if omitted)
--iters N              live_loop unroll count (default: 8)
--sample-rate HZ       Output sample rate (default: 44100)
--tail SECS            Silence after last event (default: 2.0)
--seed N               RNG seed for deterministic output (default: 42)
--scd PATH             Write .scd to this path (default: <input>.scd)
--run                  Execute the .scd with sclang after generation
--sclang PATH          Path to sclang (default: sclang)
```

## Supported Sonic Pi DSL

| Feature | Example |
|---------|---------|
| `play` | `play 60`, `play :C4`, `play chord(:C4, :major)` |
| `synth` | `synth :fm, note: 60, amp: 0.8` |
| `sample` | `sample :bd_haus, amp: 1.0, rate: 0.5` |
| `with_fx` | `with_fx :reverb, room: 0.8 do ... end` |
| `use_synth` | `use_synth :saw` |
| `use_bpm` | `use_bpm 120` |
| `sleep` | `sleep 0.5` |
| `live_loop` | `live_loop :name do ... end` (unrolled N times) |
| `in_thread` | `in_thread do ... end` |
| `N.times` | `4.times do ... end` |
| `chord` | `chord(:C4, :minor)` |
| `scale` | `scale(:C4, :major, num_octaves: 2)` |
| `ring` | `ring(60, 64, 67)` |
| `spread` | `spread(3, 8)` (Euclidean rhythm) |
| `rrand` | `rrand(0.3, 0.8)` |
| `choose` | `[60, 64, 67].choose` |

## Architecture

```
Sonic Pi source (.rb)
        │
        ▼
  ┌─────────────┐
  │   Parser    │  tokenizer.py + parser.py -> AST (ast_nodes.py)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  Evaluator  │  evaluator.py -> list[SoundEvent]
  │             │  (simulates time, BPM, FX buses, live_loop unrolling)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │   Codegen   │  codegen.py -> SuperCollider NRT .scd script
  └─────────────┘
        │
        ▼
SuperCollider .scd  ->  sclang  ->  WAV
```

## Running Tests

```bash
python -m pytest tests/ -v
```
