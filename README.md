# Sonic Pi → SuperCollider NRT Transpiler

Converts Sonic Pi DSL code into SuperCollider Non-Real-Time (NRT) render scripts (`.scd`), enabling headless, server-side audio rendering without running the full Sonic Pi application.

**No sound card required.** SuperCollider's NRT mode writes directly to disk.

## How It Works

```
Sonic Pi source (.rb)
        │
        ▼
  ┌─────────────┐
  │   Tokenizer │  tokenizer.py  →  token stream
  │   + Parser  │  parser.py     →  AST (ast_nodes.py)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  Evaluator  │  evaluator.py  →  list[SoundEvent]
  │             │  simulates time, BPM, FX buses, live_loop unrolling
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │   Codegen   │  codegen.py    →  SuperCollider NRT .scd script
  └─────────────┘
        │
        ▼
SuperCollider .scd  →  sclang  →  WAV
```

The evaluator is a Python interpreter for the Sonic Pi DSL. It does not execute real Ruby — it walks the AST, simulates time, and collects `SoundEvent` objects (`synth`, `sample`, `fx_open`, `fx_close`, `control`). The codegen then serialises these as SuperCollider OSC bundles for NRT rendering.

## Requirements

- Python 3.10+
- SuperCollider (`sclang` + `scsynth`) — only needed to actually render audio, not to transpile

## Quick Start

```bash
# Transpile only (produces .scd)
python cli.py examples/simple_beat.rb output.wav

# Transpile and render to WAV immediately
python cli.py examples/simple_beat.rb output.wav --run

# With a specific sonic-pi install for sample path resolution
python cli.py my_beat.rb output.wav --sonic-pi-root /path/to/sonic-pi --run
```

From Python:

```python
from transpiler import transpile

scd = transpile(
    source=open("my_beat.rb").read(),
    output_wav="output.wav",
    sonic_pi_root="/path/to/sonic-pi",
)
open("my_beat.scd", "w").write(scd)
# Then: sclang my_beat.scd
```

## CLI Reference

```
python cli.py input.rb output.wav [options]

--sonic-pi-root PATH   Path to sonic-pi repo root (for sample resolution)
--iters N              live_loop unroll count (default: 8)
--sample-rate HZ       Output sample rate (default: 44100)
--tail SECS            Silence appended after last event (default: 2.0)
--seed N               RNG seed for deterministic random calls (default: 42)
--scd PATH             Write .scd to this path (default: replaces .rb extension)
--run                  Execute the .scd with sclang after generation
--sclang PATH          Path to sclang executable (default: sclang)
```

## Supported Sonic Pi DSL

### Playback

| Feature | Notes |
|---------|-------|
| `play note, **opts` | `:C4`, MIDI integers, `chord(...).choose` |
| `synth :name, note: n, **opts` | All built-in synth names |
| `sample :name, **opts` | `rate`, `amp`, `start`, `finish`, `attack`, `release` |
| `play_chord notes, **opts` | Plays all notes simultaneously |
| `play_pattern_timed notes, times` | Sequenced pattern |
| `with_synth :name do...end` | Scoped synth change |
| `use_synth :name` | Global synth change |
| `with_fx :name, **opts do...end` | Nested FX chains, correct bus routing |
| `control node, param: val` | Emits `/n_set` control event at current time |

### Time

| Feature | Notes |
|---------|-------|
| `sleep n` | Advances time cursor |
| `use_bpm n` / `with_bpm n` | Scales all subsequent sleep values |
| `density n do...end` | Runs block n times at n× BPM |
| `at [times] do...end` | Fork execution at time offsets |

### Loops

| Feature | Notes |
|---------|-------|
| `live_loop :name do...end` | Unrolled `--iters` times (default 8) |
| `in_thread do...end` | Forked, merged back into event list |
| `N.times do...end` | |
| `loop do...end` | Unrolled up to safety limit |

### Music Theory

| Feature | Notes |
|---------|-------|
| `chord(:C4, :minor7)` | All standard chord types |
| `scale(:C4, :major, num_octaves: 2)` | All standard scale names |
| `note(:C4)` / `note(60)` | Note name → MIDI integer |
| `note_info(:C4)` | Returns note metadata |
| `midi_notes` | Chromatic scale helper |
| `spread(hits, total)` | Euclidean / Björklund rhythm |
| `ring(...)` | Circular list with `.tick` / `.look` |
| `rrand(lo, hi)` / `rrand_i(lo, hi)` | Seeded random |
| `choose` / `pick` | Random element from array/ring |

### Ruby Language

Full subset of Ruby syntax is supported:

- Variables, arithmetic, string interpolation (`"#{expr}"`)
- `if/elsif/else`, `case/when`, ternary `? :`
- `while`, `until`, `for x in`, `loop`, `N.times`, `each`, `each_with_index`, `map`, `select`, `reject`, `reduce`, `flat_map`, `zip`, `group_by`, `tally`, `each_cons`, `chunk`
- `def` / `return`, default params, splat `*rest`, `&block` params
- `lambda { }`, `-> (x) { }`, `proc { }`, `Proc.new { }`; `curry`, `arity`
- `begin/rescue/ensure/end`, `raise`
- `yield` inside `def`
- String: `gsub`, `sub`, `split`, `strip`, `upcase`, `downcase`, `include?`, `scan`, `tr`, `%` formatting
- Array: `push`, `pop`, `flatten`, `uniq`, `sort`, `reverse`, `rotate`, `combination`, `permutation`, `product`, `partition`, `mirror`, `reflect`, `stretch`, `butlast`
- Hash: `merge`, `merge!`, `keys`, `values`, `each_pair`, `select`, `map`
- `puts`, `print`, `p`, `pp`

### MIDI Output

```ruby
midi :C4, sustain: 0.5
midi_note_on 60, 100          # note, velocity
midi_note_off 60
midi_cc 74, 100               # controller, value
```

MIDI calls emit `SoundEvent(kind='midi_*')` objects. Mapping to a real MIDI device requires post-processing the event list.

## Limitations vs Real Sonic Pi

These features are intentionally NRT-incompatible and are silently no-ops or approximated:

| Feature | Behaviour |
|---------|-----------|
| `cue` / `sync` | No-op (no real-time thread synchronisation) |
| `live_audio` | No-op |
| `use_tuning` | Stored but does not alter MIDI note numbers |
| `in_thread` parallelism | Simulated sequentially, merged by time |

`live_loop` is unrolled a fixed number of times (see `--iters`).

## Running Tests

```bash
python -m pytest tests/ -v
# Run a single file
python -m pytest tests/test_new_features.py -v
```

422 tests covering tokenizer, parser, evaluator, music theory, and codegen.
