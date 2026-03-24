"""Tests for the evaluator (time simulator)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transpiler.parser import parse
from transpiler.evaluator import evaluate, Evaluator, SoundEvent

SONIC_PI_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'sonic-pi')


def ev(source, iters=4, seed=42):
    """Helper: parse + evaluate source, return event list."""
    return evaluate(parse(source), sonic_pi_root=SONIC_PI_ROOT, live_loop_iters=iters, rng_seed=seed)


# ── Timing ──────────────────────────────────────────────────────────────────

def test_sleep_advances_time():
    events = ev("play 60\nsleep 1\nplay 64")
    times = [e.time for e in events]
    assert times[0] == 0.0
    assert times[1] > 0.0


def test_use_bpm_scales_time():
    # BPM 60 → 1 beat = 1s;  BPM 120 → 1 beat = 0.5s
    e60 = ev("use_bpm 60\nplay 60\nsleep 1\nplay 64")
    e120 = ev("use_bpm 120\nplay 60\nsleep 1\nplay 64")
    assert e120[1].time == pytest_approx(e60[1].time / 2)


def pytest_approx(v):
    """Simple approximate comparison."""
    class _Approx:
        def __init__(self, val): self.val = val
        def __eq__(self, other): return abs(other - self.val) < 1e-6
    return _Approx(v)


def test_bpm_default_60():
    events = ev("play 60\nsleep 2\nplay 64")
    assert events[0].time == 0.0
    assert abs(events[1].time - 2.0) < 1e-6   # 2 beats × 1s/beat = 2s


def test_bpm_120():
    events = ev("use_bpm 120\nplay 60\nsleep 2\nplay 64")
    assert events[0].time == 0.0
    assert abs(events[1].time - 1.0) < 1e-6   # 2 beats × 0.5s/beat = 1s


# ── Synths ──────────────────────────────────────────────────────────────────

def test_play_emits_synth_event():
    events = ev("play 60")
    assert len(events) == 1
    assert events[0].kind == 'synth'


def test_play_uses_current_synth():
    events = ev("use_synth :saw\nplay 60")
    assert events[0].synth_name == 'sonic-pi-saw'


def test_play_default_synth_is_beep():
    events = ev("play 60")
    assert events[0].synth_name == 'sonic-pi-beep'


def test_play_note_symbol():
    events = ev("play :C4")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_play_note_Eb4():
    events = ev("play :Eb4")
    assert abs(events[0].args['note'] - 63.0) < 0.01


def test_synth_explicit():
    events = ev("synth :fm, note: 48, amp: 0.5")
    assert events[0].synth_name == 'sonic-pi-fm'
    assert events[0].args['amp'] == 0.5


def test_play_chord_emits_multiple():
    events = ev("play chord(:C4, :major)")
    assert len(events) == 3   # C E G


# ── Samples ─────────────────────────────────────────────────────────────────

def test_sample_emits_event():
    events = ev("sample :bd_haus")
    assert len(events) == 1
    assert events[0].kind == 'sample'


def test_sample_default_rate():
    events = ev("sample :bd_haus")
    assert events[0].args['rate'] == 1.0


def test_sample_with_rate():
    events = ev("sample :bd_haus, rate: 2.0")
    assert events[0].args['rate'] == 2.0


def test_sample_with_amp():
    events = ev("sample :bd_haus, amp: 0.5")
    assert events[0].args['amp'] == 0.5


# ── FX ──────────────────────────────────────────────────────────────────────

def test_with_fx_emits_fx_open_close():
    events = ev("with_fx :reverb do\n  play 60\nend")
    kinds = [e.kind for e in events]
    assert 'fx_open' in kinds
    assert 'fx_close' in kinds


def test_with_fx_bus_routing():
    events = ev("with_fx :reverb do\n  play 60\nend")
    fx_open  = next(e for e in events if e.kind == 'fx_open')
    synth    = next(e for e in events if e.kind == 'synth')
    # Synth writes to the FX's in_bus
    assert synth.args.get('out_bus') == fx_open.bus_in


def test_with_fx_close_after_block():
    events = ev("with_fx :reverb do\n  play 60\nend")
    fx_open  = next(e for e in events if e.kind == 'fx_open')
    fx_close = next(e for e in events if e.kind == 'fx_close')
    assert fx_close.node_id == fx_open.node_id
    assert fx_close.time >= fx_open.time


# ── live_loop ────────────────────────────────────────────────────────────────

def test_live_loop_unrolls():
    events = ev("live_loop :x do\n  play 60\n  sleep 1\nend", iters=4)
    synths = [e for e in events if e.kind == 'synth']
    assert len(synths) == 4


def test_live_loop_time_advances():
    events = ev("live_loop :x do\n  play 60\n  sleep 1\nend", iters=3)
    synths = sorted([e for e in events if e.kind == 'synth'], key=lambda e: e.time)
    assert abs(synths[1].time - synths[0].time - 1.0) < 1e-6


# ── in_thread ────────────────────────────────────────────────────────────────

def test_in_thread_does_not_advance_parent_time():
    src = "in_thread do\n  sleep 10\nend\nplay 60"
    events = ev(src)
    synth = next(e for e in events if e.kind == 'synth')
    assert synth.time == 0.0


def test_in_thread_emits_events():
    src = "in_thread do\n  play 60\nend"
    events = ev(src)
    assert len(events) == 1


# ── times loop ──────────────────────────────────────────────────────────────

def test_times_loop():
    events = ev("4.times do\n  play 60\n  sleep 0.5\nend")
    synths = [e for e in events if e.kind == 'synth']
    assert len(synths) == 4


# ── Music theory ─────────────────────────────────────────────────────────────

def test_chord_helper():
    events = ev("play chord(:C4, :major)")
    notes = sorted(e.args['note'] for e in events)
    assert abs(notes[0] - 60.0) < 0.01  # C4
    assert abs(notes[1] - 64.0) < 0.01  # E4
    assert abs(notes[2] - 67.0) < 0.01  # G4


def test_scale_helper():
    events = ev("notes = scale(:C4, :major)\nplay notes[0]")
    assert events[0].args['note'] == 60.0


# ── Event ordering ───────────────────────────────────────────────────────────

def test_events_sorted_by_time():
    src = """
in_thread do
  sleep 0.5
  play 64
end
play 60
sleep 1
play 67
"""
    events = ev(src)
    times = [e.time for e in events]
    assert times == sorted(times)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
