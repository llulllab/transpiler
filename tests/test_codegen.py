"""Tests for the SC NRT code generator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transpiler import transpile
from transpiler.parser import parse
from transpiler.evaluator import evaluate, SoundEvent
from transpiler.codegen import generate, SCNRTCodegen

SONIC_PI_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'sonic-pi')
SYNTHDEF_DIR = os.path.join(SONIC_PI_ROOT, 'etc', 'synthdefs', 'compiled')


def make_scd(source, iters=4):
    return transpile(
        source,
        output_wav="/tmp/test_out.wav",
        sonic_pi_root=SONIC_PI_ROOT,
        live_loop_iters=iters,
    )


# ── Structure checks ─────────────────────────────────────────────────────────

def test_scd_contains_score_recordNRT():
    scd = make_scd("play 60")
    assert "Score.recordNRT" in scd


def test_scd_contains_d_loadDir():
    scd = make_scd("play 60")
    assert "/d_loadDir" in scd


def test_scd_contains_s_new():
    scd = make_scd("play 60")
    assert "/s_new" in scd


def test_scd_no_n_free_for_synth():
    # Sonic Pi synths use doneAction:2 (self-freeing envelope) — no /n_free needed
    scd = make_scd("play 60")
    assert "/n_free" not in scd

def test_scd_n_free_for_fx():
    # FX nodes do not self-free — codegen must send /n_free at fx_close time
    scd = make_scd("with_fx :reverb do; play 60; end")
    assert "/n_free" in scd


def test_scd_opens_and_closes_parens():
    scd = make_scd("play 60")
    # The SC script wraps code in ( ... ); comments may precede the opening paren
    non_comment = "\n".join(l for l in scd.splitlines() if not l.startswith("//")).strip()
    assert non_comment.startswith("(")
    assert scd.rstrip().endswith(")")


# ── Synth names ──────────────────────────────────────────────────────────────

def test_default_synth_name_in_scd():
    scd = make_scd("play 60")
    assert "sonic-pi-beep" in scd


def test_use_synth_name_in_scd():
    scd = make_scd("use_synth :saw\nplay 60")
    assert "sonic-pi-saw" in scd


def test_explicit_synth_name():
    scd = make_scd("synth :fm, note: 60")
    assert "sonic-pi-fm" in scd


# ── Timing ───────────────────────────────────────────────────────────────────

def test_two_notes_have_different_times():
    scd = make_scd("play 60\nsleep 1\nplay 64")
    # Should contain two different /s_new timestamps
    lines = [l for l in scd.splitlines() if '/s_new' in l]
    assert len(lines) == 2
    # Extract times
    import re
    times = [float(re.search(r'\[([\d.]+),', l).group(1)) for l in lines]
    assert times[0] != times[1]


def test_bpm_affects_timing():
    scd60  = make_scd("use_bpm 60\nplay 60\nsleep 1\nplay 64")
    scd120 = make_scd("use_bpm 120\nplay 60\nsleep 1\nplay 64")
    import re
    def second_note_time(scd):
        lines = [l for l in scd.splitlines() if '/s_new' in l]
        return float(re.search(r'\[([\d.]+),', lines[1]).group(1))
    t60  = second_note_time(scd60)
    t120 = second_note_time(scd120)
    assert abs(t60 - 2 * t120) < 0.01


# ── Samples ──────────────────────────────────────────────────────────────────

def test_sample_adds_buffer_alloc():
    scd = make_scd("sample :bd_haus")
    assert "/b_allocRead" in scd


def test_sample_uses_stereo_player():
    scd = make_scd("sample :bd_haus")
    assert "basic_stereo_player" in scd


# ── FX ──────────────────────────────────────────────────────────────────────

def test_with_fx_reverb_in_scd():
    scd = make_scd("with_fx :reverb do\n  play 60\nend")
    assert "sonic-pi-fx_reverb" in scd


def test_with_fx_has_in_bus():
    scd = make_scd("with_fx :reverb do\n  play 60\nend")
    assert "in_bus" in scd


# ── Duration / end marker ────────────────────────────────────────────────────

def test_c_set_end_marker():
    scd = make_scd("play 60")
    assert "/c_set" in scd


def test_output_wav_in_scd():
    scd = make_scd("play 60")
    assert "/tmp/test_out.wav" in scd or "test_out.wav" in scd


# ── live_loop unrolling ──────────────────────────────────────────────────────

def test_live_loop_unrolled_events():
    scd = make_scd("live_loop :x do\n  play 60\n  sleep 1\nend", iters=4)
    lines = [l for l in scd.splitlines() if '/s_new' in l]
    assert len(lines) == 4


# ── High-level transpile ─────────────────────────────────────────────────────

def test_transpile_returns_string():
    result = transpile(
        "play 60",
        output_wav="/tmp/out.wav",
        sonic_pi_root=SONIC_PI_ROOT,
    )
    assert isinstance(result, str)
    assert len(result) > 50


def test_transpile_complex():
    src = """
use_bpm 120

with_fx :reverb, room: 0.8 do
  4.times do
    play chord(:C4, :minor), amp: 0.7
    sleep 0.5
  end
end

sample :bd_haus, amp: 1.0
"""
    result = transpile(
        src,
        output_wav="/tmp/complex.wav",
        sonic_pi_root=SONIC_PI_ROOT,
    )
    assert "sonic-pi-fx_reverb" in result
    assert "/b_allocRead" in result
    assert result.count("/s_new") >= 5   # 4 chords × 3 notes + FX + sample


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
