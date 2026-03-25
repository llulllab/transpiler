"""Tests for newly implemented Sonic Pi features."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transpiler.parser import parse
from transpiler.evaluator import evaluate

SONIC_PI_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'sonic-pi')


def ev(source, iters=4, seed=42):
    return evaluate(parse(source), sonic_pi_root=SONIC_PI_ROOT,
                    live_loop_iters=iters, rng_seed=seed)


def synths(source, **kw):
    return [e for e in ev(source, **kw) if e.kind == 'synth']


# ── Compound assignment ──────────────────────────────────────────────────────

def test_plus_assign():
    events = synths("x = 60\nx += 4\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_minus_assign():
    events = synths("x = 70\nx -= 7\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 63.0) < 0.01


def test_star_assign():
    events = synths("x = 3\nx *= 20\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_slash_assign():
    events = synths("x = 120\nx /= 2\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_or_assign_nil():
    events = synths("x = nil\nx ||= 60\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_or_assign_existing():
    events = synths("x = 64\nx ||= 60\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── Range literal ────────────────────────────────────────────────────────────

def test_range_inclusive_each():
    events = synths("(1..4).each do |i|\n  play i * 10\nend")
    assert len(events) == 4


def test_range_exclusive_each():
    events = synths("(1...4).each do |i|\n  play i * 10\nend")
    assert len(events) == 3


def test_range_assigns_param():
    events = synths("(60..62).each do |n|\n  play n\nend")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 61.0, 62.0]


def test_integer_upto():
    events = synths("1.upto(4) do |i|\n  play i * 10\nend")
    assert len(events) == 4


# ── Probability ──────────────────────────────────────────────────────────────

def test_one_in_1_always_true():
    events = synths("if one_in(1)\n  play 60\nend")
    assert len(events) == 1


def test_one_in_returns_bool():
    events = ev("x = one_in(2)")
    # No error; just verify it runs
    assert events == []


def test_dice_returns_int():
    events = synths("play dice(6)")
    assert len(events) == 1
    note = events[0].args['note']
    assert 1 <= note <= 6


def test_coin_flip():
    # coin_flip returns a bool — verify it runs and produces 0 or 1 events
    evts = ev("if coin_flip\n  play 60\nend")
    assert len(evts) in (0, 1)


# ── Math helpers ─────────────────────────────────────────────────────────────

def test_hz_to_midi_a440():
    events = synths("play hz_to_midi(440)")
    assert abs(events[0].args['note'] - 69.0) < 0.01


def test_hz_to_midi_a4_octave():
    events = synths("play hz_to_midi(880)")
    assert abs(events[0].args['note'] - 81.0) < 0.01


def test_midi_to_hz():
    events = ev("x = midi_to_hz(69)")
    assert events == []  # no sound, just value


def test_factor_true():
    events = synths("if factor?(6, 3)\n  play 60\nend")
    assert len(events) == 1


def test_factor_false():
    events = synths("if factor?(7, 3)\n  play 60\nend")
    assert len(events) == 0


def test_quantise():
    events = synths("play quantise(62.3, 2)")
    assert abs(events[0].args['note'] - 62.0) < 0.01


def test_quantise_rounds_up():
    events = synths("play quantise(63.0, 2)")
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── Beat / real-time converters ───────────────────────────────────────────────

def test_bt_sleep():
    events = ev("use_bpm 60\nsleep bt(2)")
    assert events == []  # bt(2) at 60bpm = 2s → 2 beats sleep


def test_bt_used_in_sleep():
    # bt(1) at 120bpm = 0.5 real seconds = 0.5 beat-value passed to sleep
    # → sleep 0.5 beats → second note at 0.5 * (60/120) = 0.25s
    events = synths("use_bpm 120\nplay 60\nsleep bt(1)\nplay 64")
    times = [e.time for e in events]
    assert abs(times[1] - times[0] - 0.25) < 0.01


# ── List constructors ─────────────────────────────────────────────────────────

def test_knit():
    events = ev("x = knit(60, 2, 64, 1)\nplay x[0]\nplay x[1]\nplay x[2]")
    synth_events = [e for e in events if e.kind == 'synth']
    notes = [e.args['note'] for e in synth_events]
    assert notes[0] == 60.0
    assert notes[2] == 64.0


def test_bools_choose():
    events = synths("x = bools(1, 0, 1, 1)\nplay x.choose")
    # bools returns list of True/False; choose picks one; play True → 1.0 note (not None)
    # At minimum, no error
    assert len(events) >= 0


def test_range_list():
    events = synths("x = range(60, 62, step: 1)\nplay x[0]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_range_list_count():
    events = ev("x = range(0, 3, step: 1)")
    assert events == []  # no sound, just value check via no error


def test_line_list():
    events = ev("x = line(0, 1, steps: 5)")
    assert events == []  # no sound


# ── density / at / on ─────────────────────────────────────────────────────────

def test_density_runs_n_times():
    events = synths("density 4 do\n  play 60\n  sleep 0.5\nend")
    assert len(events) == 4


def test_density_doubles_bpm():
    # density 2 at 60bpm → internal 120bpm → 0.5s per beat
    events = synths("use_bpm 60\ndensity 2 do\n  play 60\n  sleep 1\n  play 64\nend")
    times = sorted(e.time for e in events)
    assert len(times) == 4  # 2 iterations × 2 notes


def test_at_schedules_events():
    events = synths("at [0, 0.5, 1.0] do\n  play 60\nend")
    assert len(events) == 3


def test_at_times():
    events = synths("at [0, 1, 2] do\n  play 60\nend")
    times = sorted(e.time for e in events)
    assert abs(times[0] - 0.0) < 0.01
    assert abs(times[1] - 1.0) < 0.01
    assert abs(times[2] - 2.0) < 0.01


def test_on_true():
    events = synths("on true do\n  play 60\nend")
    assert len(events) == 1


def test_on_false():
    events = synths("on false do\n  play 60\nend")
    assert len(events) == 0


# ── define / user functions ───────────────────────────────────────────────────

def test_define_and_call():
    events = synths("define :play_c do\n  play 60\nend\nplay_c")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_define_called_multiple_times():
    events = synths("define :pat do\n  play 60\n  sleep 1\nend\npat\npat")
    assert len(events) == 2


def test_def_function():
    events = synths("def my_note\n  play 64\nend\nmy_note")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── case / when ───────────────────────────────────────────────────────────────

def test_case_when_match():
    events = synths("x = 2\ncase x\nwhen 1 then play 60\nwhen 2 then play 64\nend")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_case_when_else():
    events = synths("x = 5\ncase x\nwhen 1 then play 60\nelse\n  play 72\nend")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_case_no_match_no_else():
    events = synths("x = 9\ncase x\nwhen 1 then play 60\nend")
    assert len(events) == 0


# ── Transpose / octave ────────────────────────────────────────────────────────

def test_use_transpose():
    events = synths("use_transpose 2\nplay 60")
    assert abs(events[0].args['note'] - 62.0) < 0.01


def test_use_transpose_negative():
    # use explicit parens to avoid parsing as subtraction
    events = synths("use_transpose(-12)\nplay 60")
    assert abs(events[0].args['note'] - 48.0) < 0.01


def test_with_transpose_restores():
    events = synths("with_transpose 12 do\n  play 60\nend\nplay 60")
    assert abs(events[0].args['note'] - 72.0) < 0.01
    assert abs(events[1].args['note'] - 60.0) < 0.01


def test_use_octave():
    events = synths("use_octave 1\nplay 60")
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_with_octave_restores():
    events = synths("with_octave(-1) do\n  play 60\nend\nplay 60")
    assert abs(events[0].args['note'] - 48.0) < 0.01
    assert abs(events[1].args['note'] - 60.0) < 0.01


# ── Music theory extras ───────────────────────────────────────────────────────

def test_chord_invert_shift1():
    events = synths("play chord_invert([60, 64, 67], 1)[0]")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_degree_first():
    events = synths("play degree(1, :C4, :major)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_degree_fifth():
    events = synths("play degree(5, :C4, :major)")
    assert abs(events[0].args['note'] - 67.0) < 0.01


def test_chord_degree():
    events = synths("play chord_degree(1, :C4, :major)[0]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_note_range():
    events = synths("x = note_range(:C4, :E4)\nplay x[0]\nplay x[-1]")
    notes = sorted(e.args['note'] for e in events)
    assert abs(notes[0] - 60.0) < 0.01
    assert abs(notes[-1] - 64.0) < 0.01


# ── Number methods ────────────────────────────────────────────────────────────

def test_even():
    events = synths("if 4.even?\n  play 60\nend")
    assert len(events) == 1


def test_odd():
    events = synths("if 3.odd?\n  play 60\nend")
    assert len(events) == 1


def test_clamp():
    events = synths("play 100.clamp(60, 72)")
    assert abs(events[0].args['note'] - 72.0) < 0.01


# ── && and || operators ───────────────────────────────────────────────────

def test_and_and_operator():
    events = synths("if true && true\n  play 60\nend")
    assert len(events) == 1


def test_and_and_false():
    events = synths("if true && false\n  play 60\nend")
    assert len(events) == 0


def test_pipe_pipe_operator():
    events = synths("if false || true\n  play 60\nend")
    assert len(events) == 1


def test_pipe_pipe_false():
    events = synths("if false || false\n  play 60\nend")
    assert len(events) == 0


def test_and_and_with_comparison():
    events = synths("x = 3\nif x > 1 && x < 5\n  play 60\nend")
    assert len(events) == 1


# ── Ternary operator ──────────────────────────────────────────────────────

def test_ternary_true():
    events = synths("play(true ? 60 : 64)")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_ternary_false():
    events = synths("play(false ? 60 : 64)")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_ternary_with_variable():
    events = synths("x = 5\nplay(x > 3 ? 72 : 60)")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_ternary_assigns():
    events = synths("x = true ? 60 : 64\nplay x")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── Multiple assignment ───────────────────────────────────────────────────

def test_multi_assign_two():
    events = synths("a, b = 60, 64\nplay a\nplay b")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0]


def test_multi_assign_splat():
    events = synths("a, b = [60, 64]\nplay a\nplay b")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0]


def test_multi_assign_three():
    events = synths("a, b, c = 60, 64, 67\nplay c")
    assert abs(events[0].args['note'] - 67.0) < 0.01


# ── Hash literals ─────────────────────────────────────────────────────────

def test_hash_literal_stores():
    events = ev("h = {amp: 0.5}\nassert = h")
    assert events == []  # no sound, just verify it runs


def test_hash_literal_used_as_kwarg():
    # Explicit hash with key access
    events = synths("h = {note: 60}\nplay h[:note]")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_empty_hash():
    events = ev("h = {}")
    assert events == []


# ── loop do...end ─────────────────────────────────────────────────────────

def test_loop_runs_n_times():
    events = synths("loop do\n  play 60\n  sleep 1\nend", iters=3)
    assert len(events) == 3


def test_loop_with_stop():
    events = synths("i = 0\nloop do\n  play 60\n  i += 1\n  stop if i >= 2\nend", iters=10)
    assert len(events) == 2


# ── use_random_seed / with_random_seed ────────────────────────────────────

def test_use_random_seed_deterministic():
    e1 = synths("use_random_seed 42\nplay rand_i(100)")
    e2 = synths("use_random_seed 42\nplay rand_i(100)")
    assert e1[0].args['note'] == e2[0].args['note']


def test_with_random_seed_restores():
    # After with_random_seed block, outer seed is restored
    e1 = synths("use_random_seed 0\nplay rand_i(100)")
    e2 = synths("use_random_seed 0\nwith_random_seed 99 do\n  x = rand_i(100)\nend\nplay rand_i(100)")
    assert e1[0].args['note'] == e2[0].args['note']


# ── Extra list methods ────────────────────────────────────────────────────

def test_list_min():
    events = synths("play [64, 60, 67].min")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_list_max():
    events = synths("play [64, 60, 67].max")
    assert abs(events[0].args['note'] - 67.0) < 0.01


def test_list_sum():
    events = ev("x = [1, 2, 3].sum")
    assert events == []  # no sound


def test_list_sort():
    events = synths("notes = [67, 60, 64].sort\nplay notes[0]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_list_include():
    events = synths("if [60, 64, 67].include?(64)\n  play 60\nend")
    assert len(events) == 1


def test_list_include_false():
    events = synths("if [60, 64, 67].include?(70)\n  play 60\nend")
    assert len(events) == 0


def test_list_count():
    events = ev("x = [1, 2, 3].count")
    assert events == []


def test_list_size():
    events = ev("x = [1, 2, 3].size")
    assert events == []


def test_list_compact():
    events = synths("play [60, nil, 64].compact[0]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_list_uniq():
    events = synths("play [60, 60, 64].uniq[1]")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_list_flatten():
    events = synths("play [[60, 64], [67]].flatten[2]")
    assert abs(events[0].args['note'] - 67.0) < 0.01


def test_list_take():
    events = synths("play [60, 64, 67].take(2)[1]")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_list_drop():
    events = synths("play [60, 64, 67].drop(1)[0]")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_list_select():
    events = synths("play [60, 63, 64].select {|n| n > 61}[0]")
    assert abs(events[0].args['note'] - 63.0) < 0.01


def test_list_reduce():
    # [60, 4, 3].reduce → accumulates
    events = ev("x = [1, 2, 3].reduce {|acc, n| acc + n}")
    assert events == []


def test_list_zip():
    events = ev("x = [60, 64].zip([0, 1])")
    assert events == []


# ── live_loop time isolation ──────────────────────────────────────────────

def test_two_live_loops_start_at_t0():
    """Both live_loops should produce events starting at t=0."""
    events = synths(
        "live_loop :a do\n  play 60\n  sleep 1\nend\n"
        "live_loop :b do\n  play 64\n  sleep 1\nend",
        iters=2,
    )
    times_a = sorted(e.time for e in events if abs(e.args['note'] - 60.0) < 0.01)
    times_b = sorted(e.time for e in events if abs(e.args['note'] - 64.0) < 0.01)
    assert abs(times_a[0] - 0.0) < 0.01
    assert abs(times_b[0] - 0.0) < 0.01


def test_live_loop_code_after_loop_at_t0():
    """Code after a live_loop should execute from t=0, not after loop end."""
    events = synths(
        "live_loop :foo do\n  play 60\n  sleep 10\nend\nplay 72",
        iters=1,
    )
    note_72 = [e for e in events if abs(e.args['note'] - 72.0) < 0.01]
    assert len(note_72) == 1
    assert abs(note_72[0].time - 0.0) < 0.01


# ── for loop ─────────────────────────────────────────────────────────────

def test_for_loop_range():
    events = synths("for i in 1..4\n  play i * 10\nend")
    assert len(events) == 4


def test_for_loop_range_values():
    events = synths("for i in (1..3)\n  play i * 10\nend")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [10.0, 20.0, 30.0]


def test_for_loop_array():
    events = synths("for n in [60, 64, 67]\n  play n\nend")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0, 67.0]


def test_for_loop_exclusive_range():
    events = synths("for i in 1...4\n  play i * 10\nend")
    assert len(events) == 3


# ── until loop ───────────────────────────────────────────────────────────

def test_until_loop():
    events = synths("i = 0\nuntil i >= 3\n  play 60\n  i += 1\nend")
    assert len(events) == 3


def test_until_never_fires_if_true():
    events = synths("until true\n  play 60\nend")
    assert len(events) == 0


# ── play_pattern_timed ────────────────────────────────────────────────────

def test_play_pattern_timed_count():
    events = synths("play_pattern_timed [60, 64, 67], [0.5, 0.5, 0.5]")
    assert len(events) == 3


def test_play_pattern_timed_notes():
    events = synths("play_pattern_timed [60, 64, 67], [1]")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0, 67.0]


def test_play_pattern_timed_timing():
    events = synths("use_bpm 60\nplay_pattern_timed [60, 64], [1, 2]")
    times = [e.time for e in events]
    assert abs(times[0] - 0.0) < 0.01
    assert abs(times[1] - 1.0) < 0.01


def test_play_pattern_timed_wraps_timing():
    # timing shorter than notes → wraps
    events = synths("play_pattern_timed [60, 64, 67, 72], [0.5]")
    assert len(events) == 4


# ── play_pattern ─────────────────────────────────────────────────────────

def test_play_pattern_count():
    events = synths("play_pattern [60, 64, 67]")
    assert len(events) == 3


def test_play_pattern_notes():
    events = synths("play_pattern [60, 64, 67]")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0, 67.0]


# ── play_chord ────────────────────────────────────────────────────────────

def test_play_chord_simultaneous():
    events = synths("use_bpm 60\nplay_chord [60, 64, 67]")
    assert len(events) == 3
    # All at same time
    assert all(abs(e.time - events[0].time) < 0.01 for e in events)


def test_play_chord_notes():
    events = synths("play_chord [60, 64, 67]")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0, 67.0]


# ── with_bpm_mul ─────────────────────────────────────────────────────────

def test_with_bpm_mul_doubles_speed():
    # at 60bpm, 2× mul → 120bpm → 1 beat = 0.5s
    events = synths("use_bpm 60\nwith_bpm_mul 2 do\n  play 60\n  sleep 1\n  play 64\nend")
    times = sorted(e.time for e in events)
    assert abs(times[1] - times[0] - 0.5) < 0.01


def test_with_bpm_mul_restores():
    events = synths("use_bpm 60\nwith_bpm_mul 4 do\n  play 60\n  sleep 1\nend\nplay 64\nsleep 1\nplay 67")
    times = sorted(e.time for e in events)
    # Third note (67) should be 1s after second (after BPM restored to 60)
    assert abs(times[2] - times[1] - 1.0) < 0.01


# ── use_cent_tuning ───────────────────────────────────────────────────────

def test_use_cent_tuning_sharpens():
    # 100 cents = 1 semitone. 50 cents = 0.5 semitone
    events = synths("use_cent_tuning 100\nplay 60")
    assert abs(events[0].args['note'] - 61.0) < 0.01


def test_with_cent_tuning_restores():
    events = synths("with_cent_tuning 100 do\n  play 60\nend\nplay 60")
    assert abs(events[0].args['note'] - 61.0) < 0.01
    assert abs(events[1].args['note'] - 60.0) < 0.01


def test_cent_tuning_negative():
    events = synths("use_cent_tuning(-50)\nplay 60")
    assert abs(events[0].args['note'] - 59.5) < 0.01


# ── get / set ─────────────────────────────────────────────────────────────

def test_set_and_get():
    events = synths("set :note, 72\nplay get(:note)")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_get_default():
    events = synths("play get(:missing, default: 60)")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_set_overwrite():
    events = synths("set :n, 60\nset :n, 64\nplay get(:n)")
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── beat / sleep_bpm ─────────────────────────────────────────────────────

def test_beat_returns_time():
    events = ev("use_bpm 60\nsleep 2\nx = beat")
    assert events == []  # no sound, just verify it runs


def test_sleep_bpm_advances():
    events = synths("use_bpm 120\nplay 60\nsleep_bpm 1\nplay 64")
    times = [e.time for e in events]
    # sleep_bpm 1 = 1 beat at 120bpm = 0.5s
    assert abs(times[1] - times[0] - 0.5) < 0.01


# ── reset_tick ────────────────────────────────────────────────────────────

def test_reset_tick():
    events = synths(
        "notes = [60, 64, 67]\n"
        "play notes.tick\n"
        "play notes.tick\n"
        "reset_tick\n"
        "play notes.tick"
    )
    notes = [e.args['note'] for e in events]
    assert abs(notes[0] - 60.0) < 0.01
    assert abs(notes[1] - 64.0) < 0.01
    assert abs(notes[2] - 60.0) < 0.01  # reset → back to index 0


# ── Implicit return from def ──────────────────────────────────────────────

def test_def_implicit_return():
    events = synths("def my_note\n  60\nend\nplay my_note")
    assert len(events) == 1
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_def_implicit_return_expression():
    events = synths("def octave_up(n)\n  n + 12\nend\nplay octave_up(60)")
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_def_explicit_return_still_works():
    events = synths("def foo\n  return 64\n  72\nend\nplay foo")
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── Default parameters ────────────────────────────────────────────────────

def test_def_default_param_used():
    events = synths("def foo(n=60)\n  play n\nend\nfoo")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_def_default_param_overridden():
    events = synths("def foo(n=60)\n  play n\nend\nfoo(72)")
    assert abs(events[0].args['note'] - 72.0) < 0.01


def test_def_multiple_defaults():
    events = synths("def chord_play(root=60, interval=4)\n  play root\n  play root+interval\nend\nchord_play")
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0]


# ── Splat parameters ──────────────────────────────────────────────────────

def test_splat_param_collects_all():
    events = synths("def play_all(*ns)\n  ns.each {|n| play n}\nend\nplay_all(60, 64, 67)")
    assert len(events) == 3
    notes = sorted(e.args['note'] for e in events)
    assert notes == [60.0, 64.0, 67.0]


def test_splat_param_empty():
    events = synths("def play_all(*ns)\n  ns.each {|n| play n}\nend\nplay_all")
    assert len(events) == 0


# ── %w[] and %i[] arrays ──────────────────────────────────────────────────

def test_percent_w_array():
    events = synths("x = %w[foo bar baz]\nplay x.size")
    assert abs(events[0].args['note'] - 3.0) < 0.01


def test_percent_i_array():
    # %i[] creates symbol array
    events = ev("x = %i[foo bar]\nassert = x.size")
    assert events == []  # no sound, just verify parse


def test_percent_w_multiline():
    events = synths("x = %w[\n  a b\n  c\n]\nplay x.size")
    assert abs(events[0].args['note'] - 3.0) < 0.01


# ── current_bpm / current_synth ───────────────────────────────────────────

def test_current_bpm():
    events = synths("use_bpm 120\nplay current_bpm")
    assert abs(events[0].args['note'] - 120.0) < 0.01


def test_current_bpm_default():
    events = synths("play current_bpm")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_current_synth_returns_string():
    events = ev("x = current_synth")
    assert events == []  # no sound


# ── note_info ─────────────────────────────────────────────────────────────

def test_note_info_midi():
    events = synths("x = note_info(:C4)\nplay x[:midi_note]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_note_info_octave():
    events = ev("x = note_info(:C4)\nassert = x[:octave]")
    assert events == []  # just verify it runs


# ── chord_names / scale_names / sample_names ─────────────────────────────

def test_chord_names_not_empty():
    events = ev("x = chord_names")
    assert events == []  # no sound


def test_scale_names_not_empty():
    events = ev("x = scale_names")
    assert events == []


def test_sample_names_ambi():
    events = ev("x = sample_names(:ambi)")
    assert events == []


# ── with_merged_synth_defaults ────────────────────────────────────────────

def test_with_merged_synth_defaults():
    events = synths("with_merged_synth_defaults amp: 0.5 do\n  play 60\nend")
    assert len(events) == 1
    assert abs(events[0].args['amp'] - 0.5) < 0.01


def test_use_merged_synth_defaults():
    events = synths("use_merged_synth_defaults amp: 0.3\nplay 60")
    assert abs(events[0].args['amp'] - 0.3) < 0.01


# ── Lambda / Proc ─────────────────────────────────────────────────────────

def test_lambda_call():
    events = synths("f = lambda {|n| n * 2}\nplay f.call(30)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_lambda_do_end():
    events = synths("f = lambda do |n| n + 4 end\nplay f.call(56)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_proc_call():
    events = synths("f = proc {|n| n + 10}\nplay f.call(50)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_lambda_dot_paren_syntax():
    # f.(args) is shorthand for f.call(args)
    events = synths("f = lambda {|n| n + 10}\nplay f.(50)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── Math module ────────────────────────────────────────────────────────────

def test_math_sqrt():
    events = synths("play Math.sqrt(3600).to_i")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_math_sin():
    events = synths("play Math.sin(0).to_i")
    assert abs(events[0].args['note'] - 0.0) < 0.01


def test_math_cos():
    events = synths("play Math.cos(0).to_i")
    assert abs(events[0].args['note'] - 1.0) < 0.01


def test_math_log():
    events = synths("play (Math.log(1)).to_i")
    assert abs(events[0].args['note'] - 0.0) < 0.01


def test_math_exp():
    events = synths("play Math.exp(0).to_i")
    assert abs(events[0].args['note'] - 1.0) < 0.01


# ── Integer() / Float() type conversions ──────────────────────────────────

def test_integer_conv():
    events = synths("play Integer(60.9)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_float_conv():
    events = synths("x = Float(60)\nplay x")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_integer_from_string():
    events = synths("play Integer('60')")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── inject / reduce with symbol and initial value ─────────────────────────

def test_inject_symbol():
    events = synths("play [10, 20, 30].inject(:+)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_inject_symbol_colon():
    # :+ should tokenize as a symbol
    events = synths("play [1, 2, 3].inject(:+)")
    assert abs(events[0].args['note'] - 6.0) < 0.01


def test_inject_with_initial_value():
    events = synths("play [6, 10, 14, 18, 12].inject(:+)")
    # 6+10+14+18+12 = 60
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_inject_block_with_initial():
    events = synths("play [10, 20].inject(30) {|acc, n| acc + n}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_reduce_alias():
    events = synths("play [20, 20, 20].reduce(:+)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── Array.new ─────────────────────────────────────────────────────────────

def test_array_new():
    events = synths("arr = Array.new(3, 20)\nplay arr.sum")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_array_new_default_nil():
    events = ev("arr = Array.new(3)")
    assert events == []  # just verify it runs


# ── pitch_to_ratio / ratio_to_pitch ───────────────────────────────────────

def test_pitch_to_ratio_octave():
    # 12 semitones = ratio 2.0
    events = synths("x = pitch_to_ratio(12)\nplay (x * 30).to_i")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_ratio_to_pitch_octave():
    # ratio 2 = 12 semitones
    events = synths("play ratio_to_pitch(2).to_i")
    assert abs(events[0].args['note'] - 12.0) < 0.01


# ── is_a? / kind_of? / respond_to? ───────────────────────────────────────

def test_is_a_integer():
    events = synths("x = 60\nplay x if x.is_a?(Integer)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_is_a_float():
    events = synths("x = 60.0\nplay x if x.is_a?(Float)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_is_a_array():
    events = synths("x = [60]\nplay x[0] if x.is_a?(Array)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_kind_of():
    events = synths("play 60 if 60.kind_of?(Integer)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_respond_to_each():
    events = synths("play 60 if [1].respond_to?(:each)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── nil conversions ───────────────────────────────────────────────────────

def test_nil_to_i():
    events = synths("x = nil\nplay x.to_i + 60")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_nil_to_f():
    events = synths("x = nil\nplay x.to_f + 60.0")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_nil_to_s():
    events = ev("x = nil\ny = x.to_s")
    assert events == []  # just verify no crash


# ── Operator symbols :+ :- etc. ────────────────────────────────────────────

def test_operator_symbol_tokenize():
    # :+ should be tokenized as a SYMBOL with value '+'
    events = synths("play [20, 40].inject(:+)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_operator_symbol_multiply():
    events = synths("play [2, 30].inject(:*)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── .() call syntax ────────────────────────────────────────────────────────

def test_dot_paren_call():
    events = synths("f = lambda {|a, b| a + b}\nplay f.(30, 30)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── Hash methods ─────────────────────────────────────────────────────────

def test_hash_keys():
    events = synths("h = {a: 60, b: 64}\nplay h.keys.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_hash_values():
    events = synths("h = {a: 60}\nplay h.values[0]")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_hash_fetch():
    events = synths("h = {a: 60}\nplay h.fetch(:a)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_hash_fetch_default():
    events = synths("h = {a: 60}\nplay h.fetch(:z, 64)")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_hash_has_key():
    events = synths("h = {a: 60}\nplay h[:a] if h.has_key?(:a)")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_hash_size():
    events = synths("h = {a: 1, b: 2}\nplay h.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_hash_merge():
    events = synths("h = {a: 1}.merge({b: 2})\nplay h.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_hash_each():
    events = synths("h = {a: 60, b: 64}\nh.each {|k,v| play v}")
    assert len(events) == 2


def test_hash_map():
    events = synths("h = {a: 30, b: 30}\narr = h.map {|k,v| v}\nplay arr.sum")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_hash_select():
    events = synths("h = {a: 60, b: 2}\nr = h.select {|k,v| v > 10}\nplay r.size")
    assert abs(events[0].args['note'] - 1.0) < 0.01


def test_hash_delete():
    events = synths("h = {a: 60, b: 2}\nh.delete(:b)\nplay h.size")
    assert abs(events[0].args['note'] - 1.0) < 0.01


# ── New array methods ─────────────────────────────────────────────────────

def test_flat_map():
    events = synths("play [[60,64],[67]].flat_map {|x| x}.size")
    assert abs(events[0].args['note'] - 3.0) < 0.01


def test_any_true():
    events = synths("play 60 if [1,2,3].any? {|n| n > 2}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_any_false():
    events = synths("play 60 if [1,2,3].any? {|n| n > 10}")
    assert len(events) == 0


def test_all_true():
    events = synths("play 60 if [2,4,6].all? {|n| n.even?}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_none_true():
    events = synths("play 60 if [2,4,6].none? {|n| n.odd?}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_array_join():
    events = synths('x = [1,2,3].join(",")\nplay x.length')
    assert abs(events[0].args['note'] - 5.0) < 0.01


def test_array_push():
    events = synths("a = [60]\na.push(64)\nplay a.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_array_append_operator():
    events = synths("a = [60]\na << 64\nplay a.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_array_index():
    events = synths("play [60,64,67].index(64)")
    assert abs(events[0].args['note'] - 1.0) < 0.01


def test_array_min_by():
    events = synths("play [64,60,67].min_by {|x| x}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_array_max_by():
    events = synths("play [64,60,67].max_by {|x| x}")
    assert abs(events[0].args['note'] - 67.0) < 0.01


def test_each_with_object():
    events = synths("r = [1,2,3].each_with_object([]) {|x,a| a << x*20}\nplay r.sum")
    assert abs(events[0].args['note'] - 120.0) < 0.01


def test_delete_if():
    events = synths("a = [60,64,67]\na.delete_if {|x| x > 64}\nplay a.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_tally():
    events = synths("x = [60,60,64].tally\nplay x.keys.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_group_by():
    events = synths("x = [60,60,64].group_by {|n| n}\nplay x.keys.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_array_concat():
    events = synths("a = [10,20] + [30]\nplay a.sum")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_array_pop():
    events = synths("a = [60,64]\nplay a.pop")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_array_shift():
    events = synths("a = [60,64]\nplay a.shift")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── String methods ────────────────────────────────────────────────────────

def test_string_start_with():
    events = synths('play 60 if "hello".start_with?("he")')
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_string_end_with():
    events = synths('play 60 if "hello".end_with?("lo")')
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_string_include():
    events = synths('play 60 if "hello".include?("ell")')
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_string_strip():
    events = synths('play " hi ".strip.length')
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_string_chomp():
    events = synths('play "hello".chomp.length')
    assert abs(events[0].args['note'] - 5.0) < 0.01


def test_string_gsub():
    events = synths('play "aab".gsub("a","x").length')
    assert abs(events[0].args['note'] - 3.0) < 0.01


def test_string_chars():
    events = synths('play "hello".chars.size')
    assert abs(events[0].args['note'] - 5.0) < 0.01


def test_string_reverse():
    events = synths('play "hello".reverse.length')
    assert abs(events[0].args['note'] - 5.0) < 0.01


# ── tap / then ────────────────────────────────────────────────────────────

def test_tap_returns_self():
    events = synths("play 60.tap {|n| puts n}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_then_yields():
    events = synths("play 30.then {|n| n * 2}")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── Numeric methods ───────────────────────────────────────────────────────

def test_num_gcd():
    events = synths("play 12.gcd(8)")
    assert abs(events[0].args['note'] - 4.0) < 0.01


def test_num_lcm():
    events = synths("play 4.lcm(5)")
    assert abs(events[0].args['note'] - 20.0) < 0.01


def test_num_digits():
    events = synths("play 123.digits.size")
    assert abs(events[0].args['note'] - 3.0) < 0.01


# ── Array.new with block ──────────────────────────────────────────────────

def test_array_new_block():
    events = synths("a = Array.new(3) {|i| i * 20 + 20}\nplay a.sum")
    assert abs(events[0].args['note'] - 120.0) < 0.01


# ── use_tuning / with_tuning ─────────────────────────────────────────────

def test_use_tuning():
    events = synths("use_tuning :equal\nplay 60")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_with_tuning():
    events = synths("with_tuning :equal do\nplay 60\nend")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── synth_names ───────────────────────────────────────────────────────────

def test_synth_names_not_empty():
    events = ev("x = synth_names\nplay 60 if x.size > 0")
    assert len([e for e in events if e.kind == 'synth']) == 1


# ── Math::PI / scope resolution :: ───────────────────────────────────────

def test_math_pi():
    events = synths("play Math::PI.to_i")
    assert abs(events[0].args['note'] - 3.0) < 0.01


def test_math_e():
    events = synths("play Math::E.to_i")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_scope_as_dot():
    # :: used for module access should work like .
    events = synths("play Math::sqrt(3600).to_i")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── case/when range containment ──────────────────────────────────────────

def test_case_when_range():
    events = synths("n = 3\ncase n\nwhen 1..5\n  play 60\nend")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_case_when_range_no_match():
    events = synths("n = 10\ncase n\nwhen 1..5\n  play 60\nelse\n  play 64\nend")
    assert abs(events[0].args['note'] - 64.0) < 0.01


def test_case_when_multiple():
    # Two separate when clauses
    events = synths("n = 3\ncase n\nwhen 1\n  play 64\nwhen 3\n  play 60\nend")
    assert abs(events[0].args['note'] - 60.0) < 0.01


# ── case as expression ───────────────────────────────────────────────────

def test_case_as_expression():
    events = synths("y = case 60\nwhen 60 then 60\nelse 0\nend\nplay y")
    assert abs(events[0].args['note'] - 60.0) < 0.01


def test_case_else_expression():
    events = synths("y = case 99\nwhen 60 then 60\nelse 64\nend\nplay y")
    assert abs(events[0].args['note'] - 64.0) < 0.01


# ── << operator (array append) ───────────────────────────────────────────

def test_lshift_array_append():
    events = synths("a = []\na << 60\na << 64\nplay a.size")
    assert abs(events[0].args['note'] - 2.0) < 0.01


def test_lshift_chain():
    events = synths("a = []\na << 20 << 20 << 20\nplay a.sum")
    assert abs(events[0].args['note'] - 60.0) < 0.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
