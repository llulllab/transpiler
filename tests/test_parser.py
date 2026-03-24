"""Tests for the parser."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transpiler.parser import parse
from transpiler.ast_nodes import (
    Program, IntLit, FloatLit, SymbolLit, StringLit, BoolLit, NilLit, ArrayLit,
    Identifier, Assign, BinOp, UnaryOp, MethodCall, Block,
    IfStmt,
)


def test_int_literal():
    p = parse("60")
    assert isinstance(p.statements[0], IntLit)
    assert p.statements[0].value == 60


def test_float_literal():
    p = parse("0.5")
    assert isinstance(p.statements[0], FloatLit)
    assert p.statements[0].value == 0.5


def test_symbol_literal():
    p = parse(":beep")
    assert isinstance(p.statements[0], SymbolLit)
    assert p.statements[0].value == "beep"


def test_string_literal():
    p = parse('"hello"')
    assert isinstance(p.statements[0], StringLit)
    assert p.statements[0].value == "hello"


def test_array_literal():
    p = parse("[1, 2, 3]")
    a = p.statements[0]
    assert isinstance(a, ArrayLit)
    assert len(a.elements) == 3
    assert a.elements[0].value == 1


def test_assignment():
    p = parse("x = 42")
    a = p.statements[0]
    assert isinstance(a, Assign)
    assert a.name == "x"
    assert isinstance(a.value, IntLit)
    assert a.value.value == 42


def test_binop():
    p = parse("2 + 3")
    b = p.statements[0]
    assert isinstance(b, BinOp)
    assert b.op == "+"


def test_play_simple():
    p = parse("play 60")
    mc = p.statements[0]
    assert isinstance(mc, MethodCall)
    assert mc.method == "play"
    assert mc.args[0].value == 60


def test_play_with_kwargs():
    p = parse("play 60, amp: 0.5, release: 2")
    mc = p.statements[0]
    assert mc.method == "play"
    assert mc.args[0].value == 60
    assert "amp" in mc.kwargs
    assert "release" in mc.kwargs


def test_play_symbol_note():
    p = parse("play :C4")
    mc = p.statements[0]
    assert mc.args[0].value == "C4"


def test_synth_call():
    p = parse("synth :beep, note: 60, amp: 0.8")
    mc = p.statements[0]
    assert mc.method == "synth"
    assert mc.args[0].value == "beep"
    assert "note" in mc.kwargs


def test_sample_call():
    p = parse("sample :bd_haus")
    mc = p.statements[0]
    assert mc.method == "sample"
    assert mc.args[0].value == "bd_haus"


def test_sleep():
    p = parse("sleep 1")
    mc = p.statements[0]
    assert mc.method == "sleep"
    assert mc.args[0].value == 1


def test_use_synth():
    p = parse("use_synth :saw")
    mc = p.statements[0]
    assert mc.method == "use_synth"
    assert mc.args[0].value == "saw"


def test_use_bpm():
    p = parse("use_bpm 120")
    mc = p.statements[0]
    assert mc.method == "use_bpm"
    assert mc.args[0].value == 120


def test_with_fx_block():
    src = """
with_fx :reverb, room: 0.8 do
  play 60
end
"""
    p = parse(src)
    mc = p.statements[0]
    assert mc.method == "with_fx"
    assert mc.args[0].value == "reverb"
    assert "room" in mc.kwargs
    assert mc.block is not None
    assert len(mc.block.body) == 1


def test_live_loop():
    src = """
live_loop :myloop do
  play 60
  sleep 1
end
"""
    p = parse(src)
    mc = p.statements[0]
    assert mc.method == "live_loop"
    assert mc.args[0].value == "myloop"
    assert mc.block is not None
    assert len(mc.block.body) == 2


def test_in_thread():
    src = """
in_thread do
  play 60
end
"""
    p = parse(src)
    mc = p.statements[0]
    assert mc.method == "in_thread"
    assert mc.block is not None


def test_times_loop():
    src = """
4.times do
  play 60
  sleep 0.5
end
"""
    p = parse(src)
    mc = p.statements[0]
    assert isinstance(mc, MethodCall)
    assert mc.method == "times"
    assert mc.block is not None


def test_chord_call():
    p = parse("chord :E3, :minor")
    mc = p.statements[0]
    assert mc.method == "chord"


def test_scale_call():
    p = parse("scale :C4, :major, num_octaves: 2")
    mc = p.statements[0]
    assert mc.method == "scale"
    assert "num_octaves" in mc.kwargs


def test_multiline():
    src = """
use_bpm 120
play 60
sleep 1
play 64
"""
    p = parse(src)
    assert len(p.statements) == 4


def test_nested_with_fx():
    src = """
with_fx :reverb do
  with_fx :echo, phase: 0.25 do
    play 60
  end
end
"""
    p = parse(src)
    outer = p.statements[0]
    assert outer.method == "with_fx"
    inner = outer.block.body[0]
    assert inner.method == "with_fx"
    assert inner.args[0].value == "echo"


def test_if_statement():
    src = """
if true
  play 60
end
"""
    p = parse(src)
    stmt = p.statements[0]
    assert isinstance(stmt, IfStmt)


def test_comment_ignored():
    src = """
# This is a comment
play 60  # inline comment
"""
    p = parse(src)
    assert len(p.statements) == 1
    assert p.statements[0].method == "play"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
