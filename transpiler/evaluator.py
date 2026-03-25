"""
Evaluator – walks the AST and produces a flat, time-ordered list of
SoundEvent objects.  Simulates Sonic Pi's thread-based runtime:

  * time is tracked in beats; converted to seconds on emit
  * use_bpm changes the conversion ratio going forward
  * live_loop is unrolled N times (configurable)
  * in_thread forks the clock; parent time is unchanged after the thread
  * with_fx allocates an audio bus and wraps its block's events
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from .ast_nodes import (
    Node, Program, Block,
    IntLit, FloatLit, StringLit, SymbolLit, BoolLit, NilLit, ArrayLit, RangeLit,
    Identifier, Assign, BinOp, UnaryOp, MethodCall,
    IfStmt, WhileStmt, ReturnStmt, CaseStmt, FuncDef,
    MultiAssign, TernaryExpr, HashLit,
)
from .music_theory import (
    note_to_midi, chord as mk_chord, scale as mk_scale,
    chord_invert as mk_chord_invert, chord_degree as mk_chord_degree,
    degree as mk_degree, note_range as mk_note_range,
    CHORD_INTERVALS, SCALE_INTERVALS,
)
from .sample_map import SampleResolver


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SoundEvent:
    """
    A single audio event scheduled at an absolute time.

    kind:
      'synth'    – trigger a named synth  (SC: /s_new)
      'sample'   – play a sample file     (SC: /s_new basic_*_player)
      'fx_open'  – open an FX synth       (SC: /s_new at fx_start)
      'fx_close' – free the FX synth node (SC: /n_free at fx_end)
    """
    time: float          # absolute time in seconds
    kind: str
    synth_name: str      # e.g. 'sonic-pi-beep'  or path for samples
    node_id: int
    args: dict[str, Any]
    bus_out: int = 0     # output bus (0 = main stereo out)
    bus_in: int = 0      # input bus  (only relevant for FX)

    def duration(self) -> float:
        """Estimated synth duration in seconds (ADSR sum)."""
        a = self.args.get('attack',  0.0) or 0.0
        d = self.args.get('decay',   0.0) or 0.0
        s = self.args.get('sustain', 0.0) or 0.0
        r = self.args.get('release', 1.0) or 1.0
        return a + d + s + r


@dataclass
class _Lambda:
    """Callable object created by lambda or proc."""
    params: list
    body: list   # AST nodes
    closure: dict


@dataclass
class _FXFrame:
    """Tracks an active with_fx context."""
    fx_name: str         # e.g. 'reverb'
    opts: dict[str, Any]
    bus_in: int          # bus that wrapped synths write to
    bus_out: int         # bus that this FX writes to
    node_id: int
    start_time: float    # seconds


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class EvalError(Exception):
    pass


class StopIteration_(Exception):
    """Internal: raised by 'stop' inside a live_loop."""
    pass

class _ReturnSignal(Exception):
    """Internal: raised by 'return' inside a user function."""
    def __init__(self, value): self.value = value


class Evaluator:
    """
    Evaluate a Sonic Pi AST and collect SoundEvent objects.

    Args:
        sonic_pi_root:    Path to the sonic-pi repo (for sample resolution).
        live_loop_iters:  How many times to unroll each live_loop.
        rng_seed:         Seed for random calls (None = random).
    """

    def __init__(
        self,
        sonic_pi_root: str = ".",
        live_loop_iters: int = 8,
        rng_seed: Optional[int] = 42,
    ):
        self._sample_resolver = SampleResolver(sonic_pi_root)
        self._live_loop_iters = live_loop_iters
        self._rng = random.Random(rng_seed)

        # ── per-thread state (we simulate sequentially) ──────────────────
        self._time: float = 0.0          # current time in BEATS
        self._bpm: float = 60.0
        self._current_synth: str = 'beep'
        self._synth_defaults: dict[str, Any] = {}
        self._sample_defaults: dict[str, Any] = {}
        self._variables: dict[str, Any] = {}
        self._fx_stack: list[_FXFrame] = []
        self._transpose: float = 0.0     # semitone offset (use_transpose)
        self._octave: float = 0.0        # octave shift    (use_octave)
        self._cent_tuning: float = 0.0   # cent offset     (use_cent_tuning)
        self._user_funcs: dict[str, FuncDef] = {}  # define/def storage
        self._store: dict[str, Any] = {}  # get/set shared key-value store

        # ── global counters ───────────────────────────────────────────────
        self._node_id: int = 1000
        self._bus_id: int = 16           # buses 0-15 reserved; 16+ for FX

        # ── output ───────────────────────────────────────────────────────
        self.events: list[SoundEvent] = []

    # ── Internal helpers ────────────────────────────────────────────────

    def _alloc_node(self) -> int:
        n = self._node_id
        self._node_id += 1
        return n

    def _alloc_bus(self) -> int:
        b = self._bus_id
        self._bus_id += 1
        return b

    @property
    def _time_secs(self) -> float:
        return self._time * (60.0 / self._bpm)

    def _current_bus_out(self) -> int:
        """The bus that the current synth should write to."""
        if self._fx_stack:
            return self._fx_stack[-1].bus_in
        return 0

    def _emit_synth(self, synth_name: str, args: dict) -> SoundEvent:
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='synth',
            synth_name=f'sonic-pi-{synth_name}',
            node_id=nid,
            args=dict(args),
            bus_out=self._current_bus_out(),
        )
        self.events.append(evt)
        return evt

    def _emit_sample(self, path: str, args: dict) -> SoundEvent:
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='sample',
            synth_name=path,
            node_id=nid,
            args=dict(args),
            bus_out=self._current_bus_out(),
        )
        self.events.append(evt)
        return evt

    def _clone_state(self) -> dict:
        """Snapshot mutable thread-local state for forking."""
        return {
            'time': self._time,
            'bpm': self._bpm,
            'current_synth': self._current_synth,
            'synth_defaults': dict(self._synth_defaults),
            'sample_defaults': dict(self._sample_defaults),
            'variables': dict(self._variables),
            'fx_stack': list(self._fx_stack),
            'transpose': self._transpose,
            'octave': self._octave,
            'cent_tuning': self._cent_tuning,
        }

    def _restore_state(self, snap: dict):
        self._time = snap['time']
        self._bpm = snap['bpm']
        self._current_synth = snap['current_synth']
        self._synth_defaults = snap['synth_defaults']
        self._sample_defaults = snap['sample_defaults']
        self._variables = snap['variables']
        self._fx_stack = snap['fx_stack']
        self._transpose = snap.get('transpose', 0.0)
        self._octave = snap.get('octave', 0.0)
        self._cent_tuning = snap.get('cent_tuning', 0.0)

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate(self, program: Program) -> list[SoundEvent]:
        self.events = []
        self._eval_body(program.statements)
        self.events.sort(key=lambda e: e.time)
        return self.events

    # ── Statement dispatch ───────────────────────────────────────────────

    def _eval_body(self, stmts: list[Node]):
        for stmt in stmts:
            self._eval_node(stmt)

    def _eval_node(self, node: Node) -> Any:
        name = type(node).__name__
        handler = getattr(self, f'_eval_{name}', None)
        if handler:
            return handler(node)
        return None

    # ── Literals ─────────────────────────────────────────────────────────

    def _eval_IntLit(self, n: IntLit): return n.value
    def _eval_FloatLit(self, n: FloatLit): return n.value
    def _eval_StringLit(self, n: StringLit): return n.value
    def _eval_SymbolLit(self, n: SymbolLit): return n.value   # string without ':'
    def _eval_BoolLit(self, n: BoolLit): return n.value
    def _eval_NilLit(self, n: NilLit): return None

    def _eval_ArrayLit(self, n: ArrayLit) -> list:
        return [self._eval_node(e) for e in n.elements]

    def _eval_RangeLit(self, n: RangeLit) -> list:
        start = self._eval_node(n.start)
        end = self._eval_node(n.end)
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return []
        step = 1 if end >= start else -1
        end_i = int(end) + (0 if n.exclusive else step)
        return list(range(int(start), end_i, step))

    # ── Variables ────────────────────────────────────────────────────────

    # Built-in no-arg calls that should execute even when parsed as Identifier
    _BARE_BUILTINS = {'stop', 'coin_flip', 'reset_tick', 'tick', 'look',
                      'beat', 'current_beat', 'current_bpm', 'current_synth',
                      'current_synth_name', 'current_time', 'current_time_in_beats',
                      'chord_names', 'scale_names', 'free_all'}

    def _eval_body_last(self, stmts: list) -> Any:
        """Evaluate body statements, returning the last expression's value."""
        result = None
        for stmt in stmts:
            result = self._eval_node(stmt)
        return result

    def _eval_Identifier(self, n: Identifier) -> Any:
        # Special module/class sentinels
        if n.name == 'Math':
            return '__Math__'
        if n.name in ('Array', 'Integer', 'Float', 'String', 'Hash',
                       'Numeric', 'NilClass', 'TrueClass', 'FalseClass',
                       'Fixnum', 'Bignum'):
            return n.name  # return class name as string for is_a? etc.
        val = self._variables.get(n.name)
        if val is None and n.name in self._user_funcs:
            # Bare identifier matching a user function → call with no args
            return self._call_user_func(
                self._user_funcs[n.name],
                MethodCall(None, n.name, [], {}, None),
            )
        if val is None and n.name in self._BARE_BUILTINS:
            return self._eval_MethodCall(MethodCall(None, n.name, [], {}, None))
        return val

    def _eval_Assign(self, n: Assign) -> Any:
        val = self._eval_node(n.value)
        self._variables[n.name] = val
        return val

    def _eval_MultiAssign(self, n: MultiAssign) -> Any:
        values = [self._eval_node(v) for v in n.values]
        for i, name in enumerate(n.names):
            if len(n.values) == 1 and isinstance(values[0], list):
                # a, b = some_array → splat
                lst = values[0]
                self._variables[name] = lst[i] if i < len(lst) else None
            else:
                self._variables[name] = values[i] if i < len(values) else None
        return None

    def _eval_TernaryExpr(self, n: TernaryExpr) -> Any:
        cond = self._eval_node(n.cond)
        return self._eval_node(n.then_) if cond else self._eval_node(n.else_)

    def _eval_HashLit(self, n: HashLit) -> dict:
        return {k: self._eval_node(v) for k, v in n.pairs.items()}

    # ── Operators ────────────────────────────────────────────────────────

    def _eval_BinOp(self, n: BinOp) -> Any:
        l = self._eval_node(n.left)
        r = self._eval_node(n.right)
        try:
            if n.op == '+':  return l + r
            if n.op == '-':  return l - r
            if n.op == '*':  return l * r
            if n.op == '/':  return l / r if r else 0
            if n.op == '%':  return l % r if r else 0
            if n.op == '**': return l ** r
            if n.op == '==': return l == r
            if n.op == '!=': return l != r
            if n.op == '<':  return l < r
            if n.op == '>':  return l > r
            if n.op == '<=': return l <= r
            if n.op == '>=': return l >= r
            if n.op == 'and': return l and r
            if n.op == 'or':  return l or r
        except Exception:
            pass
        return None

    def _eval_UnaryOp(self, n: UnaryOp) -> Any:
        v = self._eval_node(n.operand)
        if n.op == '-' and isinstance(v, (int, float)):
            return -v
        if n.op == 'not':
            return not v
        return v

    # ── Control flow ─────────────────────────────────────────────────────

    def _eval_IfStmt(self, n: IfStmt):
        cond = self._eval_node(n.condition)
        if cond:
            return self._eval_body_last(n.then_body)
        else:
            for ec, eb in n.elsif_clauses:
                if self._eval_node(ec):
                    return self._eval_body_last(eb)
            if n.else_body:
                return self._eval_body_last(n.else_body)
        return None

    def _eval_WhileStmt(self, n: WhileStmt):
        guard = 0
        while self._eval_node(n.condition) and guard < 10_000:
            self._eval_body(n.body)
            guard += 1

    def _eval_ReturnStmt(self, n: ReturnStmt):
        raise _ReturnSignal(self._eval_node(n.value) if n.value else None)

    def _eval_Program(self, n: Program):
        self._eval_body(n.statements)

    def _eval_CaseStmt(self, n: CaseStmt):
        subject = self._eval_node(n.expr) if n.expr else True
        for val_node, body in n.whens:
            val = self._eval_node(val_node)
            # case expr; when val → subject == val
            # case; when cond    → cond is truthy
            match = (subject == val) if n.expr else bool(val)
            if match:
                self._eval_body(body)
                return
        if n.else_body:
            self._eval_body(n.else_body)

    def _eval_FuncDef(self, n: FuncDef):
        # Store user-defined function/define
        self._user_funcs[n.name] = n

    # ── Method calls (main dispatch) ─────────────────────────────────────

    def _eval_MethodCall(self, node: MethodCall) -> Any:
        method = node.method

        # Receiver-based calls first (e.g. 8.times, ring.tick)
        if node.receiver is not None:
            return self._eval_receiver_call(node)

        # Top-level Sonic Pi DSL calls
        dispatch = {
            # Time
            'sleep':               self._call_sleep,
            'use_bpm':             self._call_use_bpm,
            'with_bpm':            self._call_with_bpm,
            'use_bpm_multiplier':  self._call_use_bpm_multiplier,
            # Synth
            'play':                self._call_play,
            'synth':               self._call_synth,
            'use_synth':           self._call_use_synth,
            'with_synth':          self._call_with_synth,
            'use_synth_defaults':  self._call_use_synth_defaults,
            'with_synth_defaults': self._call_with_synth_defaults,
            # Sample
            'sample':              self._call_sample,
            'use_sample_defaults': self._call_use_sample_defaults,
            # FX
            'with_fx':             self._call_with_fx,
            # Concurrency
            'live_loop':           self._call_live_loop,
            'in_thread':           self._call_in_thread,
            # Music theory
            'note':                self._call_note,
            'chord':               self._call_chord,
            'scale':               self._call_scale,
            'midi_notes':          self._call_chord,
            # Sequencing / misc  (evaluate block / return value)
            'ring':                self._call_ring,
            'spread':              self._call_spread,
            'rrand':               self._call_rrand,
            'rrand_i':             self._call_rrand_i,
            'rand':                self._call_rand,
            'rand_i':              self._call_rand_i,
            'choose':              self._call_choose,
            'shuffle':             self._call_shuffle,
            'reverse':             self._call_reverse,
            'rotate':              self._call_rotate,
            'mirror':              self._call_mirror,
            'tick':                self._call_tick,
            'look':                self._call_look,
            'stop':                self._call_stop,
            # Probability / random
            'one_in':              self._call_one_in,
            'dice':                self._call_dice,
            'coin_flip':           self._call_coin_flip,
            'rdist':               self._call_rdist,
            'pick':                self._call_pick,
            # Math helpers
            'hz_to_midi':          self._call_hz_to_midi,
            'midi_to_hz':          self._call_midi_to_hz,
            'amp_to_db':           self._call_amp_to_db,
            'db_to_amp':           self._call_db_to_amp,
            'factor?':             self._call_factor,
            'quantise':            self._call_quantise,
            'quantize':            self._call_quantise,
            'inc':                 self._call_inc,
            'dec':                 self._call_dec,
            # Beat/time helpers
            'bt':                  self._call_bt,
            'rt':                  self._call_rt,
            # Concurrency helpers
            'density':             self._call_density,
            'at':                  self._call_at,
            'on':                  self._call_on,
            # Music theory extras
            'chord_invert':        self._call_chord_invert,
            'invert_chord':        self._call_chord_invert,
            'chord_degree':        self._call_chord_degree,
            'degree':              self._call_degree,
            'note_range':          self._call_note_range,
            # Transpose / octave
            'use_transpose':       self._call_use_transpose,
            'with_transpose':      self._call_with_transpose,
            'use_octave':          self._call_use_octave,
            'with_octave':         self._call_with_octave,
            # List constructors
            'knit':                self._call_knit,
            'bools':               self._call_bools,
            'range':               self._call_range,
            'stretch':             self._call_stretch,
            'line':                self._call_line,
            'vector':              self._call_ring,
            'ramp':                self._call_ring,
            # Define / defonce
            'define':              self._call_define,
            'defonce':             self._call_define,
            # Swing
            'with_swing':          self._call_with_swing,
            # Pattern helpers
            'play_pattern_timed':  self._call_play_pattern_timed,
            'play_pattern':        self._call_play_pattern,
            'play_chord':          self._call_play_chord,
            # BPM multiplier
            'with_bpm_mul':        self._call_with_bpm_mul,
            'use_bpm_mul':         self._call_with_bpm_mul,
            # Loop (infinite loop, unrolled like live_loop)
            'loop':                self._call_loop,
            # Random seed
            'use_random_seed':     self._call_use_random_seed,
            'with_random_seed':    self._call_with_random_seed,
            # State getters
            'current_bpm':         self._call_current_bpm,
            'current_synth':       self._call_current_synth,
            'current_synth_name':  self._call_current_synth,
            'current_time':        self._call_beat,
            'current_time_in_beats': self._call_beat,
            # Music info
            'note_info':           self._call_note_info,
            'chord_names':         self._call_chord_names,
            'scale_names':         self._call_scale_names,
            'sample_names':        self._call_sample_names,
            # Merged synth defaults (merge with existing, don't replace)
            'use_merged_synth_defaults':  self._call_use_synth_defaults,
            'with_merged_synth_defaults': self._call_with_synth_defaults,
            # Cent tuning
            'use_cent_tuning':     self._call_use_cent_tuning,
            'with_cent_tuning':    self._call_with_cent_tuning,
            # Beat / time helpers
            'beat':                self._call_beat,
            'sleep_bpm':           self._call_sleep_bpm,
            'current_beat':        self._call_beat,
            # Tick helpers
            'reset_tick':          self._call_reset_tick,
            'tick_set':            self._call_tick_set,
            # Key-value store
            'set':                 self._call_set,
            'get':                 self._call_get,
            # Lambda / Proc
            'lambda':              self._call_lambda,
            'proc':                self._call_lambda,
            # Type conversions
            'Integer':             self._call_integer_conv,
            'Float':               self._call_float_conv,
            'String':              self._call_string_conv,
            # Pitch / frequency helpers
            'pitch_to_ratio':      self._call_pitch_to_ratio,
            'ratio_to_pitch':      self._call_ratio_to_pitch,
            # Noops
            'load_sample':         self._call_noop,
            'load_samples':        self._call_noop,
            'puts':                self._call_noop,
            'print':               self._call_noop,
            'p':                   self._call_noop,
            'with_fx_level':       self._call_noop,
            'sync':                self._call_noop,
            'cue':                 self._call_noop,
            'comment':             self._call_noop,
            'uncomment':           self._call_noop,
            'use_timing_guarantees': self._call_noop,
            'use_real_time':       self._call_noop,
            'use_debug':           self._call_noop,
            'use_arg_checks':      self._call_noop,
            'use_midi_defaults':   self._call_noop,
            'spark':               self._call_noop,
            'midi':                self._call_noop,
            'assert':              self._call_noop,
            'assert_equal':        self._call_noop,
            'run_file':            self._call_noop,
            'load_synthdefs':      self._call_noop,
            'free_all':            self._call_noop,
            'with_efx':            self._call_with_fx,   # alias
        }

        handler = dispatch.get(method)
        if handler:
            return handler(node)

        # User-defined function call
        func = self._user_funcs.get(method)
        if func is not None:
            return self._call_user_func(func, node)

        # Unknown free-standing call – evaluate block if present
        if node.block:
            self._eval_body(node.block.body)
        return None

    # ── Receiver-based calls ─────────────────────────────────────────────

    def _eval_receiver_call(self, node: MethodCall) -> Any:
        method = node.method
        recv_val = self._eval_node(node.receiver)

        # Math module  Math.sqrt(...) etc.
        if recv_val == '__Math__':
            args = [self._eval_node(a) for a in node.args]
            x = self._to_float(args[0]) if args else 0.0
            if method == 'sqrt':   return math.sqrt(max(0.0, x))
            if method == 'sin':    return math.sin(x)
            if method == 'cos':    return math.cos(x)
            if method == 'tan':    return math.tan(x)
            if method == 'log':    return math.log(x) if x > 0 else 0.0
            if method == 'log2':   return math.log2(x) if x > 0 else 0.0
            if method == 'log10':  return math.log10(x) if x > 0 else 0.0
            if method == 'exp':    return math.exp(x)
            if method == 'cbrt':
                return math.copysign(abs(x) ** (1/3), x)
            if method == 'hypot':
                y = self._to_float(args[1]) if len(args) > 1 else 0.0
                return math.hypot(x, y)
            if method == 'floor':  return math.floor(x)
            if method == 'ceil':   return math.ceil(x)
            if method == 'abs':    return abs(x)
            if method == 'PI':     return math.pi
            if method == 'E':      return math.e
            return None

        # Array class methods  Array.new(n, val)
        if recv_val == 'Array' and method == 'new':
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 0
            val = args[1] if len(args) > 1 else None
            return [val] * n

        # nil receiver  nil.to_i / nil.to_f / nil.to_s / nil.to_a
        if recv_val is None:
            if method == 'to_i':   return 0
            if method == 'to_f':   return 0.0
            if method == 'to_s':   return ''
            if method == 'to_a':   return []
            if method in ('nil?', 'null?'): return True
            return None

        # Lambda / Proc call
        if isinstance(recv_val, _Lambda) and method == 'call':
            args = [self._eval_node(a) for a in node.args]
            saved = dict(self._variables)
            self._variables.update(recv_val.closure)
            for i, param in enumerate(recv_val.params):
                self._variables[param] = args[i] if i < len(args) else None
            result = None
            try:
                result = self._eval_body_last(recv_val.body)
            except _ReturnSignal as r:
                result = r.value
            self._variables = saved
            return result

        # N.times do ... end
        if method == 'times':
            n = int(recv_val) if isinstance(recv_val, (int, float)) else 0
            if node.block:
                for i in range(n):
                    if node.block.params:
                        self._variables[node.block.params[0]] = i
                    self._eval_body(node.block.body)
            return None

        if method == 'upto' and isinstance(recv_val, (int, float)) and node.block:
            args = [self._eval_node(a) for a in node.args]
            end = int(args[0]) if args else int(recv_val)
            for i in range(int(recv_val), end + 1):
                if node.block.params:
                    self._variables[node.block.params[0]] = i
                self._eval_body(node.block.body)
            return None

        if method == 'downto' and isinstance(recv_val, (int, float)) and node.block:
            args = [self._eval_node(a) for a in node.args]
            end = int(args[0]) if args else int(recv_val)
            for i in range(int(recv_val), end - 1, -1):
                if node.block.params:
                    self._variables[node.block.params[0]] = i
                self._eval_body(node.block.body)
            return None

        # N.to_f / N.to_i / N.abs / N.floor / N.ceil / N.round
        if method == 'to_f' and isinstance(recv_val, (int, float)):
            return float(recv_val)
        if method == 'to_i' and isinstance(recv_val, (int, float)):
            return int(recv_val)
        if method in ('abs',):
            return abs(recv_val) if isinstance(recv_val, (int, float)) else recv_val
        if method == 'floor':
            return math.floor(recv_val) if isinstance(recv_val, (int, float)) else recv_val
        if method == 'ceil':
            return math.ceil(recv_val) if isinstance(recv_val, (int, float)) else recv_val
        if method == 'round':
            return round(recv_val) if isinstance(recv_val, (int, float)) else recv_val
        if method in ('even?',):
            return int(recv_val) % 2 == 0 if isinstance(recv_val, (int, float)) else False
        if method in ('odd?',):
            return int(recv_val) % 2 != 0 if isinstance(recv_val, (int, float)) else False
        if method in ('zero?',):
            return recv_val == 0
        if method in ('nil?', 'null?'):
            return recv_val is None
        if method in ('positive?',):
            return recv_val > 0 if isinstance(recv_val, (int, float)) else False
        if method in ('negative?',):
            return recv_val < 0 if isinstance(recv_val, (int, float)) else False
        if method in ('between?',):
            args = [self._eval_node(a) for a in node.args]
            lo = self._to_float(args[0]) if args else 0
            hi = self._to_float(args[1]) if len(args) > 1 else 0
            return lo <= recv_val <= hi if isinstance(recv_val, (int, float)) else False
        if method == 'clamp':
            args = [self._eval_node(a) for a in node.args]
            lo = self._to_float(args[0]) if args else 0
            hi = self._to_float(args[1]) if len(args) > 1 else 1
            return max(lo, min(hi, recv_val)) if isinstance(recv_val, (int, float)) else recv_val

        # Array / ring operations
        if method == 'tick' and isinstance(recv_val, list):
            key = self._eval_kwargs(node.kwargs).get('name', '_default')
            idx = self._variables.setdefault(f'__tick_{key}', 0)
            val = recv_val[idx % len(recv_val)] if recv_val else None
            self._variables[f'__tick_{key}'] = idx + 1
            return val
        if method == 'look' and isinstance(recv_val, list):
            key = self._eval_kwargs(node.kwargs).get('name', '_default')
            idx = self._variables.get(f'__tick_{key}', 0)
            return recv_val[idx % len(recv_val)] if recv_val else None
        if method == 'choose' and isinstance(recv_val, list):
            return self._rng.choice(recv_val) if recv_val else None
        if method == 'shuffle' and isinstance(recv_val, list):
            lst = list(recv_val); self._rng.shuffle(lst); return lst
        if method == 'reverse' and isinstance(recv_val, list):
            return list(reversed(recv_val))
        if method == 'first' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            return recv_val[:n] if n > 1 else (recv_val[0] if recv_val else None)
        if method == 'last' and isinstance(recv_val, list):
            return recv_val[-1] if recv_val else None
        if method == 'length' and isinstance(recv_val, list):
            return len(recv_val)
        if method == 'map' and isinstance(recv_val, list) and node.block:
            result = []
            for item in recv_val:
                if node.block.params:
                    self._variables[node.block.params[0]] = item
                val = None
                for stmt in node.block.body:
                    val = self._eval_node(stmt)
                result.append(val)
            return result
        if method == 'each' and isinstance(recv_val, list) and node.block:
            for item in recv_val:
                if node.block.params:
                    self._variables[node.block.params[0]] = item
                self._eval_body(node.block.body)
            return None
        if method == 'each_with_index' and isinstance(recv_val, list) and node.block:
            for i, item in enumerate(recv_val):
                if len(node.block.params) >= 2:
                    self._variables[node.block.params[0]] = item
                    self._variables[node.block.params[1]] = i
                self._eval_body(node.block.body)
            return None

        # Additional list operations
        if method == 'min' and isinstance(recv_val, list):
            return min(recv_val) if recv_val else None
        if method == 'max' and isinstance(recv_val, list):
            return max(recv_val) if recv_val else None
        if method == 'sum' and isinstance(recv_val, list):
            return sum(recv_val) if recv_val else 0
        if method in ('count', 'size', 'len') and isinstance(recv_val, list):
            return len(recv_val)
        if method == 'sort' and isinstance(recv_val, list):
            try:
                return sorted(recv_val)
            except TypeError:
                return list(recv_val)
        if method == 'sort_by' and isinstance(recv_val, list) and node.block:
            result = list(recv_val)
            def key_fn(item):
                if node.block.params:
                    self._variables[node.block.params[0]] = item
                v = None
                for s in node.block.body:
                    v = self._eval_node(s)
                return v if v is not None else 0
            try:
                return sorted(result, key=key_fn)
            except TypeError:
                return result
        if method == 'flatten' and isinstance(recv_val, list):
            def _flat(lst):
                out = []
                for item in lst:
                    if isinstance(item, list):
                        out.extend(_flat(item))
                    else:
                        out.append(item)
                return out
            return _flat(recv_val)
        if method == 'compact' and isinstance(recv_val, list):
            return [x for x in recv_val if x is not None]
        if method == 'uniq' and isinstance(recv_val, list):
            seen = []
            for x in recv_val:
                if x not in seen:
                    seen.append(x)
            return seen
        if method == 'include?' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            return args[0] in recv_val if args else False
        if method == 'zip' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            other = args[0] if args and isinstance(args[0], list) else []
            return [[recv_val[i], other[i] if i < len(other) else None]
                    for i in range(len(recv_val))]
        if method == 'take' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            return recv_val[:n]
        if method == 'drop' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            return recv_val[n:]
        if method == 'flatten_one_level' and isinstance(recv_val, list):
            out = []
            for item in recv_val:
                if isinstance(item, list):
                    out.extend(item)
                else:
                    out.append(item)
            return out
        if method == 'sample' and isinstance(recv_val, list):
            return self._rng.choice(recv_val) if recv_val else None
        if method == 'rotate' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            if not recv_val:
                return []
            n = n % len(recv_val)
            return recv_val[n:] + recv_val[:n]
        if method == 'flatten_depth' and isinstance(recv_val, list):
            return recv_val  # no-op for now
        if method == 'select' and isinstance(recv_val, list) and node.block:
            result = []
            for item in recv_val:
                if node.block.params:
                    self._variables[node.block.params[0]] = item
                val = None
                for s in node.block.body:
                    val = self._eval_node(s)
                if val:
                    result.append(item)
            return result
        if method in ('reject', 'filter_map') and isinstance(recv_val, list) and node.block:
            result = []
            negate = method == 'reject'
            for item in recv_val:
                if node.block.params:
                    self._variables[node.block.params[0]] = item
                val = None
                for s in node.block.body:
                    val = self._eval_node(s)
                keep = not val if negate else bool(val)
                if keep:
                    result.append(item if negate else val)
            return result
        if method in ('reduce', 'inject') and isinstance(recv_val, list):
            if not recv_val:
                return None
            args = [self._eval_node(a) for a in node.args]
            # Determine if there's an initial accumulator value and/or a symbol op
            # Forms: inject(:+), inject(0, :+), inject(0) {|a,b| ...}, inject {|a,b| ...}
            sym_op = None
            initial = None
            for a in args:
                if isinstance(a, str) and a in ('+', '-', '*', '/', '%'):
                    sym_op = a
                elif isinstance(a, (int, float)):
                    initial = a
            if node.block:
                acc = initial if initial is not None else recv_val[0]
                items = recv_val if initial is not None else recv_val[1:]
                for item in items:
                    if len(node.block.params) >= 2:
                        self._variables[node.block.params[0]] = acc
                        self._variables[node.block.params[1]] = item
                    acc = None
                    for s in node.block.body:
                        acc = self._eval_node(s)
                return acc
            elif sym_op:
                _ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
                        '*': lambda a, b: a * b, '/': lambda a, b: a / b if b else 0,
                        '%': lambda a, b: a % b if b else 0}
                op_fn = _ops[sym_op]
                acc = initial if initial is not None else recv_val[0]
                items = recv_val if initial is not None else recv_val[1:]
                for item in items:
                    acc = op_fn(acc, item)
                return acc
            return None
        if method == 'each_slice' and isinstance(recv_val, list) and node.block:
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            for i in range(0, len(recv_val), n):
                slice_ = recv_val[i:i + n]
                if node.block.params:
                    self._variables[node.block.params[0]] = slice_
                self._eval_body(node.block.body)
            return None
        if method == 'each_cons' and isinstance(recv_val, list) and node.block:
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            for i in range(len(recv_val) - n + 1):
                window = recv_val[i:i + n]
                if node.block.params:
                    self._variables[node.block.params[0]] = window
                self._eval_body(node.block.body)
            return None

        # Dict methods (for HashLit results stored in variables)
        if method == '[]' and isinstance(recv_val, dict):
            args = [self._eval_node(a) for a in node.args]
            key = args[0] if args else None
            return recv_val.get(str(key).lstrip(':'))

        # String methods
        if method == 'to_sym' and isinstance(recv_val, str):
            return recv_val
        if method == 'to_s':
            return str(recv_val) if recv_val is not None else ''
        if method == 'upcase' and isinstance(recv_val, str):
            return recv_val.upper()
        if method == 'downcase' and isinstance(recv_val, str):
            return recv_val.lower()
        if method == 'length' and isinstance(recv_val, str):
            return len(recv_val)
        if method == 'split' and isinstance(recv_val, str):
            args = [self._eval_node(a) for a in node.args]
            sep = str(args[0]) if args else ' '
            return recv_val.split(sep)

        # Array []
        if method == '[]' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            idx = int(args[0]) if args else 0
            return recv_val[idx % len(recv_val)] if recv_val else None

        # Type checking
        if method in ('is_a?', 'kind_of?', 'instance_of?'):
            args = [self._eval_node(a) for a in node.args]
            type_name = str(args[0]) if args else ''
            _type_map = {
                'Integer': int, 'Fixnum': int, 'Bignum': int,
                'Float': float, 'Numeric': (int, float),
                'String': str, 'Array': list, 'Hash': dict,
                'NilClass': type(None), 'TrueClass': bool, 'FalseClass': bool,
            }
            t = _type_map.get(type_name)
            if t is not None:
                return isinstance(recv_val, t)
            return False

        if method == 'respond_to?':
            args = [self._eval_node(a) for a in node.args]
            meth_name = str(args[0]) if args else ''
            if isinstance(recv_val, list):
                return meth_name in ('each', 'map', 'select', 'reject', 'include?',
                                     'length', 'size', 'first', 'last', 'push', 'pop',
                                     'sort', 'reverse', 'flatten', 'compact', 'uniq',
                                     'min', 'max', 'sum', 'reduce', 'inject')
            if isinstance(recv_val, (int, float)):
                return meth_name in ('to_i', 'to_f', 'to_s', 'abs', 'floor', 'ceil',
                                     'round', 'times', 'upto', 'downto',
                                     'even?', 'odd?', 'zero?', 'positive?', 'negative?')
            if isinstance(recv_val, str):
                return meth_name in ('to_s', 'to_i', 'to_f', 'upcase', 'downcase',
                                     'length', 'size', 'split', 'reverse', 'include?')
            return False

        if method in ('class',):
            if isinstance(recv_val, bool): return 'TrueClass' if recv_val else 'FalseClass'
            if isinstance(recv_val, int): return 'Integer'
            if isinstance(recv_val, float): return 'Float'
            if isinstance(recv_val, str): return 'String'
            if isinstance(recv_val, list): return 'Array'
            if isinstance(recv_val, dict): return 'Hash'
            if recv_val is None: return 'NilClass'
            return 'Object'

        return None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _eval_args(self, node: MethodCall) -> tuple[list, dict]:
        args = [self._eval_node(a) for a in node.args]
        kwargs = self._eval_kwargs(node.kwargs)
        return args, kwargs

    def _eval_kwargs(self, raw: dict) -> dict:
        return {k: self._eval_node(v) for k, v in raw.items()}

    def _to_float(self, v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _resolve_note(self, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            midi = float(v)
        elif isinstance(v, str):
            midi = note_to_midi(v)
        else:
            return None
        if midi is None:
            return None
        return midi + self._transpose + self._octave * 12 + self._cent_tuning / 100.0

    # ── sleep ────────────────────────────────────────────────────────────

    def _call_sleep(self, node: MethodCall):
        args, _ = self._eval_args(node)
        beats = self._to_float(args[0] if args else 1, 1.0)
        self._time += beats

    # ── use_bpm / with_bpm ───────────────────────────────────────────────

    def _call_use_bpm(self, node: MethodCall):
        args, _ = self._eval_args(node)
        bpm = self._to_float(args[0] if args else 60, 60.0)
        if bpm > 0:
            self._bpm = bpm

    def _call_with_bpm(self, node: MethodCall):
        args, _ = self._eval_args(node)
        bpm = self._to_float(args[0] if args else 60, 60.0)
        old_bpm = self._bpm
        if bpm > 0:
            self._bpm = bpm
        if node.block:
            self._eval_body(node.block.body)
        self._bpm = old_bpm

    def _call_use_bpm_multiplier(self, node: MethodCall):
        args, _ = self._eval_args(node)
        mult = self._to_float(args[0] if args else 1.0, 1.0)
        if mult > 0:
            self._bpm *= mult

    # ── use_synth / with_synth ───────────────────────────────────────────

    def _call_use_synth(self, node: MethodCall):
        args, _ = self._eval_args(node)
        if args:
            self._current_synth = str(args[0]).lstrip(':')

    def _call_with_synth(self, node: MethodCall):
        args, _ = self._eval_args(node)
        old = self._current_synth
        if args:
            self._current_synth = str(args[0]).lstrip(':')
        if node.block:
            self._eval_body(node.block.body)
        self._current_synth = old

    def _call_use_synth_defaults(self, node: MethodCall):
        _, kwargs = self._eval_args(node)
        self._synth_defaults.update(kwargs)

    def _call_with_synth_defaults(self, node: MethodCall):
        _, kwargs = self._eval_args(node)
        old = dict(self._synth_defaults)
        self._synth_defaults.update(kwargs)
        if node.block:
            self._eval_body(node.block.body)
        self._synth_defaults = old

    def _call_use_sample_defaults(self, node: MethodCall):
        _, kwargs = self._eval_args(node)
        self._sample_defaults.update(kwargs)

    # ── play ─────────────────────────────────────────────────────────────

    def _call_play(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        if not args:
            return None

        note_val = args[0]

        # on: false → skip
        if not kwargs.get('on', True):
            return None

        # Merge defaults
        merged = dict(self._synth_defaults)
        merged.update(kwargs)

        # Apply pitch offset
        pitch = self._to_float(merged.pop('pitch', 0), 0.0)

        def emit_one(n_raw):
            midi = self._resolve_note(n_raw)
            if midi is None:
                return
            midi += pitch
            a = dict(merged)
            a['note'] = midi
            a.setdefault('amp', 1.0)
            a.setdefault('pan', 0.0)
            a.setdefault('attack', 0.0)
            a.setdefault('decay', 0.0)
            a.setdefault('sustain', 0.0)
            a.setdefault('release', 1.0)
            a['out_bus'] = self._current_bus_out()
            self._emit_synth(self._current_synth, a)

        if isinstance(note_val, list):
            for n in note_val:
                emit_one(n)
        else:
            emit_one(note_val)

    # ── synth ────────────────────────────────────────────────────────────

    def _call_synth(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        if not args:
            return None

        synth_name = str(args[0]).lstrip(':')

        # on: false → skip
        if not kwargs.get('on', True):
            return None

        merged = dict(self._synth_defaults)
        merged.update(kwargs)

        # Chord via notes:
        notes_val = merged.pop('notes', None) or merged.get('note', None)
        if isinstance(notes_val, list):
            for n in notes_val:
                a = dict(merged)
                midi = self._resolve_note(n)
                if midi is None:
                    continue
                a['note'] = midi
                a.setdefault('amp', 1.0)
                a.setdefault('pan', 0.0)
                a.setdefault('release', 1.0)
                a['out_bus'] = self._current_bus_out()
                self._emit_synth(synth_name, a)
            return None

        note_raw = merged.get('note', 52)
        midi = self._resolve_note(note_raw)
        if midi is None:
            return None

        pitch = self._to_float(merged.pop('pitch', 0), 0.0)
        merged['note'] = midi + pitch
        merged.setdefault('amp', 1.0)
        merged.setdefault('pan', 0.0)
        merged.setdefault('release', 1.0)
        merged['out_bus'] = self._current_bus_out()

        self._emit_synth(synth_name, merged)

    # ── sample ───────────────────────────────────────────────────────────

    def _call_sample(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        if not args:
            return None

        # on: false → skip
        if not kwargs.get('on', True):
            return None

        # Resolve sample name / path
        sample_id = args[0]
        if isinstance(sample_id, str):
            path = self._sample_resolver.resolve(sample_id)
        else:
            path = None

        if path is None:
            # Store symbolic name so codegen can handle it
            path = str(sample_id).lstrip(':')

        merged = dict(self._sample_defaults)
        merged.update(kwargs)
        merged.setdefault('rate',    1.0)
        merged.setdefault('amp',     1.0)
        merged.setdefault('pan',     0.0)
        merged.setdefault('attack',  0.0)
        merged.setdefault('release', 0.0)
        merged.setdefault('start',   0.0)
        merged.setdefault('finish',  1.0)
        merged['out_bus'] = self._current_bus_out()

        # beat_stretch → rate conversion (approximate; needs sample duration)
        # We store beat_stretch as-is; codegen will handle it
        self._emit_sample(path, merged)

    # ── with_fx ──────────────────────────────────────────────────────────

    def _call_with_fx(self, node: MethodCall):
        if not node.block:
            return

        args, kwargs = self._eval_args(node)
        if not args:
            return

        fx_name = str(args[0]).lstrip(':')

        # Allocate a fresh bus for this FX layer
        fx_bus = self._alloc_bus()
        outer_bus = self._current_bus_out()
        nid = self._alloc_node()

        frame = _FXFrame(
            fx_name=fx_name,
            opts=dict(kwargs),
            bus_in=fx_bus,
            bus_out=outer_bus,
            node_id=nid,
            start_time=self._time_secs,
        )
        self.events.append(SoundEvent(
            time=self._time_secs,
            kind='fx_open',
            synth_name=f'sonic-pi-fx_{fx_name}',
            node_id=nid,
            args={**kwargs, 'in_bus': fx_bus, 'out_bus': outer_bus},
            bus_out=outer_bus,
            bus_in=fx_bus,
        ))

        self._fx_stack.append(frame)
        self._eval_body(node.block.body)
        self._fx_stack.pop()

        # Close FX: schedule n_free after block ends + kill_delay
        kill_delay = self._fx_kill_delay(fx_name, kwargs)
        self.events.append(SoundEvent(
            time=self._time_secs + kill_delay,
            kind='fx_close',
            synth_name=f'sonic-pi-fx_{fx_name}',
            node_id=nid,
            args={},
        ))

    def _fx_kill_delay(self, fx_name: str, opts: dict) -> float:
        """Return tail time in seconds for an FX type."""
        bps = 60.0 / self._bpm
        if fx_name == 'reverb':
            room = float(opts.get('room', 0.6))
            return min((room * 10 + 1) * bps, 11 * bps)
        if fx_name == 'echo':
            return float(opts.get('decay', 2.0)) * bps
        if fx_name == 'gverb':
            return 1.0
        return 1.0 * bps

    # ── live_loop ────────────────────────────────────────────────────────

    def _call_live_loop(self, node: MethodCall):
        if not node.block:
            return

        args, kwargs = self._eval_args(node)
        iters = kwargs.get('iters', self._live_loop_iters)

        snap = self._clone_state()
        for _ in range(int(iters)):
            try:
                self._eval_body(node.block.body)
            except StopIteration_:
                break
        # Restore parent time/state so parallel live_loops all start at same base time
        self._restore_state(snap)

    # ── in_thread ────────────────────────────────────────────────────────

    def _call_in_thread(self, node: MethodCall):
        if not node.block:
            return

        snap = self._clone_state()
        self._eval_body(node.block.body)
        # Thread time advances independently; restore parent time
        self._restore_state(snap)

    # ── Music theory ─────────────────────────────────────────────────────

    def _call_note(self, node: MethodCall) -> float | None:
        args, _ = self._eval_args(node)
        if not args:
            return None
        return note_to_midi(args[0])

    def _call_chord(self, node: MethodCall) -> list[float]:
        args, kwargs = self._eval_args(node)
        root = args[0] if args else 'C'
        kind = str(args[1]).lstrip(':') if len(args) > 1 else 'major'
        invert = int(kwargs.get('invert', 0))
        num_oct = int(kwargs.get('num_octaves', 1))
        try:
            return mk_chord(root, kind, invert=invert, num_octaves=num_oct)
        except ValueError:
            return []

    def _call_scale(self, node: MethodCall) -> list[float]:
        args, kwargs = self._eval_args(node)
        root = args[0] if args else 'C'
        kind = str(args[1]).lstrip(':') if len(args) > 1 else 'major'
        num_oct = int(kwargs.get('num_octaves', 1))
        try:
            return mk_scale(root, kind, num_octaves=num_oct)
        except ValueError:
            return []

    # ── Sequencer helpers ────────────────────────────────────────────────

    def _call_ring(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        return list(args)

    def _call_spread(self, node: MethodCall) -> list[bool]:
        """Björklund / Euclidean rhythm."""
        args, _ = self._eval_args(node)
        hits = int(args[0]) if args else 0
        total = int(args[1]) if len(args) > 1 else 8
        pattern = _euclidean(hits, total)
        return pattern

    def _call_rrand(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        lo = self._to_float(args[0] if args else 0)
        hi = self._to_float(args[1] if len(args) > 1 else 1)
        return self._rng.uniform(lo, hi)

    def _call_rrand_i(self, node: MethodCall) -> int:
        args, _ = self._eval_args(node)
        lo = int(args[0]) if args else 0
        hi = int(args[1]) if len(args) > 1 else 10
        return self._rng.randint(lo, hi)

    def _call_rand(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        hi = self._to_float(args[0] if args else 1)
        return self._rng.uniform(0, hi)

    def _call_rand_i(self, node: MethodCall) -> int:
        args, _ = self._eval_args(node)
        hi = int(args[0]) if args else 10
        return self._rng.randint(0, hi)

    def _call_choose(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        lst = args[0] if (args and isinstance(args[0], list)) else args
        return self._rng.choice(lst) if lst else None

    def _call_shuffle(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        lst = list(args[0]) if (args and isinstance(args[0], list)) else []
        self._rng.shuffle(lst)
        return lst

    def _call_reverse(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        lst = args[0] if (args and isinstance(args[0], list)) else []
        return list(reversed(lst))

    def _call_rotate(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        lst = args[0] if (args and isinstance(args[0], list)) else []
        n = int(args[1]) if len(args) > 1 else 1
        if not lst:
            return []
        n = n % len(lst)
        return lst[n:] + lst[:n]

    def _call_mirror(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        lst = args[0] if (args and isinstance(args[0], list)) else []
        return lst + list(reversed(lst))

    def _call_tick(self, node: MethodCall) -> int:
        args, kwargs = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else '_default'
        idx = self._variables.setdefault(f'__tick_{key}', 0)
        self._variables[f'__tick_{key}'] = idx + 1
        return idx

    def _call_look(self, node: MethodCall) -> int:
        args, kwargs = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else '_default'
        return self._variables.get(f'__tick_{key}', 0)

    def _call_stop(self, node: MethodCall):
        raise StopIteration_()

    def _call_noop(self, node: MethodCall):
        return None

    def _call_loop(self, node: MethodCall):
        """loop do...end – unroll like live_loop."""
        if not node.block:
            return
        for _ in range(self._live_loop_iters):
            try:
                self._eval_body(node.block.body)
            except StopIteration_:
                break

    def _call_use_random_seed(self, node: MethodCall):
        args, _ = self._eval_args(node)
        seed = int(args[0]) if args else 0
        self._rng = random.Random(seed)

    def _call_with_random_seed(self, node: MethodCall):
        args, _ = self._eval_args(node)
        seed = int(args[0]) if args else 0
        old_rng = copy.deepcopy(self._rng)
        self._rng = random.Random(seed)
        if node.block:
            self._eval_body(node.block.body)
        self._rng = old_rng

    # ── User-defined functions (define / def) ─────────────────────────────

    def _call_define(self, node: MethodCall):
        """define :name do ... end  → store block as callable."""
        if not node.block:
            return None
        args, _ = self._eval_args(node)
        name = str(args[0]).lstrip(':') if args else None
        if name:
            # Store as a synthetic FuncDef
            self._user_funcs[name] = FuncDef(name, node.block.params, node.block.body)
        return None

    def _call_user_func(self, func: FuncDef, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        old_vars = dict(self._variables)
        positional = 0  # count of positional args consumed
        for i, param in enumerate(func.params):
            if param.startswith('*'):
                # Splat: collect all remaining args as list
                self._variables[param[1:]] = list(args[i:])
                positional = len(func.params)  # consumed all
                break
            if positional < len(args):
                self._variables[param] = args[positional]
            elif param in func.defaults:
                self._variables[param] = self._eval_node(func.defaults[param])
            else:
                self._variables[param] = None
            positional += 1
        result = None
        try:
            result = self._eval_body_last(func.body)
        except _ReturnSignal as r:
            result = r.value
        self._variables = old_vars
        return result

    # ── Probability ───────────────────────────────────────────────────────

    def _call_one_in(self, node: MethodCall) -> bool:
        args, _ = self._eval_args(node)
        n = int(args[0]) if args else 2
        return self._rng.randint(1, max(n, 1)) == 1

    def _call_dice(self, node: MethodCall) -> int:
        args, _ = self._eval_args(node)
        sides = int(args[0]) if args else 6
        return self._rng.randint(1, max(sides, 1))

    def _call_coin_flip(self, node: MethodCall) -> bool:
        return self._rng.randint(1, 2) == 1

    def _call_rdist(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        width = self._to_float(args[0] if args else 1)
        centre = self._to_float(args[1] if len(args) > 1 else 0)
        # triangular distribution approximating Sonic Pi's rdist
        return self._rng.triangular(centre - width, centre + width, centre)

    def _call_pick(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        if len(args) >= 2 and isinstance(args[1], list):
            n, lst = int(args[0]), args[1]
            return [self._rng.choice(lst) for _ in range(n)]
        if args and isinstance(args[0], list):
            return self._rng.choice(args[0])
        return None

    # ── Math helpers ─────────────────────────────────────────────────────

    def _call_hz_to_midi(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        hz = self._to_float(args[0] if args else 440)
        if hz <= 0:
            return 0.0
        return 69.0 + 12.0 * math.log2(hz / 440.0)

    def _call_midi_to_hz(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        midi = self._to_float(args[0] if args else 69)
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def _call_amp_to_db(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        amp = self._to_float(args[0] if args else 1.0)
        return 20.0 * math.log10(max(amp, 1e-10))

    def _call_db_to_amp(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        db = self._to_float(args[0] if args else 0.0)
        return 10.0 ** (db / 20.0)

    def _call_factor(self, node: MethodCall) -> bool:
        args, _ = self._eval_args(node)
        val = int(args[0]) if args else 0
        factor = int(args[1]) if len(args) > 1 else 1
        return factor != 0 and val % factor == 0

    def _call_quantise(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        n = self._to_float(args[0] if args else 0)
        step = self._to_float(args[1] if len(args) > 1 else 1)
        if step == 0:
            return n
        return round(n / step) * step

    def _call_inc(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        v = args[0] if args else 0
        return v + 1 if isinstance(v, (int, float)) else v

    def _call_dec(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        v = args[0] if args else 0
        return v - 1 if isinstance(v, (int, float)) else v

    # ── Beat / real-time converters ───────────────────────────────────────

    def _call_bt(self, node: MethodCall) -> float:
        """bt(t) – convert beats to real-time seconds at current BPM."""
        args, _ = self._eval_args(node)
        beats = self._to_float(args[0] if args else 1)
        return beats * (60.0 / self._bpm)

    def _call_rt(self, node: MethodCall) -> float:
        """rt(t) – convert real-time seconds to beats at current BPM."""
        args, _ = self._eval_args(node)
        secs = self._to_float(args[0] if args else 1)
        return secs * (self._bpm / 60.0)

    # ── density / at / on ─────────────────────────────────────────────────

    def _call_density(self, node: MethodCall):
        """density d do...end – run block d times at d× BPM."""
        if not node.block:
            return
        args, _ = self._eval_args(node)
        d = self._to_float(args[0] if args else 1, 1.0)
        d = max(d, 1.0)
        old_bpm = self._bpm
        self._bpm = self._bpm * d
        for _ in range(int(d)):
            try:
                self._eval_body(node.block.body)
            except StopIteration_:
                break
        self._bpm = old_bpm

    def _call_at(self, node: MethodCall):
        """at(times) do |t|...end – fork execution at each time offset."""
        if not node.block:
            return
        args, _ = self._eval_args(node)
        times = args[0] if args and isinstance(args[0], list) else [0.0]
        params_list = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        base_time = self._time
        for i, offset in enumerate(times):
            snap = self._clone_state()
            self._time = base_time + self._to_float(offset)
            if node.block.params:
                self._variables[node.block.params[0]] = offset
            if len(node.block.params) > 1 and i < len(params_list):
                self._variables[node.block.params[1]] = params_list[i]
            try:
                self._eval_body(node.block.body)
            except StopIteration_:
                pass
            self._restore_state(snap)

    def _call_on(self, node: MethodCall):
        """on(condition) do...end – execute block if condition is truthy."""
        if not node.block:
            return
        args, _ = self._eval_args(node)
        cond = args[0] if args else False
        if cond:
            self._eval_body(node.block.body)

    # ── Music theory extras ───────────────────────────────────────────────

    def _call_chord_invert(self, node: MethodCall) -> list[float]:
        args, _ = self._eval_args(node)
        notes = args[0] if args and isinstance(args[0], list) else []
        shift = int(args[1]) if len(args) > 1 else 1
        return mk_chord_invert(notes, shift)

    def _call_chord_degree(self, node: MethodCall) -> list[float]:
        args, kwargs = self._eval_args(node)
        degree = args[0] if args else 1
        tonic = args[1] if len(args) > 1 else 'C'
        scale_name = str(args[2]).lstrip(':') if len(args) > 2 else 'major'
        num_notes = int(args[3]) if len(args) > 3 else 4
        try:
            return mk_chord_degree(degree, tonic, scale_name, num_notes)
        except (ValueError, IndexError):
            return []

    def _call_degree(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        degree = args[0] if args else 1
        tonic = args[1] if len(args) > 1 else 'C'
        scale_name = str(args[2]).lstrip(':') if len(args) > 2 else 'major'
        try:
            return mk_degree(degree, tonic, scale_name)
        except (ValueError, IndexError):
            return 60.0

    def _call_note_range(self, node: MethodCall) -> list[float]:
        args, kwargs = self._eval_args(node)
        start = args[0] if args else 'C4'
        end = args[1] if len(args) > 1 else 'C5'
        return mk_note_range(start, end)

    # ── Transpose / octave ────────────────────────────────────────────────

    def _call_use_transpose(self, node: MethodCall):
        args, _ = self._eval_args(node)
        self._transpose = self._to_float(args[0] if args else 0)

    def _call_with_transpose(self, node: MethodCall):
        args, _ = self._eval_args(node)
        old = self._transpose
        self._transpose = self._to_float(args[0] if args else 0)
        if node.block:
            self._eval_body(node.block.body)
        self._transpose = old

    def _call_use_octave(self, node: MethodCall):
        args, _ = self._eval_args(node)
        self._octave = self._to_float(args[0] if args else 0)

    def _call_with_octave(self, node: MethodCall):
        args, _ = self._eval_args(node)
        old = self._octave
        self._octave = self._to_float(args[0] if args else 0)
        if node.block:
            self._eval_body(node.block.body)
        self._octave = old

    # ── List constructors ─────────────────────────────────────────────────

    def _call_knit(self, node: MethodCall) -> list:
        """knit(a, n, b, m, ...) → [a]*n + [b]*m + ..."""
        args, _ = self._eval_args(node)
        result = []
        i = 0
        while i + 1 < len(args):
            val, count = args[i], args[i + 1]
            result.extend([val] * max(int(count), 0))
            i += 2
        return result

    def _call_bools(self, node: MethodCall) -> list:
        """bools(1, 0, 1, 1) → [True, False, True, True]"""
        args, _ = self._eval_args(node)
        return [bool(a) for a in args]

    def _call_range(self, node: MethodCall) -> list:
        """range(start, finish, step: 1) → list of numbers."""
        args, kwargs = self._eval_args(node)
        start = self._to_float(args[0] if args else 0)
        finish = self._to_float(args[1] if len(args) > 1 else 1)
        step = self._to_float(kwargs.get('step', 1.0))
        if step == 0:
            return []
        result = []
        v = start
        if step > 0:
            while v <= finish + 1e-9:
                result.append(v)
                v += step
        else:
            while v >= finish - 1e-9:
                result.append(v)
                v += step
        return result

    def _call_stretch(self, node: MethodCall) -> list:
        """stretch(a, n, b, m, ...) → same as knit but each val is a list element."""
        return self._call_knit(node)

    def _call_line(self, node: MethodCall) -> list:
        """line(start, finish, steps: n) → evenly spaced list."""
        args, kwargs = self._eval_args(node)
        start = self._to_float(args[0] if args else 0)
        finish = self._to_float(args[1] if len(args) > 1 else 1)
        steps = int(kwargs.get('steps', kwargs.get('num_steps', 4)))
        if steps <= 1:
            return [start]
        step = (finish - start) / (steps - 1)
        return [start + step * i for i in range(steps)]

    # ── Swing ─────────────────────────────────────────────────────────────

    def _call_with_swing(self, node: MethodCall):
        """with_swing(amount) do...end – approximated: just run block."""
        if node.block:
            self._eval_body(node.block.body)

    # ── Pattern helpers ───────────────────────────────────────────────────

    def _emit_play_note(self, note_val: Any, kwargs: dict):
        """Shared helper: emit one note (or list) with merged kwargs/defaults."""
        if not kwargs.get('on', True):
            return
        merged = dict(self._synth_defaults)
        merged.update({k: v for k, v in kwargs.items() if k != 'on'})
        pitch = self._to_float(merged.pop('pitch', 0), 0.0)
        if isinstance(note_val, list):
            for n in note_val:
                midi = self._resolve_note(n)
                if midi is not None:
                    a = dict(merged)
                    a['note'] = midi + pitch
                    a.setdefault('amp', 1.0); a.setdefault('pan', 0.0)
                    a.setdefault('attack', 0.0); a.setdefault('decay', 0.0)
                    a.setdefault('sustain', 0.0); a.setdefault('release', 1.0)
                    a['out_bus'] = self._current_bus_out()
                    self._emit_synth(self._current_synth, a)
        else:
            midi = self._resolve_note(note_val)
            if midi is not None:
                a = dict(merged)
                a['note'] = midi + pitch
                a.setdefault('amp', 1.0); a.setdefault('pan', 0.0)
                a.setdefault('attack', 0.0); a.setdefault('decay', 0.0)
                a.setdefault('sustain', 0.0); a.setdefault('release', 1.0)
                a['out_bus'] = self._current_bus_out()
                self._emit_synth(self._current_synth, a)

    def _call_play_pattern_timed(self, node: MethodCall):
        """play_pattern_timed notes, times, **kwargs"""
        args, kwargs = self._eval_args(node)
        notes = args[0] if args and isinstance(args[0], list) else []
        times = args[1] if len(args) > 1 else [1.0]
        if not isinstance(times, list):
            times = [self._to_float(times, 1.0)]
        for i, note in enumerate(notes):
            self._emit_play_note(note, kwargs)
            self._time += self._to_float(times[i % len(times)], 1.0)

    def _call_play_pattern(self, node: MethodCall):
        """play_pattern notes, **kwargs  (sleep 1 between each note)"""
        args, kwargs = self._eval_args(node)
        notes = args[0] if args and isinstance(args[0], list) else args
        beat = self._to_float(kwargs.pop('sleep_time', 1.0), 1.0)
        for note in notes:
            self._emit_play_note(note, kwargs)
            self._time += beat

    def _call_play_chord(self, node: MethodCall):
        """play_chord [notes], **kwargs  (all simultaneously, no advance)"""
        args, kwargs = self._eval_args(node)
        notes = args[0] if args and isinstance(args[0], list) else args
        for note in notes:
            self._emit_play_note(note, kwargs)

    def _call_with_bpm_mul(self, node: MethodCall):
        """with_bpm_mul n do...end – multiply BPM by n within block."""
        args, _ = self._eval_args(node)
        mul = self._to_float(args[0] if args else 1.0, 1.0)
        old_bpm = self._bpm
        if mul > 0:
            self._bpm *= mul
        if node.block:
            self._eval_body(node.block.body)
        self._bpm = old_bpm

    # ── Cent tuning ───────────────────────────────────────────────────────

    def _call_use_cent_tuning(self, node: MethodCall):
        args, _ = self._eval_args(node)
        self._cent_tuning = self._to_float(args[0] if args else 0)

    def _call_with_cent_tuning(self, node: MethodCall):
        args, _ = self._eval_args(node)
        old = self._cent_tuning
        self._cent_tuning = self._to_float(args[0] if args else 0)
        if node.block:
            self._eval_body(node.block.body)
        self._cent_tuning = old

    # ── Beat / time helpers ───────────────────────────────────────────────

    def _call_beat(self, node: MethodCall) -> float:
        """Return current beat position."""
        return self._time

    def _call_sleep_bpm(self, node: MethodCall):
        """sleep_bpm n – sleep for n beats regardless of current BPM."""
        args, _ = self._eval_args(node)
        beats = self._to_float(args[0] if args else 1, 1.0)
        self._time += beats

    # ── Tick helpers ──────────────────────────────────────────────────────

    def _call_reset_tick(self, node: MethodCall):
        args, _ = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else '_default'
        self._variables[f'__tick_{key}'] = 0

    def _call_tick_set(self, node: MethodCall):
        args, _ = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else '_default'
        val = int(args[1]) if len(args) > 1 else 0
        self._variables[f'__tick_{key}'] = val

    # ── State getters ─────────────────────────────────────────────────────

    def _call_current_bpm(self, node: MethodCall) -> float:
        return self._bpm

    def _call_current_synth(self, node: MethodCall) -> str:
        return self._current_synth

    # ── Music info ────────────────────────────────────────────────────────

    def _call_note_info(self, node: MethodCall) -> dict:
        args, _ = self._eval_args(node)
        n = args[0] if args else 'C4'
        midi = note_to_midi(n)
        if midi is None:
            return {}
        octave = int(midi) // 12 - 1
        semitone = int(midi) % 12
        pitch_classes = ['C', 'Cs', 'D', 'Eb', 'E', 'F', 'Fs', 'G', 'Ab', 'A', 'Bb', 'B']
        return {
            'midi_note':   int(midi),
            'note_name':   str(n).lstrip(':'),
            'pitch_class': pitch_classes[semitone],
            'octave':      octave,
            'freq':        round(440.0 * (2.0 ** ((midi - 69.0) / 12.0)), 3),
        }

    def _call_chord_names(self, node: MethodCall) -> list:
        return list(CHORD_INTERVALS.keys())

    def _call_scale_names(self, node: MethodCall) -> list:
        return list(SCALE_INTERVALS.keys())

    def _call_sample_names(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        group = str(args[0]).lstrip(':') if args else ''
        return self._sample_resolver.list_group(group) if group else []

    # ── Key-value store ───────────────────────────────────────────────────

    def _call_set(self, node: MethodCall):
        args, _ = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else None
        val = args[1] if len(args) > 1 else None
        if key:
            self._store[key] = val

    def _call_get(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else None
        default = kwargs.get('default', None)
        if key:
            return self._store.get(key, default)
        return default

    # ── Lambda / Proc ─────────────────────────────────────────────────────

    def _call_lambda(self, node: MethodCall) -> '_Lambda | None':
        if node.block:
            return _Lambda(
                params=list(node.block.params),
                body=list(node.block.body),
                closure=dict(self._variables),
            )
        return None

    # ── Type conversions ──────────────────────────────────────────────────

    def _call_integer_conv(self, node: MethodCall) -> int:
        args, _ = self._eval_args(node)
        v = args[0] if args else 0
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    def _call_float_conv(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        v = args[0] if args else 0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _call_string_conv(self, node: MethodCall) -> str:
        args, _ = self._eval_args(node)
        v = args[0] if args else ''
        return str(v) if v is not None else ''

    # ── Pitch / frequency helpers ─────────────────────────────────────────

    def _call_pitch_to_ratio(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        semitones = self._to_float(args[0] if args else 0)
        return 2 ** (semitones / 12.0)

    def _call_ratio_to_pitch(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        ratio = self._to_float(args[0] if args else 1)
        if ratio <= 0:
            return 0.0
        return 12.0 * math.log2(ratio)


# ---------------------------------------------------------------------------
# Euclidean rhythm (Björklund algorithm)
# ---------------------------------------------------------------------------

def _euclidean(hits: int, steps: int) -> list[bool]:
    if steps <= 0:
        return []
    if hits <= 0:
        return [False] * steps
    hits = min(hits, steps)
    pattern = [[True] if i < hits else [False] for i in range(steps)]
    remainder = steps - hits
    while remainder > 1:
        groups = min(hits, remainder)
        for i in range(groups):
            pattern[i] = pattern[i] + pattern[-(i + 1)]
        pattern = pattern[:-groups] if groups < len(pattern) else pattern[:groups]
        hits = min(hits, remainder)
        remainder = len(pattern) - hits
    return [x[0] for x in pattern]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    program: 'Program',
    sonic_pi_root: str = ".",
    live_loop_iters: int = 8,
    rng_seed: Optional[int] = 42,
) -> list[SoundEvent]:
    """Evaluate a parsed Program and return sorted SoundEvents."""
    ev = Evaluator(
        sonic_pi_root=sonic_pi_root,
        live_loop_iters=live_loop_iters,
        rng_seed=rng_seed,
    )
    return ev.evaluate(program)
