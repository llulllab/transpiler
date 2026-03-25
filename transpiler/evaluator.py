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
    StringInterp, BeginRescue, RaiseStmt, YieldExpr, ClassDef, ModuleDef,
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
      'control'  – update params on a running node (SC: /n_set)
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

class _NextSignal(Exception):
    """Internal: raised by 'next' inside a loop iteration."""
    pass

class _BreakSignal(Exception):
    """Internal: raised by 'break' inside a loop."""
    def __init__(self, value=None): self.value = value

class _RubyException(Exception):
    """Internal: Ruby exception raised by 'raise'."""
    def __init__(self, value=None): self.value = value


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
        self._sample_pack: str = ''       # with_sample_pack prefix
        self._tuning: str = 'equal'       # use_tuning current value
        self._self: Any = None            # current 'self' object
        self._current_method: str = ''    # __method__ support

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
            'sample_pack': self._sample_pack,
            'tuning': self._tuning,
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
        self._sample_pack = snap.get('sample_pack', '')
        self._tuning = snap.get('tuning', 'equal')

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
                      'chord_names', 'scale_names', 'sample_names', 'synth_names',
                      'free_all', 'next', 'break', 'current_random_seed',
                      '__method__'}

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
        if n.name == '__self__':
            return self._self
        if n.name in ('Array', 'Integer', 'Float', 'String', 'Hash',
                       'Numeric', 'NilClass', 'TrueClass', 'FalseClass',
                       'Fixnum', 'Bignum', 'Proc', 'Symbol', 'Object',
                       'RuntimeError', 'StandardError', 'Exception',
                       'ArgumentError', 'TypeError', 'NameError',
                       'NoMethodError', 'ZeroDivisionError', 'StopIteration',
                       'IndexError', 'KeyError', 'RangeError', 'IOError'):
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
        # Flatten single array RHS
        if len(values) == 1 and isinstance(values[0], list):
            flat = values[0]
        else:
            flat = values
        # Find splat index in names (if any)
        splat_idx = next(
            (i for i, nm in enumerate(n.names) if nm.startswith('*')), None)
        if splat_idx is not None:
            # Assign pre-splat names
            for i in range(splat_idx):
                name = n.names[i]
                self._variables[name] = flat[i] if i < len(flat) else None
            # Post-splat names count
            post = n.names[splat_idx + 1:]
            n_post = len(post)
            splat_end = max(splat_idx, len(flat) - n_post)
            splat_name = n.names[splat_idx][1:] or '_'
            self._variables[splat_name] = flat[splat_idx:splat_end]
            # Assign post-splat names
            for j, name in enumerate(post):
                idx = splat_end + j
                self._variables[name] = flat[idx] if idx < len(flat) else None
        else:
            for i, name in enumerate(n.names):
                self._variables[name] = flat[i] if i < len(flat) else None
        return None

    def _eval_TernaryExpr(self, n: TernaryExpr) -> Any:
        cond = self._eval_node(n.cond)
        return self._eval_node(n.then_) if cond else self._eval_node(n.else_)

    def _eval_HashLit(self, n: HashLit) -> dict:
        return {k: self._eval_node(v) for k, v in n.pairs.items()}

    def _eval_StringInterp(self, n: StringInterp) -> str:
        parts = []
        for part in n.parts:
            val = self._eval_node(part)
            if val is None:
                parts.append('')
            elif isinstance(val, bool):
                parts.append('true' if val else 'false')
            elif isinstance(val, float) and val == int(val):
                parts.append(str(int(val)))
            else:
                parts.append(str(val))
        return ''.join(parts)

    def _eval_BeginRescue(self, n: BeginRescue) -> Any:
        result = None
        rescued = False
        try:
            result = self._eval_body_last(n.body)
        except (_ReturnSignal, _NextSignal, _BreakSignal, StopIteration_):
            raise  # always propagate control-flow signals
        except _RubyException as e:
            rescued = True
            for exc_type, exc_var, rc_body in n.rescue_clauses:
                if exc_var:
                    self._variables[exc_var] = e.value
                result = self._eval_body_last(rc_body)
                break
            if not n.rescue_clauses:
                raise
        except Exception as e:
            rescued = True
            for exc_type, exc_var, rc_body in n.rescue_clauses:
                if exc_var:
                    self._variables[exc_var] = str(e)
                result = self._eval_body_last(rc_body)
                break
            # If no rescue clauses matched, silently absorb Python errors
        else:
            if n.else_body:
                result = self._eval_body_last(n.else_body)
        finally:
            if n.ensure_body:
                self._eval_body(n.ensure_body)
        return result

    def _eval_RaiseStmt(self, n: RaiseStmt) -> None:
        val = self._eval_node(n.value) if n.value else None
        raise _RubyException(val)

    def _eval_YieldExpr(self, n: YieldExpr) -> Any:
        block = self._variables.get('__block__')
        if block is None:
            return None
        args = [self._eval_node(a) for a in n.args]
        if isinstance(block, _Lambda):
            saved = dict(self._variables)
            self._variables.update(block.closure)
            for i, param in enumerate(block.params):
                self._variables[param] = args[i] if i < len(args) else None
            result = None
            try:
                result = self._eval_body_last(block.body)
            except _ReturnSignal as r:
                result = r.value
            self._variables = saved
            return result
        return None

    def _eval_ClassDef(self, n: ClassDef) -> None:
        old_self = self._self
        self._self = n.name
        self._eval_body(n.body)
        self._self = old_self
        return None

    def _eval_ModuleDef(self, n: ModuleDef) -> None:
        self._eval_body(n.body)
        return None

    # ── Operators ────────────────────────────────────────────────────────

    def _eval_BinOp(self, n: BinOp) -> Any:
        l = self._eval_node(n.left)
        r = self._eval_node(n.right)
        try:
            if n.op == '+':
                if isinstance(l, list) and isinstance(r, list):
                    return l + r
                return l + r
            if n.op == '-':
                if isinstance(l, list) and isinstance(r, list):
                    # array difference:  [1,2,3] - [2]  →  [1,3]
                    r_set = set(id(x) for x in r)
                    return [x for x in l if x not in r]
                return l - r
            if n.op == '&':
                if isinstance(l, list) and isinstance(r, list):
                    seen = []
                    result_l = []
                    for x in l:
                        if x in r and x not in seen:
                            seen.append(x)
                            result_l.append(x)
                    return result_l
                # fallback: bitwise AND
                return int(l) & int(r)
            if n.op == '|':
                if isinstance(l, list) and isinstance(r, list):
                    seen = list(l)
                    for x in r:
                        if x not in seen:
                            seen.append(x)
                    return seen
                # fallback: bitwise OR
                return int(l) | int(r)
            if n.op == '*':  return l * r
            if n.op == '/':  return l / r if r else 0
            if n.op == '%':
                if isinstance(l, str):
                    try:
                        return l % (tuple(r) if isinstance(r, list) else r)
                    except Exception:
                        return l
                return l % r if r else 0
            if n.op == '**': return l ** r
            if n.op == '==': return l == r
            if n.op == '!=': return l != r
            if n.op == '<':  return l < r
            if n.op == '>':  return l > r
            if n.op == '<=': return l <= r
            if n.op == '>=': return l >= r
            if n.op == 'and': return l and r
            if n.op == 'or':  return l or r
            if n.op == '<<':
                if isinstance(l, list):
                    l.append(r)
                    return l
                if isinstance(l, str):
                    new_str = l + str(r) if r is not None else l
                    # Ruby strings are mutable: update the variable so s << x mutates s
                    if isinstance(n.left, Identifier):
                        self._variables[n.left.name] = new_str
                    return new_str
                if isinstance(l, int) and isinstance(r, int):
                    return l << r
                return l
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
            try:
                self._eval_body(n.body)
            except _NextSignal:
                pass
            except _BreakSignal:
                break
            guard += 1

    def _eval_ReturnStmt(self, n: ReturnStmt):
        raise _ReturnSignal(self._eval_node(n.value) if n.value else None)

    def _eval_Program(self, n: Program):
        self._eval_body(n.statements)

    def _eval_CaseStmt(self, n: CaseStmt):
        subject = self._eval_node(n.expr) if n.expr else True
        for val_node, body in n.whens:
            val = self._eval_node(val_node)
            if n.expr:
                # Range/list containment check
                if isinstance(val, list):
                    match = subject in val
                else:
                    match = subject == val
            else:
                match = bool(val)
            if match:
                return self._eval_body_last(body)
        if n.else_body:
            return self._eval_body_last(n.else_body)
        return None

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
            # (use_tuning / with_tuning handled below)
            # Synth info
            'synth_names':         self._call_synth_names,
            # String formatting
            'format':              self._call_format,
            'sprintf':             self._call_format,
            # Standalone math / utility
            'min':                 self._call_min_standalone,
            'max':                 self._call_max_standalone,
            'abs':                 self._call_abs_standalone,
            # Node control
            'control':             self._call_control,
            'live_audio':          self._call_noop,
            'kill':                self._call_noop,
            'with_arg_bpm_scaling': self._call_with_block_noop,
            'use_arg_bpm_scaling': self._call_noop,
            # Noops
            'load_sample':         self._call_noop,
            'load_samples':        self._call_noop,
            # (puts/print/p handled below)
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
            # (midi handled below)
            'assert':              self._call_noop,
            'assert_equal':        self._call_noop,
            'run_file':            self._call_noop,
            'load_synthdefs':      self._call_noop,
            'free_all':            self._call_noop,
            'with_efx':            self._call_with_fx,   # alias
            # Loop control
            'next':                self._call_next,
            'break':               self._call_break,
            # MIDI output
            'midi':                self._call_midi,
            'midi_note_on':        self._call_midi_note_on,
            'midi_note_off':       self._call_midi_note_off,
            'midi_cc':             self._call_midi_cc,
            'midi_pc':             self._call_noop,
            'midi_pitch_bend':     self._call_noop,
            'midi_all_notes_off':  self._call_noop,
            # Sample helpers
            'sample_duration':     self._call_sample_duration,
            'with_sample_pack':    self._call_with_sample_pack,
            'with_sample_pack_as': self._call_with_sample_pack,
            'use_sample_pack':     self._call_use_sample_pack,
            # Random state
            'current_random_seed': self._call_current_random_seed,
            'use_random_source':   self._call_noop,
            # Tuning
            'use_tuning':          self._call_use_tuning,
            'with_tuning':         self._call_with_tuning,
            # Run code
            'run_code':            self._call_run_code,
            'eval':                self._call_run_code,
            # method reference
            'method':              self._call_method_ref,
            # puts/print with output
            'puts':                self._call_puts,
            'print':               self._call_puts,
            'p':                   self._call_puts,
            'pp':                  self._call_puts,
            # Kernel conversion functions
            'Array':               self._call_kernel_Array,
            'Float':               self._call_kernel_Float,
            # __method__ inside defs
            '__method__':          self._call_current_method,
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

        # obj.send(:method_name, args...) — dynamic dispatch
        if method == 'send' or method == '__send__' or method == 'public_send':
            args = [self._eval_node(a) for a in node.args]
            if not args:
                return None
            meth_name = str(args[0]).lstrip(':')
            from .ast_nodes import IntLit as _IL, FloatLit as _FL, StringLit as _SL
            sub_args = []
            for a in args[1:]:
                if isinstance(a, int):
                    sub_args.append(_IL(a))
                elif isinstance(a, float):
                    sub_args.append(_FL(a))
                else:
                    sub_args.append(_SL(str(a)))
            sub_node = MethodCall(node.receiver, meth_name, sub_args, {}, node.block)
            return self._eval_receiver_call(sub_node)

        # Proc.new { ... }
        if recv_val == 'Proc' and method == 'new':
            if node.block:
                return _Lambda(node.block.params, node.block.body, dict(self._variables))
            return None

        # Hash.new(default_val) / String.new
        if recv_val == 'Hash' and method == 'new':
            return {}
        if recv_val == 'String' and method == 'new':
            args = [self._eval_node(a) for a in node.args]
            return str(args[0]) if args else ''

        # Array class methods  Array.new(n, val) / Array.new(n) { |i| ... }
        if recv_val == 'Array' and method == 'new':
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 0
            if node.block:
                result = []
                for i in range(n):
                    if node.block.params:
                        self._variables[node.block.params[0]] = i
                    result.append(self._eval_body_last(node.block.body))
                return result
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
        if isinstance(recv_val, _Lambda):
            if method in ('call', '[]', 'yield'):
                args = [self._eval_node(a) for a in node.args]
                saved = dict(self._variables)
                self._variables.update(recv_val.closure)
                for i, param in enumerate(recv_val.params):
                    if param.startswith('*'):
                        self._variables[param[1:]] = list(args[i:])
                        break
                    self._variables[param] = args[i] if i < len(args) else None
                result = None
                try:
                    result = self._eval_body_last(recv_val.body)
                except _ReturnSignal as r:
                    result = r.value
                self._variables = saved
                return result
            if method == 'lambda?':
                return True
            if method == 'arity':
                return sum(1 for p in recv_val.params if not p.startswith('*'))
            if method == 'curry':
                return recv_val  # simplified — just return self

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
                    try:
                        self._eval_body(node.block.body)
                    except _NextSignal:
                        continue
                    except _BreakSignal:
                        break
            return None

        if method == 'upto' and isinstance(recv_val, (int, float)) and node.block:
            args = [self._eval_node(a) for a in node.args]
            end = int(args[0]) if args else int(recv_val)
            for i in range(int(recv_val), end + 1):
                if node.block.params:
                    self._variables[node.block.params[0]] = i
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
            return None

        if method == 'downto' and isinstance(recv_val, (int, float)) and node.block:
            args = [self._eval_node(a) for a in node.args]
            end = int(args[0]) if args else int(recv_val)
            for i in range(int(recv_val), end - 1, -1):
                if node.block.params:
                    self._variables[node.block.params[0]] = i
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
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
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
            return None
        if method == 'each_with_index' and isinstance(recv_val, list) and node.block:
            for i, item in enumerate(recv_val):
                if len(node.block.params) >= 2:
                    self._variables[node.block.params[0]] = item
                    self._variables[node.block.params[1]] = i
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
            return None

        # Additional list operations
        if method == 'min' and isinstance(recv_val, list):
            return min(recv_val) if recv_val else None
        if method == 'max' and isinstance(recv_val, list):
            return max(recv_val) if recv_val else None
        if method == 'sum' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            initial = args[0] if args else 0
            try:
                return initial + (sum(recv_val) if recv_val else 0)
            except TypeError:
                return initial
        if method in ('count', 'size', 'len') and isinstance(recv_val, list):
            if method == 'count' and node.block:
                n = 0
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    if self._eval_body_last(node.block.body):
                        n += 1
                return n
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
            args_f = [self._eval_node(a) for a in node.args]
            depth = int(args_f[0]) if args_f else None
            def _flat(lst, d):
                out = []
                for item in lst:
                    if isinstance(item, list) and (d is None or d > 0):
                        out.extend(_flat(item, None if d is None else d - 1))
                    else:
                        out.append(item)
                return out
            return _flat(recv_val, depth)
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
            others = [a if isinstance(a, list) else [] for a in args]
            return [[recv_val[i]] + [o[i] if i < len(o) else None for o in others]
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
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
            return None
        if method == 'each_cons' and isinstance(recv_val, list) and node.block:
            args = [self._eval_node(a) for a in node.args]
            n = int(args[0]) if args else 1
            for i in range(len(recv_val) - n + 1):
                window = recv_val[i:i + n]
                if node.block.params:
                    self._variables[node.block.params[0]] = window
                try:
                    self._eval_body(node.block.body)
                except _NextSignal:
                    continue
                except _BreakSignal:
                    break
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
            args = [self._eval_node(a) for a in node.args]
            base = int(args[0]) if args else 10
            if isinstance(recv_val, int) and base != 10:
                return format(recv_val, 'x' if base == 16 else
                              'o' if base == 8 else 'b' if base == 2 else 'd')
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

        # Subscript assignment  obj[k] = v
        if method == '[]=' :
            args = [self._eval_node(a) for a in node.args]
            if isinstance(recv_val, dict):
                key = str(args[0]).lstrip(':') if args else ''
                val = args[1] if len(args) > 1 else None
                recv_val[key] = val
                return val
            if isinstance(recv_val, list):
                idx = int(args[0]) if args else 0
                val = args[1] if len(args) > 1 else None
                # Grow list if needed
                while idx >= len(recv_val):
                    recv_val.append(None)
                recv_val[idx] = val
                return val
            return None

        # Array []  (also handles range slicing and negative indexing)
        if method == '[]' and isinstance(recv_val, list):
            args = [self._eval_node(a) for a in node.args]
            if not args or not recv_val:
                return None
            idx_val = args[0]
            if isinstance(idx_val, list):
                # range slice: arr[1..3] evaluates to a list of indices
                return [recv_val[i % len(recv_val)] for i in idx_val if isinstance(i, int)]
            idx = int(idx_val)
            # Support negative indexing (Ruby: arr[-1] = last element)
            if idx < 0:
                return recv_val[idx] if abs(idx) <= len(recv_val) else None
            if len(args) > 1:
                # arr[start, length] slice
                length = int(args[1])
                return recv_val[idx:idx + length]
            return recv_val[idx] if idx < len(recv_val) else None

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
            meth_name = str(args[0]).lstrip(':') if args else ''
            if isinstance(recv_val, _Lambda):
                return meth_name in ('call', '[]', 'yield', 'lambda?', 'arity', 'curry')
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

        # ── tap / then / yield_self ───────────────────────────────────────
        if method == 'tap':
            if node.block:
                if node.block.params:
                    self._variables[node.block.params[0]] = recv_val
                self._eval_body(node.block.body)
            return recv_val

        if method in ('then', 'yield_self'):
            if node.block:
                if node.block.params:
                    self._variables[node.block.params[0]] = recv_val
                return self._eval_body_last(node.block.body)
            return recv_val

        if method in ('freeze', 'frozen?', 'dup', 'clone', 'itself'):
            return recv_val

        # ── Additional array methods ──────────────────────────────────────
        if isinstance(recv_val, list):
            if method in ('to_a', 'to_ary', 'entries'):
                return recv_val

            if method == 'pick':
                # pick(n=1) — random element(s) from the array
                args = [self._eval_node(a) for a in node.args]
                if not recv_val:
                    return None
                if args:
                    n = int(args[0])
                    return [self._rng.choice(recv_val) for _ in range(n)]
                return self._rng.choice(recv_val)

            if method == 'mirror':
                # [1,2,3].mirror → [1,2,3,2,1]  (don't repeat last element)
                return recv_val + list(reversed(recv_val[:-1]))

            if method == 'reflect':
                # [1,2,3].reflect → [1,2,3,3,2,1]
                return recv_val + list(reversed(recv_val))

            if method == 'ring':
                # Convert array to ring (just return as list — same semantics)
                return recv_val

            if method == 'stretch':
                # [60, 64].stretch(2) → [60, 60, 64, 64]
                args = [self._eval_node(a) for a in node.args]
                factor = int(args[0]) if args else 1
                result = []
                for item in recv_val:
                    result.extend([item] * factor)
                return result

            if method == 'repeat':
                # [60, 64].repeat(3) → [60, 64, 60, 64, 60, 64]
                args = [self._eval_node(a) for a in node.args]
                factor = int(args[0]) if args else 1
                return recv_val * factor

            if method == 'butlast':
                return recv_val[:-1] if recv_val else []

            if method == 'drop':
                args = [self._eval_node(a) for a in node.args]
                n = int(args[0]) if args else 1
                return recv_val[n:]

            if method == 'take':
                args = [self._eval_node(a) for a in node.args]
                n = int(args[0]) if args else 1
                return recv_val[:n]

            if method in ('flat_map', 'collect_concat') and node.block:
                result = []
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    val = self._eval_body_last(node.block.body)
                    if isinstance(val, list):
                        result.extend(val)
                    elif val is not None:
                        result.append(val)
                return result

            if method in ('any?',):
                if node.block:
                    for item in recv_val:
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if self._eval_body_last(node.block.body):
                            return True
                    return False
                return bool(recv_val)

            if method in ('all?',):
                if node.block:
                    for item in recv_val:
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if not self._eval_body_last(node.block.body):
                            return False
                    return True
                return all(recv_val)

            if method in ('none?',):
                if node.block:
                    for item in recv_val:
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if self._eval_body_last(node.block.body):
                            return False
                    return True
                return not any(recv_val)

            if method == 'join':
                args = [self._eval_node(a) for a in node.args]
                sep = str(args[0]) if args else ''
                return sep.join(str(x) for x in recv_val)

            if method in ('push', 'append'):
                args = [self._eval_node(a) for a in node.args]
                for a in args:
                    recv_val.append(a)
                return recv_val

            if method == 'pop':
                return recv_val.pop() if recv_val else None

            if method == 'shift':
                return recv_val.pop(0) if recv_val else None

            if method == 'unshift':
                args = [self._eval_node(a) for a in node.args]
                for a in reversed(args):
                    recv_val.insert(0, a)
                return recv_val

            if method == 'concat':
                args = [self._eval_node(a) for a in node.args]
                for a in args:
                    if isinstance(a, list):
                        recv_val.extend(a)
                return recv_val

            if method in ('find', 'detect'):
                if node.block:
                    for item in recv_val:
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if self._eval_body_last(node.block.body):
                            return item
                return None

            if method == 'partition' and node.block:
                yes, no = [], []
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    if self._eval_body_last(node.block.body):
                        yes.append(item)
                    else:
                        no.append(item)
                return [yes, no]

            if method == 'chunk' and isinstance(recv_val, list) and node.block:
                from itertools import groupby
                keyed = []
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    key = self._eval_body_last(node.block.body)
                    keyed.append((key, item))
                groups = []
                from itertools import groupby as _groupby
                for key, grp in _groupby(keyed, key=lambda kv: kv[0]):
                    groups.append([key, [kv[1] for kv in grp]])
                return groups

            if method == 'chunk_while' and isinstance(recv_val, list) and node.block:
                if not recv_val:
                    return []
                groups = [[recv_val[0]]]
                for item in recv_val[1:]:
                    prev = groups[-1][-1]
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = prev
                        self._variables[params[1]] = item
                    elif params:
                        self._variables[params[0]] = prev
                    if self._eval_body_last(node.block.body):
                        groups[-1].append(item)
                    else:
                        groups.append([item])
                return groups

            if method == 'tally' and isinstance(recv_val, list):
                result: dict = {}
                for item in recv_val:
                    k = str(item) if not isinstance(item, str) else item
                    result[k] = result.get(k, 0) + 1
                return result

            if method in ('combination',) and isinstance(recv_val, list):
                args = [self._eval_node(a) for a in node.args]
                n_combo = int(args[0]) if args else 2
                from itertools import combinations
                combos = list(combinations(recv_val, n_combo))
                if node.block:
                    for combo in combos:
                        if node.block.params:
                            self._variables[node.block.params[0]] = list(combo)
                        self._eval_body(node.block.body)
                    return recv_val
                return [list(c) for c in combos]

            if method in ('permutation',) and isinstance(recv_val, list):
                args = [self._eval_node(a) for a in node.args]
                n_perm = int(args[0]) if args else len(recv_val)
                from itertools import permutations
                perms = list(permutations(recv_val, n_perm))
                if node.block:
                    for perm in perms:
                        if node.block.params:
                            self._variables[node.block.params[0]] = list(perm)
                        self._eval_body(node.block.body)
                    return recv_val
                return [list(p) for p in perms]

            if method == 'product' and isinstance(recv_val, list):
                args = [self._eval_node(a) for a in node.args]
                from itertools import product
                others = [a if isinstance(a, list) else [a] for a in args]
                if not others:
                    return [[x] for x in recv_val]
                result2 = [list(p) for p in product(recv_val, *others)]
                if node.block:
                    for combo in result2:
                        if node.block.params:
                            self._variables[node.block.params[0]] = combo
                        self._eval_body(node.block.body)
                    return recv_val
                return result2

            if method in ('index', 'find_index'):
                args = [self._eval_node(a) for a in node.args]
                if args:
                    try:
                        return recv_val.index(args[0])
                    except ValueError:
                        return None
                if node.block:
                    for i, item in enumerate(recv_val):
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if self._eval_body_last(node.block.body):
                            return i
                return None

            if method == 'rindex':
                args = [self._eval_node(a) for a in node.args]
                if args:
                    for i in range(len(recv_val) - 1, -1, -1):
                        if recv_val[i] == args[0]:
                            return i
                return None

            if method == 'min_by' and node.block:
                if not recv_val:
                    return None
                def _key_min(item):
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    v = self._eval_body_last(node.block.body)
                    return v if v is not None else 0
                return min(recv_val, key=_key_min)

            if method == 'max_by' and node.block:
                if not recv_val:
                    return None
                def _key_max(item):
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    v = self._eval_body_last(node.block.body)
                    return v if v is not None else 0
                return max(recv_val, key=_key_max)

            if method == 'each_with_object' and node.block:
                args = [self._eval_node(a) for a in node.args]
                acc = args[0] if args else []
                for item in recv_val:
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = item
                        self._variables[params[1]] = acc
                    elif params:
                        self._variables[params[0]] = [item, acc]
                    self._eval_body(node.block.body)
                return acc

            if method in ('delete_if', 'reject!') and node.block:
                to_remove = []
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    if self._eval_body_last(node.block.body):
                        to_remove.append(item)
                for item in to_remove:
                    recv_val.remove(item)
                return recv_val

            if method in ('map!', 'collect!') and node.block:
                for i, item in enumerate(recv_val):
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    recv_val[i] = self._eval_body_last(node.block.body)
                return recv_val

            if method == 'delete':
                args = [self._eval_node(a) for a in node.args]
                target = args[0] if args else None
                found = None
                while target in recv_val:
                    recv_val.remove(target)
                    found = target
                return found

            if method == 'insert':
                args = [self._eval_node(a) for a in node.args]
                idx = int(args[0]) if args else 0
                vals = args[1:]
                for i, v in enumerate(vals):
                    recv_val.insert(idx + i, v)
                return recv_val

            if method == 'fill':
                args = [self._eval_node(a) for a in node.args]
                val = args[0] if args else None
                for i in range(len(recv_val)):
                    recv_val[i] = val
                return recv_val

            if method == 'product':
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args and isinstance(args[0], list) else []
                return [[a, b] for a in recv_val for b in other]

            if method == 'combination':
                args = [self._eval_node(a) for a in node.args]
                import itertools
                n = int(args[0]) if args else 2
                combos = list(itertools.combinations(recv_val, n))
                if node.block:
                    for combo in combos:
                        if node.block.params:
                            self._variables[node.block.params[0]] = list(combo)
                        self._eval_body(node.block.body)
                    return None
                return [list(c) for c in combos]

            if method == 'permutation':
                args = [self._eval_node(a) for a in node.args]
                import itertools
                n = int(args[0]) if args else len(recv_val)
                perms = list(itertools.permutations(recv_val, n))
                if node.block:
                    for perm in perms:
                        if node.block.params:
                            self._variables[node.block.params[0]] = list(perm)
                        self._eval_body(node.block.body)
                    return None
                return [list(p) for p in perms]

            if method == 'tally':
                result = {}
                for item in recv_val:
                    key = str(item)
                    result[key] = result.get(key, 0) + 1
                return result

            if method == 'group_by' and node.block:
                result = {}
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    key = self._eval_body_last(node.block.body)
                    key_s = str(key)
                    if key_s not in result:
                        result[key_s] = []
                    result[key_s].append(item)
                return result

            if method in ('empty?',):
                return len(recv_val) == 0

            if method == 'take_while' and node.block:
                result = []
                for item in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = item
                    if not self._eval_body_last(node.block.body):
                        break
                    result.append(item)
                return result

            if method == 'drop_while' and node.block:
                dropping = True
                result = []
                for item in recv_val:
                    if dropping:
                        if node.block.params:
                            self._variables[node.block.params[0]] = item
                        if not self._eval_body_last(node.block.body):
                            dropping = False
                            result.append(item)
                    else:
                        result.append(item)
                return result

            if method in ('count',) and node.args:
                args = [self._eval_node(a) for a in node.args]
                return recv_val.count(args[0]) if args else len(recv_val)

            if method == 'repeated_combination':
                args = [self._eval_node(a) for a in node.args]
                import itertools
                n = int(args[0]) if args else 2
                combos = list(itertools.combinations_with_replacement(recv_val, n))
                if node.block:
                    for c in combos:
                        if node.block.params:
                            self._variables[node.block.params[0]] = list(c)
                        self._eval_body(node.block.body)
                    return None
                return [list(c) for c in combos]

            if method == 'to_s':
                return '[' + ', '.join(str(x) for x in recv_val) + ']'

            if method == 'intersection':
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args and isinstance(args[0], list) else []
                return [x for x in recv_val if x in other]

            if method in ('union', '|'):
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args and isinstance(args[0], list) else []
                seen = list(recv_val)
                for x in other:
                    if x not in seen:
                        seen.append(x)
                return seen

            if method in ('difference', '-'):
                args = [self._eval_node(a) for a in node.args]
                other = set(args[0]) if args and isinstance(args[0], list) else set()
                return [x for x in recv_val if x not in other]

        # ── Hash receiver methods ──────────────────────────────────────────
        if isinstance(recv_val, dict):
            if method == 'keys':
                return list(recv_val.keys())
            if method == 'values':
                return list(recv_val.values())
            if method == 'fetch':
                args = [self._eval_node(a) for a in node.args]
                key = str(args[0]).lstrip(':') if args else ''
                default = args[1] if len(args) > 1 else None
                return recv_val.get(key, default)
            if method in ('has_key?', 'key?', 'include?', 'member?'):
                args = [self._eval_node(a) for a in node.args]
                key = str(args[0]).lstrip(':') if args else ''
                return key in recv_val
            if method in ('size', 'count', 'length'):
                return len(recv_val)
            if method in ('empty?',):
                return len(recv_val) == 0
            if method in ('merge', 'merge!') and isinstance(recv_val, dict):
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args and isinstance(args[0], dict) else {}
                if node.block:
                    for k, v in other.items():
                        if k in recv_val:
                            params = node.block.params
                            if len(params) >= 3:
                                self._variables[params[0]] = k
                                self._variables[params[1]] = recv_val[k]
                                self._variables[params[2]] = v
                            recv_val[k] = self._eval_body_last(node.block.body)
                        else:
                            recv_val[k] = v
                    return recv_val
                if method == 'merge!':
                    # Mutate in-place
                    recv_val.update(other)
                    return recv_val
                result = dict(recv_val)
                result.update(other)
                return result
            if method == 'update':
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args and isinstance(args[0], dict) else {}
                recv_val.update(other)
                return recv_val
            if method in ('each', 'each_pair') and node.block:
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    elif params:
                        self._variables[params[0]] = [k, v]
                    self._eval_body(node.block.body)
                return recv_val
            if method == 'map' and node.block:
                result = []
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    elif params:
                        self._variables[params[0]] = [k, v]
                    result.append(self._eval_body_last(node.block.body))
                return result
            if method in ('any?',) and node.block:
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    if self._eval_body_last(node.block.body):
                        return True
                return False
            if method in ('all?',) and node.block:
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    if not self._eval_body_last(node.block.body):
                        return False
                return True
            if method == 'select' and node.block:
                result = {}
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    elif params:
                        self._variables[params[0]] = [k, v]
                    if self._eval_body_last(node.block.body):
                        result[k] = v
                return result
            if method == 'reject' and node.block:
                result = {}
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    if not self._eval_body_last(node.block.body):
                        result[k] = v
                return result
            if method == '[]':
                args = [self._eval_node(a) for a in node.args]
                key = str(args[0]).lstrip(':') if args else ''
                return recv_val.get(key)
            if method == 'to_a':
                return [[k, v] for k, v in recv_val.items()]
            if method == 'to_s':
                return str(recv_val)
            if method == 'delete':
                args = [self._eval_node(a) for a in node.args]
                key = str(args[0]).lstrip(':') if args else ''
                return recv_val.pop(key, None)
            if method == 'store':
                args = [self._eval_node(a) for a in node.args]
                key = str(args[0]).lstrip(':') if args else ''
                val = args[1] if len(args) > 1 else None
                recv_val[key] = val
                return val
            if method in ('each_with_object',) and node.block:
                args = [self._eval_node(a) for a in node.args]
                acc = args[0] if args else {}
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = [k, v]
                        self._variables[params[1]] = acc
                    self._eval_body(node.block.body)
                return acc
            if method == 'transform_values' and node.block:
                result = {}
                for k, v in recv_val.items():
                    if node.block.params:
                        self._variables[node.block.params[0]] = v
                    result[k] = self._eval_body_last(node.block.body)
                return result
            if method == 'transform_keys' and node.block:
                result = {}
                for k, v in recv_val.items():
                    if node.block.params:
                        self._variables[node.block.params[0]] = k
                    new_key = self._eval_body_last(node.block.body)
                    result[str(new_key) if new_key is not None else k] = v
                return result
            if method in ('filter', 'filter_map'):
                # filter is alias for select on Hash
                if node.block:
                    result = {}
                    for k, v in recv_val.items():
                        params = node.block.params
                        if len(params) >= 2:
                            self._variables[params[0]] = k
                            self._variables[params[1]] = v
                        elif params:
                            self._variables[params[0]] = [k, v]
                        if self._eval_body_last(node.block.body):
                            result[k] = v
                    return result
                return recv_val
            if method == 'invert':
                return {str(v): k for k, v in recv_val.items()}
            if method in ('to_h', 'dup', 'clone'):
                return dict(recv_val)
            if method == 'flatten':
                result = []
                for k, v in recv_val.items():
                    result.append(k)
                    result.append(v)
                return result
            if method in ('none?',) and node.block:
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    if self._eval_body_last(node.block.body):
                        return False
                return True
            if method == 'count':
                if node.block:
                    count = 0
                    for k, v in recv_val.items():
                        params = node.block.params
                        if len(params) >= 2:
                            self._variables[params[0]] = k
                            self._variables[params[1]] = v
                        elif params:
                            self._variables[params[0]] = [k, v]
                        if self._eval_body_last(node.block.body):
                            count += 1
                    return count
                return len(recv_val)
            if method == 'min_by' and node.block:
                if not recv_val:
                    return None
                best = None
                best_key = None
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    elif params:
                        self._variables[params[0]] = [k, v]
                    score = self._eval_body_last(node.block.body)
                    if best_key is None or (score is not None and score < best_key):
                        best = [k, v]
                        best_key = score
                return best
            if method == 'max_by' and node.block:
                if not recv_val:
                    return None
                best = None
                best_key = None
                for k, v in recv_val.items():
                    params = node.block.params
                    if len(params) >= 2:
                        self._variables[params[0]] = k
                        self._variables[params[1]] = v
                    elif params:
                        self._variables[params[0]] = [k, v]
                    score = self._eval_body_last(node.block.body)
                    if best_key is None or (score is not None and score > best_key):
                        best = [k, v]
                        best_key = score
                return best
            return None

        # ── Additional string methods ──────────────────────────────────────
        if isinstance(recv_val, str):
            if method == 'start_with?':
                args = [self._eval_node(a) for a in node.args]
                prefix = str(args[0]) if args else ''
                return recv_val.startswith(prefix)
            if method == 'end_with?':
                args = [self._eval_node(a) for a in node.args]
                suffix = str(args[0]) if args else ''
                return recv_val.endswith(suffix)
            if method == 'include?':
                args = [self._eval_node(a) for a in node.args]
                substr = str(args[0]) if args else ''
                return substr in recv_val
            if method == 'strip':
                return recv_val.strip()
            if method == 'lstrip':
                return recv_val.lstrip()
            if method == 'rstrip':
                return recv_val.rstrip()
            if method == 'chomp':
                return recv_val.rstrip('\n\r')
            if method == 'chop':
                return recv_val[:-1] if recv_val else ''
            if method == 'each_char' and node.block:
                for ch in recv_val:
                    if node.block.params:
                        self._variables[node.block.params[0]] = ch
                    try:
                        self._eval_body(node.block.body)
                    except _NextSignal:
                        continue
                    except _BreakSignal:
                        break
                return recv_val

            if method == 'each_line' and node.block:
                for line in recv_val.splitlines(keepends=True):
                    if node.block.params:
                        self._variables[node.block.params[0]] = line
                    try:
                        self._eval_body(node.block.body)
                    except _NextSignal:
                        continue
                    except _BreakSignal:
                        break
                return recv_val

            if method == 'lines':
                return recv_val.splitlines(keepends=True)

            if method == 'gsub':
                args = [self._eval_node(a) for a in node.args]
                pat = str(args[0]) if args else ''
                if node.block:
                    import re as _re
                    def _gsub_repl(m):
                        if node.block.params:
                            self._variables[node.block.params[0]] = m.group(0)
                        result = self._eval_body_last(node.block.body)
                        return str(result) if result is not None else ''
                    try:
                        return _re.sub(_re.escape(pat), _gsub_repl, recv_val)
                    except Exception:
                        return recv_val
                repl = str(args[1]) if len(args) > 1 else ''
                return recv_val.replace(pat, repl)
            if method == 'sub':
                args = [self._eval_node(a) for a in node.args]
                pat = str(args[0]) if args else ''
                if node.block:
                    import re as _re
                    def _sub_repl(m):
                        if node.block.params:
                            self._variables[node.block.params[0]] = m.group(0)
                        result = self._eval_body_last(node.block.body)
                        return str(result) if result is not None else ''
                    try:
                        return _re.sub(_re.escape(pat), _sub_repl, recv_val, count=1)
                    except Exception:
                        return recv_val
                repl = str(args[1]) if len(args) > 1 else ''
                return recv_val.replace(pat, repl, 1)
            if method == 'tr':
                args = [self._eval_node(a) for a in node.args]
                from_ = str(args[0]) if args else ''
                to_ = str(args[1]) if len(args) > 1 else ''
                table = str.maketrans(from_[:len(to_)], to_[:len(from_)]) if from_ and to_ else {}
                return recv_val.translate(table)
            if method == 'chars':
                return list(recv_val)
            if method == 'bytes':
                return list(recv_val.encode('utf-8'))
            if method == 'lines':
                return recv_val.splitlines(keepends=True)
            if method in ('size', 'count', 'len') and not node.args:
                return len(recv_val)
            if method == 'count' and node.args:
                args = [self._eval_node(a) for a in node.args]
                return recv_val.count(str(args[0]))
            if method == 'empty?':
                return len(recv_val) == 0
            if method == 'reverse':
                return recv_val[::-1]
            if method in ('freeze', 'frozen?', 'dup', 'clone'):
                return recv_val
            if method == 'replace':
                args = [self._eval_node(a) for a in node.args]
                return str(args[0]) if args else ''
            if method == 'capitalize':
                return recv_val.capitalize()
            if method == 'center':
                args = [self._eval_node(a) for a in node.args]
                width = int(args[0]) if args else 0
                pad = str(args[1]) if len(args) > 1 else ' '
                return recv_val.center(width, pad[0] if pad else ' ')
            if method == 'ljust':
                args = [self._eval_node(a) for a in node.args]
                width = int(args[0]) if args else 0
                return recv_val.ljust(width)
            if method == 'rjust':
                args = [self._eval_node(a) for a in node.args]
                width = int(args[0]) if args else 0
                return recv_val.rjust(width)
            if method == 'match?':
                args = [self._eval_node(a) for a in node.args]
                import re as _re
                pat = str(args[0]) if args else ''
                try:
                    return bool(_re.search(pat, recv_val))
                except Exception:
                    return False
            if method == 'index':
                args = [self._eval_node(a) for a in node.args]
                sub = str(args[0]) if args else ''
                idx = recv_val.find(sub)
                return idx if idx >= 0 else None
            if method == 'to_i':
                try:
                    return int(recv_val)
                except (ValueError, TypeError):
                    return 0
            if method == 'to_f':
                try:
                    return float(recv_val)
                except (ValueError, TypeError):
                    return 0.0
            if method == 'to_r':
                try:
                    return float(recv_val)
                except (ValueError, TypeError):
                    return 0.0

        # ── Additional string methods (chr/ord) ───────────────────────────
        if isinstance(recv_val, str):
            if method == 'ord':
                return ord(recv_val[0]) if recv_val else 0
            if method == 'hex':
                try:
                    return int(recv_val, 16)
                except Exception:
                    return 0
            if method == 'oct':
                try:
                    return int(recv_val, 8)
                except Exception:
                    return 0
            if method == 'succ' or method == 'next':
                if recv_val:
                    return recv_val[:-1] + chr(ord(recv_val[-1]) + 1)
                return recv_val
            if method == 'swapcase':
                return recv_val.swapcase()
            if method == 'squeeze':
                args = [self._eval_node(a) for a in node.args]
                import re as _re
                if args:
                    ch = str(args[0])
                    return _re.sub(f'[{_re.escape(ch)}]+', ch[0] if ch else '', recv_val)
                return _re.sub(r'(.)\1+', r'\1', recv_val)
            if method == 'scan':
                args = [self._eval_node(a) for a in node.args]
                import re as _re
                pat = str(args[0]) if args else ''
                try:
                    return _re.findall(pat, recv_val)
                except Exception:
                    return []
            if method == 'delete':
                args = [self._eval_node(a) for a in node.args]
                for a in args:
                    recv_val = recv_val.replace(str(a), '')
                return recv_val
            if method == 'format':
                args = [self._eval_node(a) for a in node.args]
                try:
                    return recv_val % (tuple(args) if len(args) > 1 else args[0]) if args else recv_val
                except Exception:
                    return recv_val
            if method == 'slice' or method == '[]':
                args_e = [self._eval_node(a) for a in node.args]
                if args_e:
                    try:
                        idx = int(args_e[0])
                        if len(args_e) > 1:
                            length = int(args_e[1])
                            return recv_val[idx:idx + length]
                        return recv_val[idx]
                    except (IndexError, TypeError):
                        return None
                return None

        # ── Additional numeric methods ─────────────────────────────────────
        if isinstance(recv_val, (int, float)):
            if method == 'chr' and isinstance(recv_val, int):
                try:
                    return chr(recv_val)
                except (ValueError, OverflowError):
                    return ''
            if method in ('succ', 'next') and isinstance(recv_val, int):
                return recv_val + 1
            if method == 'pred' and isinstance(recv_val, int):
                return recv_val - 1
            if method == 'pow':
                args = [self._eval_node(a) for a in node.args]
                exp = args[0] if args else 1
                mod = args[1] if len(args) > 1 else None
                try:
                    return pow(int(recv_val), int(exp), int(mod)) if mod is not None else recv_val ** exp
                except Exception:
                    return 0
            if method == 'gcd' and isinstance(recv_val, int):
                args = [self._eval_node(a) for a in node.args]
                other = int(args[0]) if args else 1
                import math as _math
                return _math.gcd(abs(recv_val), abs(other))
            if method == 'lcm' and isinstance(recv_val, int):
                args = [self._eval_node(a) for a in node.args]
                other = int(args[0]) if args else 1
                import math as _math
                g = _math.gcd(abs(recv_val), abs(other))
                return abs(recv_val * other) // g if g else 0
            if method == 'infinite?':
                import math as _math
                if isinstance(recv_val, float) and _math.isinf(recv_val):
                    return 1 if recv_val > 0 else -1
                return None
            if method in ('finite?',):
                import math as _math
                return isinstance(recv_val, float) and not _math.isinf(recv_val) and not _math.isnan(recv_val)
            if method in ('nan?',):
                import math as _math
                return isinstance(recv_val, float) and _math.isnan(recv_val)
            if method == 'digits':
                n = abs(int(recv_val))
                if n == 0:
                    return [0]
                digits = []
                while n > 0:
                    digits.append(n % 10)
                    n //= 10
                return digits
            if method in ('to_r',):
                return float(recv_val)
            if method in ('divmod',):
                args = [self._eval_node(a) for a in node.args]
                other = self._to_float(args[0]) if args else 1
                if other == 0:
                    return [0, 0]
                return [int(recv_val // other), recv_val % other]
            if method == 'coerce':
                args = [self._eval_node(a) for a in node.args]
                other = args[0] if args else 0
                return [float(other), float(recv_val)]

        return None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _eval_args(self, node: MethodCall) -> tuple[list, dict]:
        args = []
        for a in node.args:
            if isinstance(a, UnaryOp) and a.op == 'splat':
                val = self._eval_node(a.operand)
                if isinstance(val, list):
                    args.extend(val)
                elif val is not None:
                    args.append(val)
            else:
                args.append(self._eval_node(a))
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

        # Multiple positional args (from splat) → treat as chord list
        note_val = args[0] if len(args) == 1 else args

        # on: false → skip
        if not kwargs.get('on', True):
            return None

        # Merge defaults
        merged = dict(self._synth_defaults)
        merged.update(kwargs)

        # Apply pitch offset
        pitch = self._to_float(merged.pop('pitch', 0), 0.0)

        last_event = None

        def emit_one(n_raw):
            nonlocal last_event
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
            last_event = self._emit_synth(self._current_synth, a)

        if isinstance(note_val, list):
            for n in note_val:
                emit_one(n)
        else:
            emit_one(note_val)
        return last_event

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

        return self._emit_synth(synth_name, merged)

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
            except _BreakSignal:
                break
            except _NextSignal:
                continue
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
        # Increment and store the new index; look returns the same value
        idx = self._variables.get(f'__tick_{key}', -1) + 1
        self._variables[f'__tick_{key}'] = idx
        return idx

    def _call_look(self, node: MethodCall) -> int:
        args, kwargs = self._eval_args(node)
        key = str(args[0]).lstrip(':') if args else '_default'
        # Return current tick value without incrementing
        return self._variables.get(f'__tick_{key}', 0)

    def _call_stop(self, node: MethodCall):
        raise StopIteration_()

    def _call_next(self, node: MethodCall):
        raise _NextSignal()

    def _call_break(self, node: MethodCall):
        args, _ = self._eval_args(node)
        raise _BreakSignal(args[0] if args else None)

    def _call_control(self, node: MethodCall):
        """control node, param: val – emit a parameter-update event for a running synth node."""
        args, kwargs = self._eval_args(node)
        if not args or not isinstance(args[0], SoundEvent):
            return None
        target: SoundEvent = args[0]
        if not kwargs:
            return None
        evt = SoundEvent(
            time=self._time_secs,
            kind='control',
            synth_name=target.synth_name,
            node_id=target.node_id,
            args=dict(kwargs),
            bus_out=target.bus_out,
        )
        self.events.append(evt)
        return evt

    def _call_noop(self, node: MethodCall):
        return None

    def _call_with_block_noop(self, node: MethodCall):
        """Execute block unchanged (e.g. with_tuning)."""
        if node.block:
            self._eval_body(node.block.body)
        return None

    def _call_synth_names(self, node: MethodCall) -> list:
        return [
            'beep', 'sine', 'saw', 'pulse', 'square', 'tri', 'dsaw',
            'fm', 'mod_fm', 'mod_saw', 'mod_dsaw', 'mod_sine', 'mod_tri',
            'mod_pulse', 'supersaw', 'hoover', 'synth_piano', 'tb303',
            'prophet', 'piano', 'pluck', 'pretty_bell', 'kalimba',
            'blade', 'dark_ambience', 'growl', 'hollow', 'zawa',
            'noise', 'pnoise', 'bnoise', 'gnoise', 'cnoise',
            'subpulse', 'tech_saws', 'bass_foundation', 'bass_highend',
        ]

    def _call_loop(self, node: MethodCall):
        """loop do...end – unroll like live_loop."""
        if not node.block:
            return
        for _ in range(self._live_loop_iters):
            try:
                self._eval_body(node.block.body)
            except StopIteration_:
                break
            except _BreakSignal:
                break
            except _NextSignal:
                continue

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
        old_method = self._current_method
        self._current_method = func.name
        # Pass block as __block__ for yield support
        if node.block:
            self._variables['__block__'] = _Lambda(
                node.block.params, node.block.body, dict(self._variables))
        else:
            self._variables.pop('__block__', None)
        positional = 0  # count of positional args consumed
        for i, param in enumerate(func.params):
            if param.startswith('&'):
                # Explicit block parameter: &blk → assign lambda to blk
                blk_name = param[1:]
                if node.block:
                    self._variables[blk_name] = _Lambda(
                        node.block.params, node.block.body, dict(old_vars))
                else:
                    self._variables[blk_name] = None
            elif param.startswith('*'):
                # Splat: collect all remaining positional args as list
                self._variables[param[1:]] = list(args[positional:])
                positional = len(args)
                break
            else:
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
        self._current_method = old_method
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

    # ── String formatting ────────────────────────────────────────────────

    def _call_format(self, node: MethodCall) -> str:
        args, _ = self._eval_args(node)
        if not args:
            return ''
        fmt = str(args[0])
        rest = args[1:]
        try:
            return fmt % (tuple(rest) if len(rest) > 1 else rest[0]) if rest else fmt
        except Exception:
            return fmt

    # ── Standalone min / max / abs ───────────────────────────────────────

    def _call_min_standalone(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        if len(args) == 1 and isinstance(args[0], list):
            return min(args[0]) if args[0] else None
        if len(args) >= 2:
            try:
                return min(args)
            except TypeError:
                return args[0]
        return args[0] if args else None

    def _call_max_standalone(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        if len(args) == 1 and isinstance(args[0], list):
            return max(args[0]) if args[0] else None
        if len(args) >= 2:
            try:
                return max(args)
            except TypeError:
                return args[0]
        return args[0] if args else None

    def _call_abs_standalone(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        v = args[0] if args else 0
        try:
            return abs(v)
        except TypeError:
            return v

    # ── MIDI output ──────────────────────────────────────────────────────

    def _call_midi(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        if not args:
            return None
        note_raw = args[0]
        note_val = self._resolve_note(note_raw)
        if note_val is None:
            return None
        vel = self._to_float(kwargs.get('velocity', kwargs.get('vel_f', 1.0))) * 127
        channel = int(kwargs.get('channel', kwargs.get('chan', 1)))
        sustain = self._to_float(kwargs.get('sustain', 1.0))
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='midi',
            synth_name='midi',
            node_id=nid,
            args={
                'note': note_val,
                'velocity': max(0, min(127, int(vel))),
                'channel': channel,
                'sustain': sustain,
            },
            bus_out=0,
        )
        self.events.append(evt)
        return evt

    def _call_midi_note_on(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        note_raw = args[0] if args else 60
        note_val = self._resolve_note(note_raw)
        if note_val is None:
            return None
        vel_raw = args[1] if len(args) > 1 else kwargs.get('velocity', 127)
        vel = max(0, min(127, int(self._to_float(vel_raw))))
        channel = int(kwargs.get('channel', kwargs.get('chan', 1)))
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='midi_note_on',
            synth_name='midi',
            node_id=nid,
            args={'note': note_val, 'velocity': vel, 'channel': channel},
        )
        self.events.append(evt)
        return evt

    def _call_midi_note_off(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        note_raw = args[0] if args else 60
        note_val = self._resolve_note(note_raw)
        if note_val is None:
            return None
        channel = int(kwargs.get('channel', kwargs.get('chan', 1)))
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='midi_note_off',
            synth_name='midi',
            node_id=nid,
            args={'note': note_val, 'velocity': 0, 'channel': channel},
        )
        self.events.append(evt)
        return evt

    def _call_midi_cc(self, node: MethodCall) -> Any:
        args, kwargs = self._eval_args(node)
        cc_num = int(self._to_float(args[0] if args else 0))
        cc_val = int(self._to_float(args[1] if len(args) > 1 else kwargs.get('value', 0)))
        channel = int(kwargs.get('channel', kwargs.get('chan', 1)))
        nid = self._alloc_node()
        evt = SoundEvent(
            time=self._time_secs,
            kind='midi_cc',
            synth_name='midi',
            node_id=nid,
            args={'cc_num': cc_num, 'value': cc_val, 'channel': channel},
        )
        self.events.append(evt)
        return evt

    # ── sample_duration ──────────────────────────────────────────────────

    def _call_sample_duration(self, node: MethodCall) -> float:
        args, kwargs = self._eval_args(node)
        if not args:
            return 1.0
        sample_id = args[0]
        # Try to resolve to a path and get actual duration
        if isinstance(sample_id, str):
            path = self._sample_resolver.resolve(sample_id)
            if path:
                try:
                    import wave, contextlib
                    with contextlib.closing(wave.open(path, 'r')) as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        rate_mult = self._to_float(kwargs.get('rate', 1.0)) or 1.0
                        return duration / abs(rate_mult)
                except Exception:
                    pass
        # Default durations for known sample groups
        name = str(sample_id).lstrip(':').lower()
        _defaults = {
            'bd_': 0.5, 'sn_': 0.4, 'ht_': 0.2, 'hh_': 0.15,
            'bass_': 1.0, 'drum_': 0.5, 'loop_': 2.0, 'ambi_': 3.0,
        }
        for prefix, dur in _defaults.items():
            if name.startswith(prefix):
                return dur
        return 1.0

    # ── with_sample_pack / use_sample_pack ───────────────────────────────

    def _call_with_sample_pack(self, node: MethodCall) -> None:
        args, _ = self._eval_args(node)
        pack = str(args[0]) if args else ''
        if not node.block:
            self._sample_pack = pack
            return
        old = self._sample_pack
        self._sample_pack = pack
        try:
            self._eval_body(node.block.body)
        finally:
            self._sample_pack = old

    def _call_use_sample_pack(self, node: MethodCall) -> None:
        args, _ = self._eval_args(node)
        self._sample_pack = str(args[0]) if args else ''

    # ── use_tuning / with_tuning ─────────────────────────────────────────

    def _call_use_tuning(self, node: MethodCall) -> None:
        args, _ = self._eval_args(node)
        self._tuning = str(args[0]) if args else 'equal'

    def _call_with_tuning(self, node: MethodCall) -> None:
        args, _ = self._eval_args(node)
        old = self._tuning
        self._tuning = str(args[0]) if args else 'equal'
        if node.block:
            try:
                self._eval_body(node.block.body)
            finally:
                self._tuning = old

    # ── current_random_seed ──────────────────────────────────────────────

    def _call_current_random_seed(self, node: MethodCall) -> int:
        # Return deterministic value based on current RNG state
        return self._rng.randint(0, 2**31 - 1)

    # ── puts / print ─────────────────────────────────────────────────────

    def _call_puts(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        # Evaluate and return the last arg value (useful for puts in expressions)
        return args[-1] if args else None

    # ── run_code / eval ──────────────────────────────────────────────────

    def _call_run_code(self, node: MethodCall) -> Any:
        from .parser import parse as _parse
        args, _ = self._eval_args(node)
        if not args:
            return None
        src = str(args[0])
        try:
            prog = _parse(src)
            return self._eval_body_last(prog.statements)
        except Exception:
            return None

    # ── method(:foo) reference ───────────────────────────────────────────

    # ── Kernel conversion functions ──────────────────────────────────────

    def _call_kernel_Array(self, node: MethodCall) -> list:
        args, _ = self._eval_args(node)
        val = args[0] if args else None
        if val is None:
            return []
        if isinstance(val, list):
            return val
        return [val]

    def _call_kernel_Float(self, node: MethodCall) -> float:
        args, _ = self._eval_args(node)
        val = args[0] if args else 0
        return self._to_float(val)

    def _call_current_method(self, node: MethodCall) -> str:
        return self._current_method or ''

    def _call_method_ref(self, node: MethodCall) -> Any:
        args, _ = self._eval_args(node)
        func_name = str(args[0]).lstrip(':') if args else ''
        # Return a _Lambda that will dispatch to the named function when .call'd
        # We encode the name in the closure
        from .ast_nodes import MethodCall as MC, Identifier as ID
        body = [MC(None, func_name, [ID('__arg0__'), ID('__arg1__'),
                                      ID('__arg2__')], {}, None)]
        return _Lambda(['__arg0__', '__arg1__', '__arg2__'],
                       body, dict(self._variables))


# ---------------------------------------------------------------------------
# Euclidean rhythm (Björklund algorithm)
# ---------------------------------------------------------------------------

def _euclidean(hits: int, steps: int) -> list[bool]:
    """Euclidean/Bjorklund rhythm via Bresenham line algorithm."""
    if steps <= 0:
        return []
    if hits <= 0:
        return [False] * steps
    hits = min(hits, steps)
    b = 0
    pattern = []
    for _ in range(steps):
        b += hits
        if b >= steps:
            b -= steps
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern


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
