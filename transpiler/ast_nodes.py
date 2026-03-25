"""AST node definitions for the Sonic Pi → SuperCollider NRT transpiler."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Node:
    """Base class for all AST nodes."""
    pass


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

@dataclass
class IntLit(Node):
    value: int

@dataclass
class FloatLit(Node):
    value: float

@dataclass
class StringLit(Node):
    value: str

@dataclass
class SymbolLit(Node):
    """Represents a Ruby symbol like :beep or :C4."""
    value: str  # without the leading ':'

@dataclass
class BoolLit(Node):
    value: bool

@dataclass
class NilLit(Node):
    pass

@dataclass
class ArrayLit(Node):
    elements: list[Node]

@dataclass
class RangeLit(Node):
    """Represents a..b (inclusive) or a...b (exclusive)."""
    start: Node
    end: Node
    exclusive: bool = False


# ---------------------------------------------------------------------------
# Names & assignment
# ---------------------------------------------------------------------------

@dataclass
class Identifier(Node):
    name: str

@dataclass
class Assign(Node):
    name: str
    value: Node


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class BinOp(Node):
    op: str        # '+' '-' '*' '/' '%' '==' '!=' '<' '>' '<=' '>='
    left: Node
    right: Node

@dataclass
class UnaryOp(Node):
    op: str        # '-' 'not'
    operand: Node

@dataclass
class Block(Node):
    """A do...end or {...} block, optionally with block params (|x, y|)."""
    params: list[str]
    body: list[Node]

@dataclass
class MethodCall(Node):
    """
    Represents any method call:
      - free-standing:   play 60, amp: 0.8
      - chained:         8.times do ... end
      - receiver.method: ring.tick
    receiver=None means top-level call.
    """
    receiver: Optional[Node]
    method: str
    args: list[Node]
    kwargs: dict[str, Node]   # keyword args: note: 60  →  {'note': IntLit(60)}
    block: Optional[Block]


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

@dataclass
class IfStmt(Node):
    condition: Node
    then_body: list[Node]
    elsif_clauses: list[tuple[Node, list[Node]]]   # [(cond, body), ...]
    else_body: Optional[list[Node]]

@dataclass
class WhileStmt(Node):
    condition: Node
    body: list[Node]

@dataclass
class ReturnStmt(Node):
    value: Optional[Node]

@dataclass
class CaseStmt(Node):
    expr: Optional[Node]
    whens: list[tuple[Node, list[Node]]]   # [(value_expr, body), ...]
    else_body: Optional[list[Node]]

@dataclass
class FuncDef(Node):
    name: str
    params: list[str]       # plain names; splat params prefixed with '*'
    body: list[Node]
    defaults: dict = field(default_factory=dict)  # param_name → Node

@dataclass
class MultiAssign(Node):
    """Multiple assignment: a, b = x, y"""
    names: list[str]
    values: list[Node]

@dataclass
class TernaryExpr(Node):
    """Ternary: cond ? then_ : else_"""
    cond: Node
    then_: Node
    else_: Node

@dataclass
class HashLit(Node):
    """Hash literal: {key: val, ...}"""
    pairs: dict  # str → Node (kwarg-style keys only for now)


@dataclass
class StringInterp(Node):
    """Interpolated string: "Hello #{name}!" — parts is list of StringLit|Node."""
    parts: list


@dataclass
class BeginRescue(Node):
    """begin...rescue...ensure...end"""
    body: list
    rescue_clauses: list  # [(exc_type: str|None, var: str|None, body: list), ...]
    else_body: Optional[list]
    ensure_body: Optional[list]


@dataclass
class RaiseStmt(Node):
    value: Optional[Node]


@dataclass
class YieldExpr(Node):
    args: list


@dataclass
class ClassDef(Node):
    name: str
    superclass: Optional[str]
    body: list


@dataclass
class ModuleDef(Node):
    name: str
    body: list


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class Program(Node):
    statements: list[Node]
