"""
Tokenizer for the Sonic Pi DSL (Ruby subset).

Handles: literals, symbols, keyword args (foo:), identifiers, keywords,
operators, punctuation, comments, newlines.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token types."""
    # Literals
    INT       = auto()
    FLOAT     = auto()
    STRING    = auto()
    SYMBOL    = auto()      # :foo
    TRUE      = auto()
    FALSE     = auto()
    NIL       = auto()
    # Names
    IDENT     = auto()
    KWARG     = auto()      # foo:  (keyword argument key)
    # Block keywords
    DO        = auto()
    END       = auto()
    # Control-flow keywords
    IF        = auto()
    UNLESS    = auto()
    ELSE      = auto()
    ELSIF     = auto()
    THEN      = auto()
    WHILE     = auto()
    UNTIL     = auto()
    FOR       = auto()
    IN        = auto()
    DEF       = auto()
    RETURN    = auto()
    CASE      = auto()
    WHEN      = auto()
    # Boolean operators
    AND       = auto()
    OR        = auto()
    NOT       = auto()
    # Operators
    ASSIGN    = auto()      # =
    EQ        = auto()      # ==
    NEQ       = auto()      # !=
    LT        = auto()      # <
    GT        = auto()      # >
    LTE       = auto()      # <=
    GTE       = auto()      # >=
    PLUS      = auto()      # +
    MINUS     = auto()      # -
    STAR      = auto()      # *
    SLASH     = auto()      # /
    PERCENT   = auto()      # %
    STARSTAR  = auto()      # **
    DOTDOT    = auto()      # ..
    DOTDOTDOT = auto()      # ...  (exclusive range)
    # Compound assignment
    PLUS_ASSIGN  = auto()   # +=
    MINUS_ASSIGN = auto()   # -=
    STAR_ASSIGN  = auto()   # *=
    SLASH_ASSIGN = auto()   # /=
    OR_ASSIGN    = auto()   # ||=
    LSHIFT       = auto()   # <<  (append / left-shift)
    # Punctuation
    DOT       = auto()      # .
    COMMA     = auto()      # ,
    LPAREN    = auto()      # (
    RPAREN    = auto()      # )
    LBRACKET  = auto()      # [
    RBRACKET  = auto()      # ]
    LBRACE    = auto()      # {
    RBRACE    = auto()      # }
    PIPE      = auto()      # |
    AMPER_AMPER = auto()   # &&
    PIPE_PIPE   = auto()   # ||
    QUESTION  = auto()     # ?  (ternary)
    COLON     = auto()     # :  (ternary else, standalone)
    NEWLINE   = auto()
    EOF       = auto()


KEYWORDS: dict[str, TT] = {
    'do':     TT.DO,
    'end':    TT.END,
    'if':     TT.IF,
    'unless': TT.UNLESS,
    'else':   TT.ELSE,
    'elsif':  TT.ELSIF,
    'then':   TT.THEN,
    'while':  TT.WHILE,
    'until':  TT.UNTIL,
    'for':    TT.FOR,
    'in':     TT.IN,
    'def':    TT.DEF,
    'return': TT.RETURN,
    'case':   TT.CASE,
    'when':   TT.WHEN,
    'and':    TT.AND,
    'or':     TT.OR,
    'not':    TT.NOT,
    'true':   TT.TRUE,
    'false':  TT.FALSE,
    'nil':    TT.NIL,
}

# Tokens that can never be the last token before a newline that continues
# the expression — used to decide whether a NEWLINE is significant.
_CONTINUATION_TYPES = {
    TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH, TT.PERCENT, TT.STARSTAR,
    TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE, TT.LSHIFT,
    TT.COMMA, TT.DOT, TT.ASSIGN,
    TT.LPAREN, TT.LBRACKET, TT.LBRACE,
    TT.AND, TT.OR, TT.DO,
    TT.AMPER_AMPER, TT.PIPE_PIPE,
}


@dataclass
class Token:
    type: TT
    value: object   # int | float | str | None
    line: int
    no_space_before: bool = False  # True when token immediately follows previous (no whitespace)

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, ln{self.line})"


class TokenizeError(Exception):
    pass


def tokenize(source: str) -> list[Token]:
    """
    Convert Sonic Pi source code into a flat list of Tokens.
    Consecutive newlines are collapsed; newlines that follow a
    continuation token are suppressed.
    """
    tokens: list[Token] = []
    pos = 0
    line = 1
    length = len(source)
    _had_space = True  # treat start-of-file as "had space"

    def add(tt: TT, val: object):
        nonlocal _had_space
        tokens.append(Token(tt, val, line, no_space_before=not _had_space))
        _had_space = False

    def last_significant() -> Token | None:
        for t in reversed(tokens):
            if t.type != TT.NEWLINE:
                return t
        return None

    while pos < length:
        ch = source[pos]

        # ── Whitespace (horizontal only) ────────────────────────────────
        if ch in ' \t\r':
            _had_space = True
            pos += 1
            continue

        # ── Line continuation with backslash ────────────────────────────
        if ch == '\\' and pos + 1 < length and source[pos + 1] == '\n':
            _had_space = True
            line += 1
            pos += 2
            continue

        # ── Comment ─────────────────────────────────────────────────────
        if ch == '#':
            while pos < length and source[pos] != '\n':
                pos += 1
            continue

        # ── Newline / semicolon ──────────────────────────────────────────
        if ch == '\n' or ch == ';':
            if ch == '\n':
                line += 1
            pos += 1
            # Suppress if last significant token is a continuation
            ls = last_significant()
            if ls and ls.type in _CONTINUATION_TYPES:
                continue
            # Collapse consecutive newlines
            if not tokens or tokens[-1].type != TT.NEWLINE:
                add(TT.NEWLINE, '\n')
            _had_space = True
            continue

        # ── Three-char operators ─────────────────────────────────────────
        three = source[pos:pos + 3]
        if three == '||=':
            add(TT.OR_ASSIGN, '||='); pos += 3; continue
        if three == '...':
            add(TT.DOTDOTDOT, '...'); pos += 3; continue

        # ── Two-char operators ───────────────────────────────────────────
        two = source[pos:pos + 2]
        if two == '&&':
            add(TT.AMPER_AMPER, '&&'); pos += 2; continue
        if two == '||':
            add(TT.PIPE_PIPE, '||'); pos += 2; continue
        if two == '**':
            add(TT.STARSTAR, '**'); pos += 2; continue
        if two == '==':
            add(TT.EQ, '=='); pos += 2; continue
        if two == '!=':
            add(TT.NEQ, '!='); pos += 2; continue
        if two == '<=':
            add(TT.LTE, '<='); pos += 2; continue
        if two == '>=':
            add(TT.GTE, '>='); pos += 2; continue
        if two == '..':
            add(TT.DOTDOT, '..'); pos += 2; continue
        if two == '<<':
            add(TT.LSHIFT, '<<'); pos += 2; continue
        if two == '+=':
            add(TT.PLUS_ASSIGN, '+='); pos += 2; continue
        if two == '-=':
            add(TT.MINUS_ASSIGN, '-='); pos += 2; continue
        if two == '*=':
            add(TT.STAR_ASSIGN, '*='); pos += 2; continue
        if two == '/=':
            add(TT.SLASH_ASSIGN, '/='); pos += 2; continue

        # ── Number literal ───────────────────────────────────────────────
        if ch.isdigit() or (ch == '-' and pos + 1 < length and source[pos + 1].isdigit()
                            and (not tokens or last_significant() is None
                                 or last_significant().type in _CONTINUATION_TYPES
                                 or last_significant().type in {TT.COMMA, TT.LPAREN, TT.LBRACKET, TT.ASSIGN})):
            # Negative literal only when unambiguously a literal, not subtraction
            m = re.match(r'-?\d+\.\d+', source[pos:])
            if not m:
                m = re.match(r'-?\d+', source[pos:])
            if m:
                raw = m.group()
                if '.' in raw:
                    add(TT.FLOAT, float(raw))
                else:
                    add(TT.INT, int(raw))
                pos += len(raw)
                continue

        if ch.isdigit():
            m = re.match(r'\d+\.\d+', source[pos:])
            if not m:
                m = re.match(r'\d+', source[pos:])
            raw = m.group()
            if '.' in raw:
                add(TT.FLOAT, float(raw))
            else:
                add(TT.INT, int(raw))
            pos += len(raw)
            continue

        # ── String literal ───────────────────────────────────────────────
        if ch == '"':
            pos += 1
            buf = []
            while pos < length and source[pos] != '"':
                if source[pos] == '\\' and pos + 1 < length:
                    esc = source[pos + 1]
                    buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"'}.get(esc, esc))
                    pos += 2
                elif source[pos] == '#' and pos + 1 < length and source[pos + 1] == '{':
                    # String interpolation #{expr} – skip to matching }
                    pos += 2
                    depth = 1
                    while pos < length and depth > 0:
                        if source[pos] == '{':
                            depth += 1
                        elif source[pos] == '}':
                            depth -= 1
                        pos += 1
                    # interpolated content discarded for now; leave placeholder
                else:
                    buf.append(source[pos])
                    pos += 1
            pos += 1  # closing "
            add(TT.STRING, ''.join(buf))
            continue

        if ch == "'":
            pos += 1
            buf = []
            while pos < length and source[pos] != "'":
                if source[pos] == '\\' and pos + 1 < length and source[pos + 1] == "'":
                    buf.append("'")
                    pos += 2
                else:
                    buf.append(source[pos])
                    pos += 1
            pos += 1
            add(TT.STRING, ''.join(buf))
            continue

        # ── Standalone ?  (ternary condition marker) ─────────────────────
        if ch == '?':
            add(TT.QUESTION, '?'); pos += 1; continue

        # ── Symbol  :foo  or standalone colon ────────────────────────────
        if ch == ':':
            if pos + 1 < length and (source[pos + 1].isalpha() or source[pos + 1] == '_'):
                m = re.match(r':[a-zA-Z_][a-zA-Z0-9_?!]*', source[pos:])
                add(TT.SYMBOL, m.group()[1:])
                pos += len(m.group())
            elif pos + 1 < length and source[pos + 1] in '+-*/%<>=!':
                # Operator symbols: :+  :-  :*  :/  :==  :!=  :<=  :>=  :<  :>
                m = re.match(r':[+\-*/%<>=!]+', source[pos:])
                add(TT.SYMBOL, m.group()[1:])
                pos += len(m.group())
            else:
                add(TT.COLON, ':')
                pos += 1
            continue

        # ── Identifier, keyword, or keyword-arg  foo: ────────────────────
        if ch.isalpha() or ch == '_':
            m = re.match(r'[a-zA-Z_][a-zA-Z0-9_?!]*', source[pos:])
            word = m.group()
            pos += len(word)
            # keyword arg:  word followed by ':' but not '::'
            if (pos < length and source[pos] == ':'
                    and (pos + 1 >= length or source[pos + 1] != ':')):
                add(TT.KWARG, word)
                pos += 1   # consume ':'
                continue
            kw = KEYWORDS.get(word)
            if kw is not None:
                add(kw, word)
            else:
                add(TT.IDENT, word)
            continue

        # ── %w[...] word array  /  %i[...] symbol array ─────────────────
        if ch == '%' and pos + 1 < length and source[pos + 1] in 'wWiI':
            sym_mode = source[pos + 1] in 'iI'
            opener = source[pos + 2] if pos + 2 < length else '['
            closer = {'{': '}', '(': ')', '[': ']', '<': '>'}.get(opener, opener)
            pos += 3  # skip  %w[
            words: list[str] = []
            buf: list[str] = []
            while pos < length and source[pos] != closer:
                if source[pos] in ' \t\n':
                    if buf:
                        words.append(''.join(buf))
                        buf = []
                    if source[pos] == '\n':
                        line += 1
                else:
                    buf.append(source[pos])
                pos += 1
            if buf:
                words.append(''.join(buf))
            if pos < length:
                pos += 1  # skip closer
            add(TT.LBRACKET, '[')
            for wi, word in enumerate(words):
                if wi > 0:
                    add(TT.COMMA, ',')
                add(TT.SYMBOL if sym_mode else TT.STRING, word)
            add(TT.RBRACKET, ']')
            continue

        # ── Single-char tokens ───────────────────────────────────────────
        SINGLE = {
            '=': TT.ASSIGN,
            '<': TT.LT,
            '>': TT.GT,
            '+': TT.PLUS,
            '-': TT.MINUS,
            '*': TT.STAR,
            '/': TT.SLASH,
            '%': TT.PERCENT,
            '.': TT.DOT,
            ',': TT.COMMA,
            '(': TT.LPAREN,
            ')': TT.RPAREN,
            '[': TT.LBRACKET,
            ']': TT.RBRACKET,
            '{': TT.LBRACE,
            '}': TT.RBRACE,
            '|': TT.PIPE,
        }
        if ch in SINGLE:
            add(SINGLE[ch], ch)
            pos += 1
            continue

        # ── Unknown character — skip silently ────────────────────────────
        pos += 1

    add(TT.EOF, None)
    return tokens
