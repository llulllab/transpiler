"""
Recursive-descent parser for the Sonic Pi DSL (Ruby subset).

Produces an AST (see ast_nodes.py) from a flat list of Token objects
produced by tokenizer.py.
"""
from __future__ import annotations
from typing import Optional

from .tokenizer import Token, TT, tokenize, KEYWORDS
from .ast_nodes import (
    Node, Program, Block,
    IntLit, FloatLit, StringLit, SymbolLit, BoolLit, NilLit, ArrayLit, RangeLit,
    Identifier, Assign, BinOp, UnaryOp, MethodCall,
    IfStmt, WhileStmt, ReturnStmt, CaseStmt, FuncDef,
    MultiAssign, TernaryExpr, HashLit,
    StringInterp, BeginRescue, RaiseStmt, YieldExpr, ClassDef, ModuleDef,
)


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tokens that can start a value / argument
# ---------------------------------------------------------------------------
_VALUE_START = {
    TT.INT, TT.FLOAT, TT.STRING, TT.SYMBOL,
    TT.TRUE, TT.FALSE, TT.NIL,
    TT.IDENT, TT.LBRACKET, TT.LPAREN,
    TT.MINUS, TT.NOT,
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ── Token navigation ────────────────────────────────────────────────

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        p = self.pos + offset
        if p < len(self.tokens):
            return self.tokens[p]
        return Token(TT.EOF, None, 0)

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def expect(self, tt: TT) -> Token:
        t = self.cur
        if t.type != tt:
            raise ParseError(
                f"Line {t.line}: expected {tt.name}, got {t.type.name} ({t.value!r})"
            )
        return self.advance()

    def match(self, *types: TT) -> bool:
        return self.cur.type in types

    def skip_newlines(self):
        while self.cur.type == TT.NEWLINE:
            self.advance()

    # ── Entry point ─────────────────────────────────────────────────────

    def parse(self) -> Program:
        stmts: list[Node] = []
        self.skip_newlines()
        while not self.match(TT.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
            # consume trailing newline(s)
            while self.match(TT.NEWLINE):
                self.advance()
            self.skip_newlines()
        return Program(stmts)

    # ── Statement ───────────────────────────────────────────────────────

    def _parse_statement(self) -> Optional[Node]:
        # Return statement
        if self.match(TT.RETURN):
            self.advance()
            val = None
            if not self.match(TT.NEWLINE, TT.EOF, TT.END, TT.ELSE, TT.ELSIF):
                val = self._parse_expr()
            # Fix: `return val if cond` → IfStmt(cond, [ReturnStmt(val)], [], None)
            # Without this, ReturnStmt(IfStmt(...)) raises _ReturnSignal(None) when cond is false
            if (isinstance(val, IfStmt) and len(val.then_body) == 1
                    and not val.elsif_clauses and val.else_body is None):
                return IfStmt(val.condition, [ReturnStmt(val.then_body[0])], [], None)
            return ReturnStmt(val)

        # if / unless / while / case / def as statements
        if self.match(TT.IF):
            return self._parse_if()
        if self.match(TT.UNLESS):
            return self._parse_unless()
        if self.match(TT.WHILE):
            return self._parse_while()
        if self.match(TT.UNTIL):
            return self._parse_until()
        if self.match(TT.FOR):
            return self._parse_for()
        if self.match(TT.CASE):
            return self._parse_case()
        if self.match(TT.DEF):
            return self._parse_def()

        # Multiple assignment:  a, b = x, y   or   a, b, *rest = arr
        if self.cur.type in (TT.IDENT, TT.STAR):
            # Look ahead for  (IDENT|*IDENT) COMMA ... ASSIGN
            scan = self.pos
            names: list[str] = []
            while scan < len(self.tokens):
                tok = self.tokens[scan]
                if tok.type == TT.STAR:
                    # *rest splat
                    scan += 1
                    if scan < len(self.tokens) and self.tokens[scan].type == TT.IDENT:
                        names.append('*' + self.tokens[scan].value)
                        scan += 1
                    else:
                        names.append('*')
                elif tok.type == TT.IDENT:
                    names.append(tok.value)
                    scan += 1
                else:
                    break
                if (scan < len(self.tokens)
                        and self.tokens[scan].type == TT.COMMA):
                    scan += 1  # skip comma
                else:
                    break
            if (len(names) > 1
                    and scan < len(self.tokens)
                    and self.tokens[scan].type == TT.ASSIGN
                    and (scan + 1 >= len(self.tokens)
                         or self.tokens[scan + 1].type != TT.ASSIGN)):
                # consume all the names, stars, and commas we scanned
                while self.pos < scan:
                    self.advance()
                self.advance()              # ASSIGN
                values: list[Node] = []
                values.append(self._parse_expr())
                while self.match(TT.COMMA):
                    self.advance()
                    values.append(self._parse_expr())
                return MultiAssign(names, values)

        # Compound assignment:  ident += / -= / *= / /= expr
        _COMPOUND = {
            TT.PLUS_ASSIGN:  '+',
            TT.MINUS_ASSIGN: '-',
            TT.STAR_ASSIGN:  '*',
            TT.SLASH_ASSIGN: '/',
        }
        if self.cur.type == TT.IDENT and self.peek().type in _COMPOUND:
            name = self.advance().value
            op = _COMPOUND[self.advance().type]
            val = self._parse_expr()
            return Assign(name, BinOp(op, Identifier(name), val))

        # Quick path for bare  ident = expr  and  ident ||= expr
        if self.cur.type == TT.IDENT:
            if self.peek().type == TT.OR_ASSIGN:
                name = self.advance().value
                self.advance()  # consume ||=
                val = self._parse_expr()
                return Assign(name, BinOp('or', Identifier(name), val))
            if (self.peek().type == TT.ASSIGN
                    and self.peek(2).type != TT.ASSIGN):
                name = self.advance().value
                self.advance()  # consume '='
                val = self._parse_expr()
                return Assign(name, val)

        # General expression — may be followed by compound assignment
        node = self._parse_expr()

        # Subscript / attr compound assignment:  expr[k] ||= v  /  expr[k] += v
        if self.match(TT.OR_ASSIGN):
            self.advance()
            rhs = self._parse_expr()
            if isinstance(node, Identifier):
                return Assign(node.name, BinOp('or', node, rhs))
            if isinstance(node, MethodCall) and node.method == '[]':
                return MethodCall(node.receiver, '[]=',
                                  node.args + [BinOp('or', node, rhs)], {}, None)
        elif self.match(TT.PLUS_ASSIGN):
            self.advance()
            rhs = self._parse_expr()
            if isinstance(node, Identifier):
                return Assign(node.name, BinOp('+', node, rhs))
            if isinstance(node, MethodCall) and node.method == '[]':
                return MethodCall(node.receiver, '[]=',
                                  node.args + [BinOp('+', node, rhs)], {}, None)
        elif self.match(TT.MINUS_ASSIGN):
            self.advance()
            rhs = self._parse_expr()
            if isinstance(node, Identifier):
                return Assign(node.name, BinOp('-', node, rhs))
            if isinstance(node, MethodCall) and node.method == '[]':
                return MethodCall(node.receiver, '[]=',
                                  node.args + [BinOp('-', node, rhs)], {}, None)

        return node

    # ── Expressions ─────────────────────────────────────────────────────

    def _parse_expr(self) -> Node:
        return self._parse_ternary()

    def _parse_ternary(self) -> Node:
        cond = self._parse_range()
        if self.match(TT.QUESTION):
            self.advance()
            then_ = self._parse_ternary()
            if self.match(TT.COLON):
                self.advance()
            else_ = self._parse_ternary()
            return TernaryExpr(cond, then_, else_)
        return cond

    def _parse_range(self) -> Node:
        left = self._parse_or()
        if self.match(TT.DOTDOTDOT):
            self.advance()
            right = self._parse_or()
            return RangeLit(left, right, exclusive=True)
        if self.match(TT.DOTDOT):
            self.advance()
            right = self._parse_or()
            return RangeLit(left, right, exclusive=False)
        return left

    def _parse_or(self) -> Node:
        left = self._parse_and()
        while self.match(TT.OR, TT.PIPE_PIPE):
            self.advance()
            right = self._parse_and()
            left = BinOp('or', left, right)
        return left

    def _parse_and(self) -> Node:
        left = self._parse_not()
        while self.match(TT.AND, TT.AMPER_AMPER):
            self.advance()
            right = self._parse_not()
            left = BinOp('and', left, right)
        return left

    def _parse_not(self) -> Node:
        if self.match(TT.NOT):
            self.advance()
            return UnaryOp('not', self._parse_comparison())
        return self._parse_comparison()

    def _parse_comparison(self) -> Node:
        left = self._parse_bitwise()
        while self.match(TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE):
            op = self.advance().value
            right = self._parse_bitwise()
            left = BinOp(op, left, right)
        return left

    def _parse_bitwise(self) -> Node:
        """Handles &  |  bitwise / array-intersection / array-union operators."""
        left = self._parse_additive()
        while self.match(TT.AMPER, TT.PIPE):
            op_tok = self.advance()
            op = op_tok.value  # '&' or '|'
            right = self._parse_additive()
            left = BinOp(op, left, right)
        return left

    def _parse_additive(self) -> Node:
        left = self._parse_multiplicative()
        while self.match(TT.PLUS, TT.MINUS, TT.LSHIFT):
            op = self.advance().value
            right = self._parse_multiplicative()
            left = BinOp(op, left, right)
        return left

    def _parse_multiplicative(self) -> Node:
        left = self._parse_power()
        while self.match(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance().value
            right = self._parse_power()
            left = BinOp(op, left, right)
        return left

    def _parse_power(self) -> Node:
        base = self._parse_unary()
        if self.match(TT.STARSTAR):
            self.advance()
            exp = self._parse_unary()
            return BinOp('**', base, exp)
        return base

    def _parse_unary(self) -> Node:
        if self.match(TT.MINUS):
            self.advance()
            return UnaryOp('-', self._parse_postfix())
        if self.match(TT.NOT):
            self.advance()
            return UnaryOp('not', self._parse_postfix())
        return self._parse_postfix()

    # ── Postfix (method chains, indexing) ───────────────────────────────

    def _parse_postfix(self) -> Node:
        node = self._parse_primary()

        while True:
            # .method_name ...
            if self.match(TT.DOT, TT.SCOPE):
                self.advance()
                # .( args ) shorthand for .call( args )
                if self.match(TT.LPAREN):
                    self.advance()
                    self.skip_newlines()
                    args, kwargs = self._parse_arg_list_parens()
                    self.expect(TT.RPAREN)
                    block = self._parse_block_opt()
                    node = MethodCall(node, 'call', args, kwargs, block)
                elif self.cur.type == TT.IDENT or (
                        self.cur.value is not None and self.cur.type not in (
                            TT.NEWLINE, TT.EOF, TT.LBRACE, TT.LPAREN,
                            TT.RBRACE, TT.RPAREN, TT.LBRACKET, TT.RBRACKET,
                            TT.COMMA, TT.ASSIGN, TT.PIPE,
                        )):
                    method = self.advance().value
                    args, kwargs, block = self._parse_call_tail()
                    node = MethodCall(node, method, args, kwargs, block)
                else:
                    break

            # [index]  or  [index] = val  (subscript assignment)
            elif self.match(TT.LBRACKET):
                self.advance()
                idx = self._parse_expr()
                self.expect(TT.RBRACKET)
                # Subscript assignment: obj[k] = v  →  obj.[]=(k, v)
                if self.match(TT.ASSIGN) and self.peek().type != TT.ASSIGN:
                    self.advance()  # consume =
                    val = self._parse_expr()
                    node = MethodCall(node, '[]=', [idx, val], {}, None)
                else:
                    node = MethodCall(node, '[]', [idx], {}, None)

            else:
                break

        # Trailing modifier:  expr if cond  /  expr unless cond
        if self.match(TT.IF):
            self.advance()
            cond = self._parse_expr()
            return IfStmt(cond, [node], [], None)
        if self.match(TT.UNLESS):
            self.advance()
            cond = self._parse_expr()
            return IfStmt(UnaryOp('not', cond), [node], [], None)
        if self.match(TT.UNTIL):
            self.advance()
            cond = self._parse_expr()
            return WhileStmt(UnaryOp('not', cond), [node])
        if self.match(TT.WHILE):
            self.advance()
            cond = self._parse_expr()
            return WhileStmt(cond, [node])

        return node

    # ── Primary ─────────────────────────────────────────────────────────

    def _parse_primary(self) -> Node:
        t = self.cur

        if t.type == TT.INT:
            self.advance(); return IntLit(t.value)

        if t.type == TT.FLOAT:
            self.advance(); return FloatLit(t.value)

        if t.type == TT.STRING:
            self.advance()
            val = t.value
            if isinstance(val, list):
                # Interpolated string — sub-parse each #{expr} part
                parts = []
                for kind, content in val:
                    if kind == 'lit':
                        parts.append(StringLit(content))
                    else:
                        try:
                            sub_tokens = tokenize(content)
                            sub_node = Parser(sub_tokens)._parse_expr()
                            parts.append(sub_node)
                        except Exception:
                            parts.append(StringLit(''))
                return StringInterp(parts)
            return StringLit(val)

        if t.type == TT.SYMBOL:
            self.advance(); return SymbolLit(t.value)

        if t.type == TT.TRUE:
            self.advance(); return BoolLit(True)

        if t.type == TT.FALSE:
            self.advance(); return BoolLit(False)

        if t.type == TT.NIL:
            self.advance(); return NilLit()

        if t.type == TT.LBRACKET:
            return self._parse_array()

        if t.type == TT.LBRACE:
            # Could be a hash literal {key: val, ...} if followed by KWARG or RBRACE
            if self.peek().type in (TT.KWARG, TT.RBRACE):
                return self._parse_hash_lit()

        if t.type == TT.LPAREN:
            self.advance()
            self.skip_newlines()
            expr = self._parse_expr()
            self.skip_newlines()
            self.expect(TT.RPAREN)
            return expr

        # Arrow lambda:  ->(x) { body }  or  -> { body }  or  -> (x) { body }
        if t.type == TT.ARROW:
            self.advance()
            arrow_params: list[str] = []
            if self.match(TT.LPAREN):
                self.advance()
                while not self.match(TT.RPAREN, TT.EOF):
                    if self.match(TT.STAR):
                        self.advance()
                        if self.match(TT.IDENT):
                            arrow_params.append('*' + self.advance().value)
                    elif self.match(TT.AMPER):
                        self.advance()
                        if self.match(TT.IDENT):
                            arrow_params.append('&' + self.advance().value)
                    elif self.match(TT.IDENT):
                        p = self.advance().value
                        arrow_params.append(p)
                        if self.match(TT.ASSIGN):
                            self.advance()
                            self._parse_arg_val()  # default — ignored for now
                    if self.match(TT.COMMA):
                        self.advance()
                    elif not self.match(TT.RPAREN):
                        break
                self.expect(TT.RPAREN)
            # Parse body as { } or do...end, but inject pre-parsed params
            if self.match(TT.LBRACE):
                self.advance()
                # absorb any |params| if user also wrote them
                if self.match(TT.PIPE):
                    arrow_params = self._parse_block_params()
                self.skip_newlines()
                body = self._parse_body_until(TT.RBRACE)
                if self.match(TT.RBRACE):
                    self.advance()
                blk = Block(arrow_params, body)
            elif self.match(TT.DO):
                self.advance()
                if self.match(TT.PIPE):
                    arrow_params = self._parse_block_params()
                self.skip_newlines()
                body = self._parse_body_until(TT.END)
                if self.match(TT.END):
                    self.advance()
                blk = Block(arrow_params, body)
            else:
                return NilLit()
            return MethodCall(None, 'lambda', [], {}, blk)

        if t.type == TT.IF:
            return self._parse_if()

        if t.type == TT.UNLESS:
            return self._parse_unless()

        if t.type == TT.CASE:
            return self._parse_case()

        if t.type == TT.IDENT:
            return self._parse_ident_or_call()

        # Fallback – skip and return nil
        self.advance()
        return NilLit()

    # ── Identifier / free-standing method call ──────────────────────────

    def _parse_ident_or_call(self) -> Node:
        name = self.advance().value  # consume IDENT

        # begin ... rescue ... ensure ... end → BeginRescue
        if name == 'begin':
            self.skip_newlines()
            body = self._parse_body_until(TT.END, TT.RESCUE, TT.ENSURE, TT.ELSE)
            rescue_clauses = []
            else_body = None
            ensure_body = None
            while self.match(TT.RESCUE):
                self.advance()
                exc_type = None
                exc_var = None
                # optional 'ExcType => var' or 'ExcType'
                if (self.match(TT.IDENT)
                        and self.cur.value and self.cur.value[0].isupper()
                        and not self.match(TT.NEWLINE)):
                    exc_type = self.advance().value
                    # => var  (tokenised as ASSIGN GT IDENT)
                    if self.match(TT.ASSIGN) and self.peek().type == TT.GT:
                        self.advance()   # =
                        self.advance()   # >
                        if self.match(TT.IDENT):
                            exc_var = self.advance().value
                self.skip_newlines()
                rc_body = self._parse_body_until(TT.END, TT.RESCUE, TT.ENSURE, TT.ELSE)
                rescue_clauses.append((exc_type, exc_var, rc_body))
            if self.match(TT.ELSE):
                self.advance()
                self.skip_newlines()
                else_body = self._parse_body_until(TT.END, TT.ENSURE)
            if self.match(TT.ENSURE):
                self.advance()
                self.skip_newlines()
                ensure_body = self._parse_body_until(TT.END)
            if self.match(TT.END):
                self.advance()
            return BeginRescue(body, rescue_clauses, else_body, ensure_body)

        # raise expr  /  raise
        if name == 'raise':
            if self.match(TT.NEWLINE, TT.EOF, TT.END, TT.RESCUE, TT.ENSURE):
                return RaiseStmt(None)
            val = self._parse_expr()
            return RaiseStmt(val)

        # yield args  /  yield
        if name == 'yield':
            args: list[Node] = []
            if not self.match(TT.NEWLINE, TT.EOF, TT.END, TT.ELSE, TT.ELSIF,
                              TT.RPAREN, TT.RBRACKET, TT.THEN):
                args, _ = self._parse_arg_list_implicit()
            return YieldExpr(args)

        # class Foo [< Bar] ... end  /  class << self ... end
        if name == 'class':
            if self.match(TT.LSHIFT):
                # Singleton class  class << self
                self.advance()
                if self.match(TT.IDENT):
                    self.advance()  # skip receiver
                self.skip_newlines()
                body = self._parse_body_until(TT.END)
                if self.match(TT.END):
                    self.advance()
                return IfStmt(BoolLit(True), body, [], None)
            class_name = self.advance().value if self.match(TT.IDENT) else 'Unknown'
            superclass = None
            if self.match(TT.LT):
                self.advance()
                if self.match(TT.IDENT):
                    superclass = self.advance().value
                    # Foo::Bar style
                    while self.match(TT.SCOPE) and self.peek().type == TT.IDENT:
                        self.advance()
                        superclass += '::' + self.advance().value
            self.skip_newlines()
            body = self._parse_body_until(TT.END)
            if self.match(TT.END):
                self.advance()
            return ClassDef(class_name, superclass, body)

        # module Bar ... end
        if name == 'module':
            mod_name = self.advance().value if self.match(TT.IDENT) else 'Unknown'
            self.skip_newlines()
            body = self._parse_body_until(TT.END)
            if self.match(TT.END):
                self.advance()
            return ModuleDef(mod_name, body)

        # self  (bare or as receiver before dot — handled by postfix)
        if name == 'self':
            return Identifier('__self__')

        # Explicit parens:  method(args)
        if self.match(TT.LPAREN):
            self.advance()
            self.skip_newlines()
            args, kwargs = self._parse_arg_list_parens()
            self.expect(TT.RPAREN)
            block = self._parse_block_opt()
            return MethodCall(None, name, args, kwargs, block)

        # Implicit args (no parens) – only if next token can start args.
        # If the next token is '[' with no space before it (e.g. notes[0]),
        # treat it as subscript access, not as an array argument.
        if self._can_start_args() and not (
            self.cur.type == TT.LBRACKET and self.cur.no_space_before
        ):
            args, kwargs = self._parse_arg_list_implicit()
            block = self._parse_block_opt()
            # If nothing was parsed AND no block, it's just an identifier
            if not args and not kwargs and block is None:
                return Identifier(name)
            return MethodCall(None, name, args, kwargs, block)

        # Bare block after identifier:  foo do...end
        if self.match(TT.DO, TT.LBRACE):
            block = self._parse_block_opt()
            return MethodCall(None, name, [], {}, block)

        return Identifier(name)

    def _can_start_args(self) -> bool:
        """True if the current token can start an implicit argument list."""
        t = self.cur
        # These tokens never start args in implicit position
        # MINUS can start args if it's a unary negation (next token has no space,
        # e.g.  foo -12  is foo(-12), but  foo - 12  is subtraction)
        if t.type == TT.MINUS:
            # Unary negation  foo -12  → space before '-', no space before digit
            # Binary subtract foo - 12 → space before '-', space before digit
            # Subtraction     n-1      → no space before '-' (attached to lhs)
            nxt = self.peek(1)  # token after the MINUS
            return (not t.no_space_before) and nxt.no_space_before and nxt.type in (TT.INT, TT.FLOAT)
        if t.type == TT.STAR:
            # Splat: *arr in implicit args – only when followed by a value-starting token
            # e.g.  play *chord(:C, :major)   → splat, starts args
            # e.g.  a * b                     → multiplication (won't be called here)
            nxt = self.peek(1)
            return nxt.type in (TT.IDENT, TT.LPAREN, TT.LBRACKET)
        if t.type in (TT.NEWLINE, TT.EOF, TT.END, TT.ELSE, TT.ELSIF,
                      TT.RPAREN, TT.RBRACKET, TT.RBRACE,
                      TT.DOT, TT.ASSIGN, TT.DO, TT.LBRACE,
                      TT.PLUS, TT.SLASH, TT.PERCENT,
                      TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE,
                      TT.AND, TT.OR, TT.STARSTAR, TT.PIPE,
                      TT.DOTDOT, TT.COMMA, TT.LSHIFT,
                      TT.AMPER_AMPER, TT.PIPE_PIPE, TT.QUESTION, TT.COLON):
            return False
        if t.type == TT.KWARG:
            return True
        return t.type in _VALUE_START

    # ── Argument lists ───────────────────────────────────────────────────

    def _parse_arg_list_parens(self) -> tuple[list[Node], dict[str, Node]]:
        """Comma-separated args inside explicit parens."""
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        while not self.match(TT.RPAREN, TT.EOF):
            self.skip_newlines()
            if self.match(TT.KWARG):
                key = self.advance().value
                val = self._parse_arg_val()
                kwargs[key] = val
            elif self.match(TT.STAR):
                # Splat: *arr → UnaryOp('splat', arr)
                self.advance()
                args.append(UnaryOp('splat', self._parse_arg_val()))
            elif not self.match(TT.RPAREN, TT.EOF):
                args.append(self._parse_arg_val())
            self.skip_newlines()
            if self.match(TT.COMMA):
                self.advance()
                self.skip_newlines()
        return args, kwargs

    def _parse_arg_list_implicit(self) -> tuple[list[Node], dict[str, Node]]:
        """Comma-separated args without enclosing parens, stopping at NEWLINE / DO / END."""
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        stop = {TT.NEWLINE, TT.EOF, TT.DO, TT.END, TT.ELSE, TT.ELSIF,
                TT.RPAREN, TT.RBRACKET, TT.RBRACE, TT.LBRACE}
        while not self.match(*stop):
            if self.match(TT.KWARG):
                key = self.advance().value
                val = self._parse_arg_val()
                kwargs[key] = val
            elif self.match(TT.STAR):
                self.advance()
                args.append(UnaryOp('splat', self._parse_arg_val()))
            else:
                args.append(self._parse_arg_val())
            if self.match(TT.COMMA):
                self.advance()
                self.skip_newlines()  # allow args to span lines after comma
            elif self.match(*stop):
                break
            else:
                break
        return args, kwargs

    def _parse_arg_val(self) -> Node:
        """Single argument value – delegates to ternary level (above additive, below comma)."""
        return self._parse_ternary()

    # ── Call tail (args for chained call after dot) ──────────────────────

    def _parse_call_tail(self) -> tuple[list[Node], dict[str, Node], Optional[Block]]:
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        block: Optional[Block] = None

        if self.match(TT.LPAREN):
            self.advance()
            self.skip_newlines()
            args, kwargs = self._parse_arg_list_parens()
            self.expect(TT.RPAREN)
        elif self._can_start_args() and not (
            self.cur.type == TT.LBRACKET and self.cur.no_space_before
        ):
            args, kwargs = self._parse_arg_list_implicit()

        block = self._parse_block_opt()
        return args, kwargs, block

    # ── Blocks ───────────────────────────────────────────────────────────

    def _parse_block_opt(self) -> Optional[Block]:
        if self.match(TT.DO):
            return self._parse_do_end_block()
        if self.match(TT.LBRACE):
            return self._parse_brace_block()
        return None

    def _parse_block_params(self) -> list[str]:
        params: list[str] = []
        if self.match(TT.PIPE):
            self.advance()
            while not self.match(TT.PIPE, TT.EOF):
                if self.match(TT.IDENT):
                    params.append(self.advance().value)
                elif self.match(TT.COMMA):
                    self.advance()
                else:
                    self.advance()  # skip unexpected
            if self.match(TT.PIPE):
                self.advance()
        return params

    def _parse_do_end_block(self) -> Block:
        self.expect(TT.DO)
        params = self._parse_block_params()
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return Block(params, body)

    def _parse_brace_block(self) -> Block:
        self.expect(TT.LBRACE)
        params = self._parse_block_params()
        self.skip_newlines()
        body = self._parse_body_until(TT.RBRACE)
        self.expect(TT.RBRACE)
        return Block(params, body)

    def _parse_body_until(self, *stop_types: TT) -> list[Node]:
        """Parse statements until one of stop_types is reached."""
        body: list[Node] = []
        while not self.match(*stop_types, TT.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                body.append(stmt)
            while self.match(TT.NEWLINE):
                self.advance()
            self.skip_newlines()
        return body

    # ── Arrays ───────────────────────────────────────────────────────────

    def _parse_array(self) -> ArrayLit:
        self.expect(TT.LBRACKET)
        elements: list[Node] = []
        self.skip_newlines()
        while not self.match(TT.RBRACKET, TT.EOF):
            elements.append(self._parse_expr())
            self.skip_newlines()
            if self.match(TT.COMMA):
                self.advance()
                self.skip_newlines()
        self.expect(TT.RBRACKET)
        return ArrayLit(elements)

    def _parse_hash_lit(self) -> HashLit:
        self.expect(TT.LBRACE)
        pairs: dict = {}
        self.skip_newlines()
        while not self.match(TT.RBRACE, TT.EOF):
            if self.match(TT.KWARG):
                key = self.advance().value
                val = self._parse_expr()
                pairs[key] = val
            self.skip_newlines()
            if self.match(TT.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        if self.match(TT.RBRACE):
            self.advance()
        return HashLit(pairs)

    # ── Control flow ─────────────────────────────────────────────────────

    def _parse_if(self) -> IfStmt:
        self.expect(TT.IF)
        cond = self._parse_expr()
        if self.match(TT.THEN):
            self.advance()
        self.skip_newlines()
        then_body = self._parse_body_until(TT.ELSE, TT.ELSIF, TT.END)
        elsif_clauses: list[tuple[Node, list[Node]]] = []
        else_body: Optional[list[Node]] = None
        while self.match(TT.ELSIF):
            self.advance()
            ec = self._parse_expr()
            if self.match(TT.THEN):
                self.advance()
            self.skip_newlines()
            eb = self._parse_body_until(TT.ELSE, TT.ELSIF, TT.END)
            elsif_clauses.append((ec, eb))
        if self.match(TT.ELSE):
            self.advance()
            self.skip_newlines()
            else_body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return IfStmt(cond, then_body, elsif_clauses, else_body)

    def _parse_unless(self) -> IfStmt:
        self.expect(TT.UNLESS)
        cond = self._parse_expr()
        if self.match(TT.THEN):
            self.advance()
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return IfStmt(UnaryOp('not', cond), body, [], None)

    def _parse_while(self) -> WhileStmt:
        self.expect(TT.WHILE)
        cond = self._parse_expr()
        if self.match(TT.DO):
            self.advance()
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return WhileStmt(cond, body)

    def _parse_until(self) -> WhileStmt:
        self.expect(TT.UNTIL)
        cond = self._parse_expr()
        if self.match(TT.DO):
            self.advance()
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return WhileStmt(UnaryOp('not', cond), body)

    def _parse_for(self) -> MethodCall:
        """for x in collection → collection.each do |x| body end"""
        self.expect(TT.FOR)
        var = self.expect(TT.IDENT).value
        self.expect(TT.IN)
        collection = self._parse_expr()
        if self.match(TT.DO) or self.match(TT.NEWLINE):
            self.advance()
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return MethodCall(collection, 'each', [], {}, Block([var], body))

    def _parse_case(self) -> CaseStmt:
        self.expect(TT.CASE)
        # optional subject expression
        expr = None
        if not self.match(TT.NEWLINE, TT.EOF):
            expr = self._parse_expr()
        self.skip_newlines()
        whens: list[tuple[Node, list[Node]]] = []
        else_body: Optional[list[Node]] = None
        while not self.match(TT.END, TT.EOF):
            if self.match(TT.WHEN):
                self.advance()
                # Collect comma-separated when values → wrap in ArrayLit if multiple
                vals = [self._parse_expr()]
                while self.match(TT.COMMA):
                    self.advance()
                    vals.append(self._parse_expr())
                val = ArrayLit(vals) if len(vals) > 1 else vals[0]
                if self.match(TT.THEN):
                    self.advance()
                self.skip_newlines()
                body = self._parse_body_until(TT.WHEN, TT.ELSE, TT.END)
                whens.append((val, body))
            elif self.match(TT.ELSE):
                self.advance()
                self.skip_newlines()
                else_body = self._parse_body_until(TT.END)
            else:
                self.advance()
        self.expect(TT.END)
        return CaseStmt(expr, whens, else_body)

    def _parse_def(self) -> FuncDef:
        self.expect(TT.DEF)
        name = self.expect(TT.IDENT).value
        # def self.method_name
        if name == 'self' and self.match(TT.DOT):
            self.advance()  # consume .
            name = self.expect(TT.IDENT).value
        params: list[str] = []
        defaults: dict = {}
        if self.match(TT.LPAREN):
            self.advance()
            while not self.match(TT.RPAREN, TT.EOF):
                if self.match(TT.STAR):
                    self.advance()
                    if self.match(TT.IDENT):
                        params.append('*' + self.advance().value)
                elif self.match(TT.AMPER):
                    # &block_param — block passed to method
                    self.advance()
                    if self.match(TT.IDENT):
                        params.append('&' + self.advance().value)
                elif self.match(TT.IDENT):
                    pname = self.advance().value
                    params.append(pname)
                    # optional default value:  param = expr
                    if self.match(TT.ASSIGN):
                        self.advance()
                        defaults[pname] = self._parse_arg_val()
                if self.match(TT.COMMA):
                    self.advance()
                elif not self.match(TT.RPAREN):
                    self.advance()  # skip unexpected
            self.expect(TT.RPAREN)
        self.skip_newlines()
        body = self._parse_body_until(TT.END)
        self.expect(TT.END)
        return FuncDef(name, params, body, defaults)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(source: str) -> Program:
    """Parse Sonic Pi source code and return an AST Program node."""
    tokens = tokenize(source)
    return Parser(tokens).parse()
