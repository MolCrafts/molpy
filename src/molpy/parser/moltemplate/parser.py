"""Hand-written tokenizer + recursive-descent parser for MolTemplate (.lt).

Design note:
    Real moltemplate files interleave structured directives (``ClassName {``,
    ``write_once("Section") {``, ``new Foo.move(...)``) with free-form LAMMPS
    coeff lines inside the block bodies. A pure Lark grammar is awkward
    because the body grammar differs per section. Instead we parse structure
    via a tokenizer that tracks brace depth, capture block bodies as raw text
    lines, and let ``builder.py`` regex-parse them per section.
"""

from __future__ import annotations

import re
from pathlib import Path

from .ir import (
    ArrayDim,
    ClassDef,
    Document,
    ImportStmt,
    NewStmt,
    RandomChoice,
    ReplaceStmt,
    Statement,
    Transform,
    WriteBlock,
    WriteOnceBlock,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<COMMENT>\#[^\n]*)
    | (?P<STRING>"[^"]*")
    | (?P<LBRACE>\{)
    | (?P<RBRACE>\})
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<LBRACK>\[)
    | (?P<RBRACK>\])
    | (?P<COMMA>,)
    | (?P<DOT>\.)
    | (?P<EQ>=)
    | (?P<NUMBER>-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+(?:[eE][+-]?\d+)?)
    | (?P<IDENT>[A-Za-z_][A-Za-z_0-9/\-:@$]*)
    | (?P<OTHER>.)
    """,
    re.VERBOSE,
)

_STRUCTURED_KEYWORDS = {"write", "write_once", "import", "new", "inherits"}


class Token:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind: str, value: str, line: int):
        self.kind = kind
        self.value = value
        self.line = line

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.value!r}, line={self.line})"


def tokenize(source: str) -> list[Token]:
    """Lex ``source`` into a list of tokens (whitespace/comments dropped)."""
    tokens: list[Token] = []
    line = 1
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup or "OTHER"
        value = m.group()
        if kind == "WS":
            line += value.count("\n")
            continue
        if kind == "COMMENT":
            continue
        if kind == "STRING":
            value = value[1:-1]
        tokens.append(Token(kind, value, line))
    tokens.append(Token("EOF", "", line))
    return tokens


# ---------------------------------------------------------------------------
# Brace-aware body extraction
# ---------------------------------------------------------------------------


def _extract_block_body(source: str, start_idx: int) -> tuple[list[str], int]:
    """Extract raw text lines inside a ``{ ... }`` block starting at ``start_idx``.

    ``source[start_idx]`` must be '{'. Returns (lines, end_idx_exclusive) where
    ``source[end_idx_exclusive - 1] == '}'`` and ``lines`` is the stripped-
    line body (no leading/trailing whitespace on each line, empty lines kept
    if user-written).
    """
    assert source[start_idx] == "{", f"expected '{{' at {start_idx}"
    depth = 1
    i = start_idx + 1
    body_start = i
    while i < len(source) and depth > 0:
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        elif c == "#":
            # skip to end of line
            nl = source.find("\n", i)
            if nl == -1:
                i = len(source)
                break
            i = nl
            continue
        elif c == '"':
            # skip over string literal
            close = source.find('"', i + 1)
            if close == -1:
                raise SyntaxError(f"unterminated string starting at offset {i}")
            i = close + 1
            continue
        i += 1
    if depth != 0:
        raise SyntaxError(f"unbalanced '{{' starting at {start_idx}")
    body = source[body_start:i]
    lines = [
        ln.strip()
        for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return lines, i + 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class MolTemplateParser:
    """Parse MolTemplate source into a ``Document`` IR.

    The parser uses a two-pass strategy:
      1. Scan the source character-by-character to find top-level statements,
         extracting block bodies via :func:`_extract_block_body`.
      2. Tokenize the structural prefix of each statement to identify
         keywords and names.

    This keeps brace-depth handling simple and avoids grammar explosion
    for the free-form content inside ``write``/``write_once`` blocks.
    """

    def parse(self, source: str) -> Document:
        doc = Document()
        doc.statements = self._parse_block(source, 0, len(source))
        return doc

    def _parse_block(self, source: str, start: int, end: int) -> list[Statement]:
        """Parse statements in ``source[start:end]`` (top-level or inside a class)."""
        statements: list[Statement] = []
        i = start
        while i < end:
            # skip whitespace
            while i < end and source[i].isspace():
                i += 1
            if i >= end:
                break
            # comments
            if source[i] == "#":
                nl = source.find("\n", i)
                i = end if nl == -1 else nl
                continue
            # import "file.lt"
            if source.startswith("import", i) and _is_word_boundary(source, i, i + 6):
                stmt, i = self._parse_import(source, i + 6, end)
                statements.append(stmt)
                continue
            # Parse `replace{ @atom:A @atom:B }` into ReplaceStmt so the
            # builder can apply atom-type decoration rules used by oplsaa*.lt.
            if source.startswith("replace", i) and _is_word_boundary(source, i, i + 7):
                stmt, i = self._parse_replace(source, i + 7, end)
                if stmt is not None:
                    statements.append(stmt)
                continue
            # Silently skip directives we don't yet model:
            #   create_var {...}, delete_var, category
            # Consumes either a `{...}` block or a single line.
            skipped = False
            for kw in ("create_var", "delete_var", "category"):
                if source.startswith(kw, i) and _is_word_boundary(
                    source, i, i + len(kw)
                ):
                    j = i + len(kw)
                    while j < end and source[j].isspace() and source[j] != "\n":
                        j += 1
                    if j < end and source[j] == "{":
                        _, j = _extract_block_body(source, j)
                    else:
                        nl = source.find("\n", j)
                        j = end if nl == -1 else nl + 1
                    i = j
                    skipped = True
                    break
            if skipped:
                continue
            # write(...) { ... }
            if source.startswith("write_once", i) and _is_word_boundary(
                source, i, i + 10
            ):
                stmt, i = self._parse_write(source, i + 10, end, once=True)
                statements.append(stmt)
                continue
            if source.startswith("write", i) and _is_word_boundary(source, i, i + 5):
                stmt, i = self._parse_write(source, i + 5, end, once=False)
                statements.append(stmt)
                continue
            # instance = new ClassName
            # Instance name may contain array-element subscripts `[N]`, e.g.
            # `monomers[0]`, and slashes for scoped names.
            m = re.match(
                r"([A-Za-z0-9_][\w:$@/]*(?:\[\s*\d+\s*\])?)\s*=\s*new\b",
                source[i:end],
            )
            if m:
                stmt, i = self._parse_new(source, i, end)
                statements.append(stmt)
                continue
            # class definition: IDENT (inherits ...)? { ... }
            m = re.match(
                r"([A-Za-z0-9_][\w]*)\s*(?:inherits\b([^{]*))?\s*\{",
                source[i:end],
            )
            if m and source[i + m.end() - 1] == "{":
                stmt, i = self._parse_class(source, i, end)
                statements.append(stmt)
                continue
            # Unknown construct — skip to next line to stay robust
            nl = source.find("\n", i)
            i = end if nl == -1 else nl + 1
        return statements

    def _parse_import(self, source: str, i: int, end: int) -> tuple[ImportStmt, int]:
        # Expect: whitespace then "path" or unquoted path-until-newline.
        while i < end and source[i].isspace() and source[i] != "\n":
            i += 1
        if i >= end:
            raise SyntaxError("import: missing path")
        if source[i] in ('"', "'"):
            q = source[i]
            close = source.find(q, i + 1)
            if close == -1:
                raise SyntaxError("import: unterminated string")
            return ImportStmt(path=source[i + 1 : close]), close + 1
        # Unquoted form: read until newline/semicolon/whitespace-terminator.
        m = re.match(r"\S+", source[i:end])
        if m is None:
            raise SyntaxError(f"import: expected path near offset {i}")
        return ImportStmt(path=m.group().rstrip(";")), i + m.end()

    def _parse_write(
        self, source: str, i: int, end: int, *, once: bool
    ) -> tuple[WriteBlock | WriteOnceBlock, int]:
        # Expect: '(' "section name (may contain parens)" ')' '{' ... '}'
        while i < end and source[i].isspace():
            i += 1
        if source[i] != "(":
            raise SyntaxError(f"write: expected '(' at offset {i}")
        # Skip the opening '('
        k = i + 1
        # Skip leading whitespace
        while k < end and source[k].isspace():
            k += 1
        if k < end and source[k] in ('"', "'"):
            # Quoted section: find closing quote, then the matching ')'
            quote = source[k]
            close_q = source.find(quote, k + 1)
            if close_q == -1:
                raise SyntaxError("write: unterminated quoted section name")
            section = source[k + 1 : close_q]
            # Now find the next ')' after the closing quote
            close_paren = source.find(")", close_q + 1)
            if close_paren == -1:
                raise SyntaxError("write: unterminated '('")
        else:
            # Unquoted: find next ')' at paren-depth 0
            close_paren = source.find(")", k)
            if close_paren == -1:
                raise SyntaxError("write: unterminated '('")
            section = source[k:close_paren].strip()
        j = close_paren + 1
        while j < end and source[j].isspace():
            j += 1
        if j >= end or source[j] != "{":
            raise SyntaxError(f"write: expected '{{' at offset {j}")
        body, end_idx = _extract_block_body(source, j)
        if once:
            return WriteOnceBlock(section=section, body_lines=body), end_idx
        return WriteBlock(section=section, body_lines=body), end_idx

    def _parse_new(self, source: str, i: int, end: int) -> tuple[NewStmt, int]:
        # Try the standard form first: inst = new [N]? ClassName
        # (instance name may have [k] subscript, e.g. `monomers[0]`)
        m = re.match(
            r"([A-Za-z0-9_][\w:$@/]*(?:\[\s*\d+\s*\])?)\s*=\s*new\s*"
            r"(?:\[\s*(\d+)\s*\])?\s*([A-Za-z0-9_][\w]*)",
            source[i:end],
        )
        if m is None:
            raise SyntaxError(f"new: parse failure near offset {i}")
        instance = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        cls_name = m.group(3)
        cursor = i + m.end()
        # Special case: `new random([Cls.a(...), Cls2], [w1, w2] [, seed])`.
        # Parse the argument list into ``random_choices`` and ``random_weights``
        # so the builder can materialise a weighted random mixture at
        # grid-expansion time.
        if cls_name == "random":
            choices: list[RandomChoice] = []
            weights: list[float] = []
            seed: int | None = None
            k = cursor
            while k < end and source[k].isspace():
                k += 1
            if k < end and source[k] == "(":
                # Find matching ')'
                depth = 1
                start_inner = k + 1
                k += 1
                while k < end and depth > 0:
                    ch = source[k]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                inner = source[start_inner:k]
                cursor = k + 1 if k < end else k
                choices, weights, seed = _parse_random_args(inner)
            # Continue parsing post-class arrays/transforms — same syntax as
            # a regular ``new``, e.g. ``new random(...) [10].move(0,0,5)``.
            transforms, arrays, cursor = _parse_post_new_chain(source, cursor, end)
            return NewStmt(
                instance_name=instance,
                class_name="random",
                count=count,
                transforms=transforms,
                arrays=arrays,
                random_choices=choices,
                random_weights=weights,
                random_seed=seed,
            ), cursor
        transforms, arrays, cursor = _parse_post_new_chain(source, cursor, end)
        return NewStmt(
            instance_name=instance,
            class_name=cls_name,
            count=count,
            transforms=transforms,
            arrays=arrays,
        ), cursor

    def _parse_replace(
        self, source: str, i: int, end: int
    ) -> tuple[ReplaceStmt | None, int]:
        """Parse ``replace{ @kind:A @kind:B [@kind:C @kind:D ...] }``.

        Tokens inside the braces come in pairs; each pair is ``(old, new)``.
        Multiple pairs per block are allowed (newline- or whitespace-
        separated). Returns ``None`` if the block is empty or unbalanced
        (the builder silently ignores malformed replace blocks to stay
        compatible with moltemplate's permissive behaviour).
        """
        while i < end and source[i].isspace():
            i += 1
        if i >= end or source[i] != "{":
            nl = source.find("\n", i)
            return None, end if nl == -1 else nl + 1
        body, end_idx = _extract_block_body(source, i)
        tokens: list[str] = []
        for ln in body:
            for tok in ln.split():
                if tok.startswith("@"):
                    tokens.append(tok)
        if len(tokens) % 2 != 0 or not tokens:
            return None, end_idx
        pairs = [(tokens[k], tokens[k + 1]) for k in range(0, len(tokens), 2)]
        return ReplaceStmt(pairs=pairs), end_idx

    def _parse_class(self, source: str, i: int, end: int) -> tuple[ClassDef, int]:
        m = re.match(
            r"([A-Za-z0-9_][\w]*)\s*(?:inherits\s+([^{]+))?\s*\{",
            source[i:end],
        )
        if m is None:
            raise SyntaxError(f"class: parse failure near offset {i}")
        name = m.group(1)
        bases_raw = m.group(2) or ""
        bases = [b.strip() for b in re.split(r"[,\s]+", bases_raw.strip()) if b.strip()]
        brace_idx = i + m.end() - 1
        # Find matching '}' while respecting nested braces
        depth = 1
        j = brace_idx + 1
        while j < end and depth > 0:
            c = source[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            elif c == "#":
                nl = source.find("\n", j)
                j = end if nl == -1 else nl
                continue
            elif c == '"':
                close = source.find('"', j + 1)
                if close == -1:
                    raise SyntaxError("class: unterminated string")
                j = close + 1
                continue
            j += 1
        if depth != 0:
            raise SyntaxError(f"class '{name}': unbalanced braces")
        body_start = brace_idx + 1
        body_end = j
        inner_stmts = self._parse_block(source, body_start, body_end)
        return ClassDef(name=name, bases=bases, statements=inner_stmts), j + 1


def _is_word_boundary(source: str, start: int, end: int) -> bool:
    """Return True if ``source[end]`` is not an identifier continuation char."""
    if end >= len(source):
        return True
    c = source[end]
    return not (c.isalnum() or c == "_")


def _parse_post_new_chain(
    source: str, cursor: int, end: int
) -> tuple[list[Transform], list[ArrayDim], int]:
    """Parse the ``.op(args)`` / ``[N].op(args)`` chain that follows a ``new``.

    Returns ``(transforms, arrays, new_cursor)``. Stops at the first
    non-chain character (e.g. end-of-line or start of another statement).
    """
    transforms: list[Transform] = []
    arrays: list[ArrayDim] = []
    while cursor < end:
        k = cursor
        while k < end and source[k].isspace():
            k += 1
        if k >= end:
            break
        c = source[k]
        if c == "[":
            am = re.match(
                r"\[\s*(\d+)\s*\]\s*(?:\.\s*([A-Za-z_][\w]*)\s*\(\s*([^)]*)\)\s*)?",
                source[k:end],
            )
            if am is None:
                break
            n = int(am.group(1))
            tr: Transform | None = None
            if am.group(2) is not None:
                op = am.group(2)
                args_str = (am.group(3) or "").strip()
                args = _parse_float_csv(args_str)
                tr = Transform(op=op, args=args)
            arrays.append(ArrayDim(count=n, transform=tr))
            cursor = k + am.end()
            continue
        if c == ".":
            tm = re.match(
                r"\.\s*([A-Za-z_][\w]*)\s*\(\s*([^)]*)\)",
                source[k:end],
            )
            if tm is None:
                break
            op = tm.group(1)
            args_str = tm.group(2).strip()
            transforms.append(Transform(op=op, args=_parse_float_csv(args_str)))
            cursor = k + tm.end()
            continue
        break
    return transforms, arrays, cursor


def _parse_float_csv(s: str) -> list[float]:
    """Parse a comma-separated string of floats. Empty string → empty list."""
    if not s:
        return []
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _parse_random_args(
    inner: str,
) -> tuple[list[RandomChoice], list[float], int | None]:
    """Parse the argument list of ``random(...)``.

    Expected forms (whitespace-insensitive)::

        [Cls1, Cls2]                      # uniform
        [Cls1, Cls2], [w1, w2]            # weighted
        [Cls1, Cls2], [w1, w2], 12345     # weighted with seed
        [Cls1.move(0,0,3), Cls2], [c1,c2] # per-choice transforms

    Returns ``(choices, weights, seed)`` where weights is empty when the
    caller should assume uniform weights.
    """
    bracket_blocks: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(inner):
        c = inner[i]
        if c == "[":
            depth = 1
            start = i + 1
            i += 1
            while i < len(inner) and depth > 0:
                cc = inner[i]
                if cc == "[":
                    depth += 1
                elif cc == "]":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            bracket_blocks.append(inner[start:i])
            i += 1
            continue
        if c == ",":
            i += 1
            continue
        # Tail after last ']' — may contain a seed number.
        rest.append(c)
        i += 1

    if not bracket_blocks:
        return [], [], None

    choices = [
        _parse_random_choice(tok.strip())
        for tok in _split_top_level(bracket_blocks[0])
        if tok.strip()
    ]
    weights: list[float] = []
    if len(bracket_blocks) >= 2:
        weights = _parse_float_csv(bracket_blocks[1].strip())
    seed_str = "".join(rest).strip().rstrip(",").strip()
    seed = int(seed_str) if seed_str and seed_str.lstrip("-").isdigit() else None
    return choices, weights, seed


def _split_top_level(s: str) -> list[str]:
    """Split ``s`` on commas that are outside any ``()``/``[]`` nesting."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def _parse_random_choice(tok: str) -> RandomChoice:
    """Parse a single ``random`` entry such as ``Cls.move(1,2,3).rot(...)``."""
    m = re.match(r"([A-Za-z_][\w]*)", tok)
    if m is None:
        raise SyntaxError(f"random: expected class name in {tok!r}")
    name = m.group(1)
    transforms, _, _ = _parse_post_new_chain(tok, m.end(), len(tok))
    return RandomChoice(class_name=name, transforms=transforms)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_string(source: str) -> Document:
    """Parse MolTemplate source text into a ``Document`` IR tree."""
    return MolTemplateParser().parse(source)


# Cache keyed by (resolved path, mtime-ns, size). Keeps hot-reload safe while
# giving O(1) reuse across repeat imports of the same file (e.g. every
# example importing oplsaa.lt).
_PARSE_CACHE: dict[tuple[str, int, int], Document] = {}


def parse_file(path: str | Path) -> Document:
    """Parse a ``.lt`` file from disk (memoised by mtime+size)."""
    p = Path(path).resolve()
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    doc = parse_string(p.read_text())
    if key is not None:
        _PARSE_CACHE[key] = doc
    return doc
