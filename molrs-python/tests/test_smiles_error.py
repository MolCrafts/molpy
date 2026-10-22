"""Typed exception for the SMILES / CGsmiles surface.

These are FFI-seam tests: they prove that the Rust ``SmilesError`` crosses as
one Python class carrying the four facts it owns — ``kind``, ``span``,
``input``, ``notation`` — instead of collapsing to a bare ``ValueError``
message. They re-derive no chemistry: every ``kind`` asserted here is the
variant the Rust unit tests in ``crates/molrs/src/io/smiles/`` already pin
(``parser.rs`` for ``UnclosedBranch``, ``cgsmiles/parser.rs::split_blocks``
for ``UnexpectedEnd``, ``cgsmiles/to_atomistic.rs`` /
``cgsmiles/templates.rs`` for ``CgNotExpandable``), reused only to show
Python sees the same answer.

Fixtures are inline strings; no third-party scientific software runs.
"""

from __future__ import annotations

import molrs
import pytest

# An unterminated block: ``split_blocks`` finds no ``}`` and reports
# ``UnexpectedEnd`` (crates/molrs/src/io/cgsmiles/parser.rs, the
# ``input[pos..].find('}')`` arm).
UNTERMINATED_BLOCK = "{[#A]"

# A branch that never closes: ``parse_branch`` reports ``UnclosedBranch``
# spanned at the ``(`` it opened (crates/molrs/src/io/smiles/parser.rs).
UNCLOSED_BRANCH = "C("

# A base-only CGsmiles string: it parses, but there is no fragment table to
# expand it with, so both expansions report ``CgNotExpandable``.
BASE_ONLY = "{[#A][#B]}"

# A bracket atom whose symbol is not an element: the SMILES parser reports
# ``InvalidElement`` naming ``Xx`` (crates/molrs/src/io/smiles/parser.rs, the bracket
# symbol arm; the same kind and payload
# ``smiles/validate.rs::validate_symbol`` uses for the same rule).
UNKNOWN_BRACKET_ELEMENT = "[Xx]"


# ---------------------------------------------------------------------------
# The class itself
# ---------------------------------------------------------------------------


def test_smiles_error_is_a_public_value_error() -> None:
    """One typed exception, published from ``molrs.io.smiles`` and still a
    ``ValueError`` so no existing ``except ValueError`` stops catching it."""
    assert issubclass(molrs.io.smiles.SmilesError, ValueError)
    assert "SmilesError" in molrs.io.smiles.__all__


# ---------------------------------------------------------------------------
# CGsmiles: an unterminated block
# ---------------------------------------------------------------------------


def test_cgsmiles_parse_error_raises_the_typed_exception() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError):
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)


def test_cgsmiles_error_kind_is_the_rust_variant_name() -> None:
    """``kind`` crosses as the ``SmilesErrorKind`` variant name, payload
    dropped — ``UnexpectedEnd`` for a block that never closes."""
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    kind = excinfo.value.kind
    assert isinstance(kind, str)
    assert kind
    assert kind == "UnexpectedEnd"


def test_cgsmiles_error_span_is_a_two_tuple_of_ints() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    span = excinfo.value.span
    assert isinstance(span, tuple)
    assert len(span) == 2
    assert all(isinstance(value, int) for value in span)
    start, end = span
    assert 0 <= start <= end
    assert start <= len(UNTERMINATED_BLOCK)


def test_cgsmiles_error_span_stays_within_the_input() -> None:
    """A span indexes the string it came from, so its end is a valid slice
    bound.

    The Rust side reports this one as ``Span::new(len, len + 1)``
    (``cgsmiles/parser.rs``, the unterminated-block arm), which is one past
    the end: the seam must land the end inside the text it publishes.
    """
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    assert excinfo.value.span[1] <= len(UNTERMINATED_BLOCK)


def test_cgsmiles_error_echoes_the_offending_input() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    assert excinfo.value.input == UNTERMINATED_BLOCK


def test_cgsmiles_error_names_the_cgsmiles_notation() -> None:
    """``notation`` crosses as the lowercase variant name, like every other
    enum at this seam (see ``test_cgsmiles.py``)."""
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    assert excinfo.value.notation == "cgsmiles"


def test_cgsmiles_error_renders_the_rust_message() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    assert str(excinfo.value).startswith("CGsmiles parse error at position ")


# ---------------------------------------------------------------------------
# Plain SMILES: the same class, a different notation
# ---------------------------------------------------------------------------


def test_smiles_error_names_the_smiles_notation() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.smiles.read_smiles(UNCLOSED_BRANCH)

    assert excinfo.value.notation == "smiles"


def test_smiles_unclosed_branch_crosses_its_kind() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.smiles.read_smiles(UNCLOSED_BRANCH)

    assert excinfo.value.kind == "UnclosedBranch"


def test_smiles_unclosed_branch_span_points_at_the_open_paren() -> None:
    """``parse_branch`` spans from the ``(`` that was opened, byte 1 of
    ``C(``."""
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.smiles.read_smiles(UNCLOSED_BRANCH)

    assert excinfo.value.span[0] == 1


def test_unknown_bracket_element_is_refused_by_the_constructor() -> None:
    """``Xx`` is not an element, so the string never becomes an IR.

    Constructing it would let ``to_atomistic()`` hand back a graph with an atom
    whose element is the string ``"Xx"`` — a graph no chemistry can read. The
    refusal belongs to the parser, so it happens here.
    """
    with pytest.raises(molrs.io.smiles.SmilesError):
        molrs.io.smiles.read_smiles(UNKNOWN_BRACKET_ELEMENT)


def test_unknown_bracket_element_kind_is_invalid_element() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.smiles.read_smiles(UNKNOWN_BRACKET_ELEMENT)

    assert excinfo.value.kind == "InvalidElement"


def test_unknown_bracket_element_names_the_smiles_notation() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.smiles.read_smiles(UNKNOWN_BRACKET_ELEMENT)

    assert excinfo.value.notation == "smiles"


# ---------------------------------------------------------------------------
# Errors raised past the parser: expansion of a base-only string
# ---------------------------------------------------------------------------


def test_base_only_string_is_not_expandable_to_atomistic() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(BASE_ONLY).to_atomistic()

    assert excinfo.value.kind == "CgNotExpandable"


def test_base_only_string_has_no_templates() -> None:
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(BASE_ONLY).templates()

    assert excinfo.value.kind == "CgNotExpandable"


# ---------------------------------------------------------------------------
# SmilesError is a ValueError, with one spelling per fact
# ---------------------------------------------------------------------------


def test_a_smiles_error_is_a_value_error() -> None:
    """``SmilesError`` subclasses ``ValueError``, so ``except ValueError``
    catches it."""
    with pytest.raises(ValueError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    assert isinstance(excinfo.value, molrs.io.smiles.SmilesError)


def test_smiles_error_has_no_second_spelling_of_its_facts() -> None:
    """Four attributes cross, and only four: position/offset/message/args-style
    aliases would give every fact two spellings."""
    with pytest.raises(molrs.io.smiles.SmilesError) as excinfo:
        molrs.io.cgsmiles.read_cgsmiles(UNTERMINATED_BLOCK)

    error = excinfo.value
    for absent in ("position", "offset", "start", "end", "message", "text"):
        assert not hasattr(error, absent), absent
