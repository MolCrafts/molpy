"""CGSmiles strings for common residue topologies.

:class:`~molpy.builder.assembly._polymer.PolymerBuilder` helpers
(``build_linear``, ``build_ring``, …) turn these into notation and always call
:meth:`~molpy.builder.assembly._polymer.PolymerBuilder.build` — the sole
expand + assemble entry. This module never expands a library or edits a graph.
"""

from __future__ import annotations

from collections.abc import Sequence


def _node(label: str) -> str:
    return f"[#{label}]"


def linear_cgsmiles(labels: Sequence[str]) -> str:
    """Path: ``{[#A][#B]…}`` or ``{[#EO]|n}`` when all labels are equal.

    Raises:
        ValueError: if ``labels`` is empty.
    """
    if not labels:
        raise ValueError("linear topology needs at least one residue label")
    labels = [str(lab) for lab in labels]
    if len(labels) == 1:
        return "{" + _node(labels[0]) + "}"
    if len(set(labels)) == 1:
        return "{" + _node(labels[0]) + f"|{len(labels)}" + "}"
    return "{" + "".join(_node(lab) for lab in labels) + "}"


def ring_cgsmiles(label: str, n: int) -> str:
    """Cycle of ``n`` identical residues: ``{[#EO]1[#EO]…[#EO]1}``.

    Raises:
        ValueError: if ``n < 3``.
    """
    if n < 3:
        raise ValueError(f"a residue ring needs n >= 3, got {n}")
    lab = str(label)
    # Open ring on the first node, close on the last (SMILES-style ring digits).
    body = (
        _node(lab) + "1" + "".join(_node(lab) for _ in range(n - 2)) + _node(lab) + "1"
    )
    return "{" + body + "}"


def star_cgsmiles(
    core: str,
    arm: str,
    *,
    n_arms: int,
    arm_length: int,
    cap: str | None = None,
) -> str:
    """Star: ``{[#X3]([#EO]|k)…([#EO]|k)[#EO]|k}`` with optional caps.

    Raises:
        ValueError: if ``n_arms < 2`` or ``arm_length < 1``.
    """
    if n_arms < 2:
        raise ValueError(f"a star needs n_arms >= 2, got {n_arms}")
    if arm_length < 1:
        raise ValueError(f"arm_length must be >= 1, got {arm_length}")

    arm_seg = _node(arm) + f"|{arm_length}"
    if cap is not None:
        arm_seg = arm_seg + _node(cap)
    # n_arms - 1 parenthesised branches + one main-chain arm (CGSmiles style).
    branches = "".join(f"({arm_seg})" for _ in range(n_arms - 1))
    return "{" + _node(core) + branches + arm_seg + "}"
