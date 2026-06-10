"""Behavior-preservation tests for the polymer build loop perf refactor.

Spec builder-reacter-05-perf:

- GOLDEN determinism: DP-10/DP-50 chain snapshots (counts + inter-monomer
  connectivity) must match the committed golden JSON; regenerate with
  ``MOLPY_REGEN_GOLDEN=1``.
- BOUNDED port scans: ``get_ports_on_node`` must not scan the growing
  accumulated chain — max single-call scan size must stay flat with DP.
- GROUP correctness: union-find/group bookkeeping keeps all monomer nodes
  in one structure with exact linear count arithmetic (no RDKit needed).
- SLOW smoke: DP=200 build completes within a wall-clock budget.
"""

import json
import os
import time
from pathlib import Path

import pytest

from molpy.core.atomistic import Atomistic

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# ── shared build helpers ──────────────────────────────────────────────


def _inter_monomer_edges(struct: Atomistic) -> list[list[int]]:
    """Sorted, NORMALIZED node-id pairs of bonds crossing monomer nodes.

    Raw ``monomer_node_id`` values come from a process-global parser
    counter and therefore depend on how many parses ran earlier in the
    process; they are relabeled here by order of first appearance so the
    snapshot is test-order independent.
    """
    raw_edges: list[tuple[int, int]] = []
    relabel: dict[int, int] = {}
    for bond in struct.bonds:
        node_i = bond.itom.get("monomer_node_id")
        node_j = bond.jtom.get("monomer_node_id")
        if node_i is None or node_j is None or node_i == node_j:
            continue
        for nid in sorted((node_i, node_j)):
            if nid not in relabel:
                relabel[nid] = len(relabel)
        raw_edges.append((node_i, node_j))
    edges = [sorted([relabel[node_i], relabel[node_j]]) for node_i, node_j in raw_edges]
    return sorted(edges)


def _build_peo_chain(
    dp: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[Atomistic, list[list[int]]]:
    """Build ``{[<]CCO[>]}|dp|`` and capture inter-monomer edges.

    Build markers (``monomer_node_id``) are stripped by
    ``cleanup_build_markers`` at the end of ``PolymerBuilder.build``, so
    the connectivity summary is captured just before cleanup runs
    (``core.py`` resolves the function from the module at call time).
    """
    from molpy.builder import polymer
    from molpy.builder.polymer import port_utils

    captured: dict[str, list[list[int]]] = {}
    real_cleanup = port_utils.cleanup_build_markers

    def capturing_cleanup(struct: Atomistic) -> None:
        captured["edges"] = _inter_monomer_edges(struct)
        real_cleanup(struct)

    with monkeypatch.context() as patch:
        patch.setattr(port_utils, "cleanup_build_markers", capturing_cleanup)
        chain = polymer(f"{{[<]CCO[>]}}|{dp}|", optimize=False, random_seed=42)

    return chain, captured.get("edges", [])


def _snapshot(chain: Atomistic, edges: list[list[int]]) -> dict:
    return {
        "n_atoms": len(list(chain.atoms)),
        "n_bonds": len(list(chain.bonds)),
        "n_angles": len(list(chain.angles)),
        "n_dihedrals": len(list(chain.dihedrals)),
        "inter_monomer_bonds": edges,
    }


# ── GOLDEN determinism ────────────────────────────────────────────────


class TestBuildDeterminism:
    """Chain snapshot must be bit-identical to the committed golden."""

    @pytest.mark.parametrize("dp", [10, 50])
    def test_build_determinism_matches_golden(
        self, dp: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("rdkit")

        chain, edges = _build_peo_chain(dp, monkeypatch)
        snapshot = _snapshot(chain, edges)

        golden_path = GOLDEN_DIR / f"polymer_dp{dp}_seed42.json"
        if os.environ.get("MOLPY_REGEN_GOLDEN") == "1":
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        assert golden_path.exists(), (
            f"Golden file {golden_path} missing; regenerate with MOLPY_REGEN_GOLDEN=1"
        )
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        assert snapshot == golden


# ── BOUNDED port scans ────────────────────────────────────────────────


class TestBoundedPortScans:
    """Port lookups must not scan the growing accumulated chain."""

    @staticmethod
    def _max_scan_size(dp: int, monkeypatch: pytest.MonkeyPatch) -> int:
        """Build DP=dp; record the largest struct scanned per call."""
        from molpy.builder.polymer import port_utils

        sizes: list[int] = []
        real = port_utils.get_ports_on_node

        def recording(struct: Atomistic, node_id: int):
            sizes.append(len(list(struct.atoms)))
            return real(struct, node_id)

        with monkeypatch.context() as patch:
            patch.setattr(port_utils, "get_ports_on_node", recording)
            _build_peo_chain(dp, monkeypatch)

        return max(sizes, default=0)

    def test_port_scan_bounded_as_chain_grows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("rdkit")

        max_dp10 = self._max_scan_size(10, monkeypatch)
        max_dp50 = self._max_scan_size(50, monkeypatch)

        assert max_dp50 <= 1.5 * max_dp10, (
            f"Largest single get_ports_on_node scan grew with chain "
            f"length: DP=50 scanned {max_dp50} atoms vs DP=10 scanned "
            f"{max_dp10} atoms (ratio "
            f"{max_dp50 / max_dp10 if max_dp10 else float('inf'):.2f}, "
            f"budget 1.5) — port lookups must come from a per-build "
            f"registry, not the accumulated chain"
        )


# ── GROUP correctness (no RDKit) ──────────────────────────────────────


def _make_linear_monomer() -> Atomistic:
    """3-atom C-C-C monomer with explicit ``<`` / ``>`` ports and xyz."""
    struct = Atomistic()
    c1 = struct.def_atom(element="C", symbol="C", x=0.0, y=0.0, z=0.0, port="<")
    c2 = struct.def_atom(element="C", symbol="C", x=1.54, y=0.0, z=0.0)
    c3 = struct.def_atom(element="C", symbol="C", x=3.08, y=0.0, z=0.0, port=">")
    struct.def_bond(c1, c2, order=1)
    struct.def_bond(c2, c3, order=1)
    return struct


def _make_ring_monomer() -> Atomistic:
    """3-atom C-C-C monomer with two symmetric ``$`` ports."""
    struct = Atomistic()
    c1 = struct.def_atom(element="C", symbol="C", x=0.0, y=0.0, z=0.0, port="$")
    c2 = struct.def_atom(element="C", symbol="C", x=1.54, y=0.0, z=0.0)
    c3 = struct.def_atom(element="C", symbol="C", x=3.08, y=0.0, z=0.0, port="$")
    struct.def_bond(c1, c2, order=1)
    struct.def_bond(c2, c3, order=1)
    return struct


def _make_no_leaving_reacter():
    from molpy.reacter import Reacter, form_single_bond

    return Reacter(
        name="group_test",
        anchor_selector_left=lambda assembly, port_atom: port_atom,
        anchor_selector_right=lambda assembly, port_atom: port_atom,
        leaving_selector_left=lambda assembly, anchor: [],
        leaving_selector_right=lambda assembly, anchor: [],
        bond_former=form_single_bond,
    )


def _build_manual(cgsmiles: str, monomer: Atomistic, port_map: dict):
    from molpy.builder.polymer import Connector, PolymerBuilder

    connector = Connector(reacter=_make_no_leaving_reacter(), port_map=port_map)
    builder = PolymerBuilder(library={"A": monomer}, connector=connector)
    return builder.build(cgsmiles)


class TestGroupCorrectness:
    """Group/union bookkeeping yields one structure with exact counts."""

    @staticmethod
    def _linear_counts(dp: int) -> tuple[int, int, int]:
        result = _build_manual(
            f"{{[#A]|{dp}}}", _make_linear_monomer(), {("A", "A"): (">", "<")}
        )
        chain = result.polymer
        return (
            len(list(chain.atoms)),
            len(list(chain.bonds)),
            result.total_steps,
        )

    def test_group_linear_dp30_counts_extrapolate_from_small_builds(self) -> None:
        """DP=30 counts follow the linear relation fixed by DP=2 and DP=3."""
        atoms_2, bonds_2, steps_2 = self._linear_counts(2)
        atoms_3, bonds_3, steps_3 = self._linear_counts(3)
        atoms_30, bonds_30, steps_30 = self._linear_counts(30)

        assert steps_2 == 1
        assert steps_3 == 2
        assert steps_30 == 29

        atoms_per_monomer = atoms_3 - atoms_2
        bonds_per_monomer = bonds_3 - bonds_2
        assert atoms_30 == atoms_2 + 28 * atoms_per_monomer
        assert bonds_30 == bonds_2 + 28 * bonds_per_monomer

    def test_group_linear_dp30_forms_single_connected_structure(self) -> None:
        """All 30 node ids merge into one tree-connected structure."""
        atoms_30, bonds_30, steps_30 = self._linear_counts(30)

        assert steps_30 == 29
        # No-leaving-group linear chain is a tree: |E| == |V| - 1 means
        # every monomer landed in the single final structure.
        assert bonds_30 == atoms_30 - 1

    def test_group_ring_closure_merges_all_nodes_into_one_structure(self) -> None:
        """Cyclic graph {[#A]1[#A][#A]1} closes the ring on one structure."""
        result = _build_manual(
            "{[#A]1[#A][#A]1}", _make_ring_monomer(), {("A", "A"): ("$", "$")}
        )
        chain = result.polymer

        n_atoms = len(list(chain.atoms))
        n_bonds = len(list(chain.bonds))
        assert result.total_steps == 3
        assert n_atoms == 9  # 3 monomers x 3 atoms, no leaving groups
        assert n_bonds == n_atoms  # one cycle: |E| == |V|


# ── SLOW smoke ────────────────────────────────────────────────────────


class TestLargeBuildSmoke:
    """DP=200 end-to-end smoke under a wall-clock budget."""

    @pytest.mark.slow
    def test_smoke_dp200_build_under_60s(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("rdkit")

        start = time.monotonic()
        chain, _ = _build_peo_chain(200, monkeypatch)
        elapsed = time.monotonic() - start

        n_atoms = len(list(chain.atoms))
        n_bonds = len(list(chain.bonds))
        assert n_atoms > 0
        assert n_bonds >= n_atoms - 1
        assert elapsed < 60.0, (
            f"DP=200 build took {elapsed:.1f} s (budget 60 s) — build "
            f"loop is superlinear in chain length"
        )
