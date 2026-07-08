"""Design / interface tests for the RDKit adapter.

These tests verify the *design contract* of :class:`molpy.adapter.RDKitAdapter`
and the graceful optional-import behavior of :mod:`molpy.adapter`. They do NOT
require rdkit to run:

* The optional-import contract (base ``Adapter`` always importable; the RDKit
  symbols degrade to ``None`` when the backend is absent) is asserted directly,
  and the "rdkit missing" branch is exercised via a simulated blocked import.
* The interface / pure-Python-logic checks (MP_ID tagging, atom-id assignment)
  run only when rdkit is importable, and are skipped otherwise — so the module
  passes whether or not rdkit is installed.

No ``import rdkit`` and no ``pytest.importorskip("rdkit")`` appear here.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

import molpy.adapter as adapter_mod
from molpy.adapter import Adapter

_HAS_RDKIT = adapter_mod._HAS_RDKIT
requires_rdkit = pytest.mark.skipif(
    not _HAS_RDKIT, reason="rdkit adapter backend not importable"
)

# Base Adapter interface every concrete adapter must expose.
_BASE_ADAPTER_METHODS = (
    "get_internal",
    "get_external",
    "set_internal",
    "set_external",
    "sync_to_internal",
    "sync_to_external",
    "has_internal",
    "has_external",
    "check",
)


class TestOptionalImportContract:
    """molpy.adapter must import cleanly and expose a consistent surface
    regardless of whether the optional rdkit backend is installed."""

    def test_base_adapter_always_importable(self):
        # The base contract lives in a pure-Python module with no optional deps.
        from molpy.adapter.base import Adapter as BaseAdapter

        assert Adapter is BaseAdapter

    def test_rdkit_symbols_are_exposed(self):
        # The names always exist on the package (either the real objects or the
        # None sentinels), so downstream code can import them unconditionally.
        for name in ("RDKitAdapter", "MP_ID"):
            assert hasattr(adapter_mod, name)

    def test_sentinels_consistent_with_availability(self):
        # Availability flag and the exported objects must agree.
        if _HAS_RDKIT:
            assert adapter_mod.RDKitAdapter is not None
            assert adapter_mod.MP_ID is not None
        else:
            assert adapter_mod.RDKitAdapter is None
            assert adapter_mod.MP_ID is None

    def test_missing_rdkit_degrades_gracefully(self, monkeypatch):
        """Importing molpy.adapter with rdkit unavailable must not raise; the
        RDKit symbols degrade to ``None`` sentinels while the base ``Adapter``
        stays available."""
        # Only evict the two modules that depend on rdkit plus rdkit itself so
        # that the reload re-executes the optional-import guard. Keep
        # molpy.adapter.base cached so identity comparisons still hold.
        targets = ("rdkit", "molpy.adapter", "molpy.adapter.rdkit")

        def _is_target(name: str) -> bool:
            return name == "rdkit" or name.startswith("rdkit.") or name in targets

        saved = {name: mod for name, mod in sys.modules.items() if _is_target(name)}
        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "rdkit" or name.startswith("rdkit."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        try:
            for name in list(sys.modules):
                if _is_target(name):
                    del sys.modules[name]
            monkeypatch.setattr(builtins, "__import__", _blocking_import)

            reloaded = importlib.import_module("molpy.adapter")

            assert reloaded._HAS_RDKIT is False
            assert reloaded.RDKitAdapter is None
            assert reloaded.MP_ID is None
            # Base contract remains usable without the optional backend.
            assert reloaded.Adapter is Adapter
        finally:
            monkeypatch.undo()
            for name in list(sys.modules):
                if _is_target(name):
                    del sys.modules[name]
            sys.modules.update(saved)
            # Rebind the submodule attribute on the parent package too.
            if "molpy" in sys.modules and "molpy.adapter" in saved:
                sys.modules["molpy"].adapter = saved["molpy.adapter"]


@requires_rdkit
class TestRDKitAdapterDesign:
    """The RDKit adapter must honor the base Adapter contract."""

    def test_is_adapter_subclass(self):
        assert issubclass(adapter_mod.RDKitAdapter, Adapter)

    def test_mp_id_constant_value(self):
        assert adapter_mod.MP_ID == "mp_id"

    def test_implements_base_interface(self):
        adapter_cls = adapter_mod.RDKitAdapter
        for method in _BASE_ADAPTER_METHODS:
            assert callable(getattr(adapter_cls, method, None)), (
                f"RDKitAdapter is missing base method {method!r}"
            )

    def test_exposes_rdkit_specific_surface(self):
        adapter_cls = adapter_mod.RDKitAdapter
        # copy() for independent deep copies; internal/mol convenience props.
        assert callable(getattr(adapter_cls, "copy", None))
        assert isinstance(getattr(adapter_cls, "internal", None), property)
        assert isinstance(getattr(adapter_cls, "mol", None), property)

    def test_exported_in_all(self):
        for name in ("RDKitAdapter", "MP_ID"):
            assert name in adapter_mod.__all__


@requires_rdkit
class TestAtomIDAssignment:
    """Pure-Python MP_ID / atom-id tagging on molpy structures.

    ``_ensure_atom_ids`` runs at construction and operates only on the internal
    ``Atomistic`` (no rdkit objects are built), so it exercises the adapter's
    identity-management design without any conversion round-trip.
    """

    def _atomistic(self, n: int = 3):
        from molpy import Atomistic

        asm = Atomistic()
        atoms = [asm.def_atom(element="C") for _ in range(n)]
        for a, b in zip(atoms, atoms[1:]):
            asm.def_bond(a, b, order=1.0)
        return asm, atoms

    def test_atoms_without_ids_get_assigned(self):
        asm, _ = self._atomistic()
        adapter_mod.RDKitAdapter(internal=asm)  # triggers _ensure_atom_ids
        for atom in asm.atoms:
            assert atom.get("id") is not None
            assert atom.get(adapter_mod.MP_ID) is not None

    def test_assigned_mp_ids_are_unique(self):
        asm, _ = self._atomistic()
        adapter_mod.RDKitAdapter(internal=asm)
        mp_ids = [int(a.get(adapter_mod.MP_ID)) for a in asm.atoms]
        assert len(set(mp_ids)) == len(mp_ids)

    def test_existing_ids_are_preserved(self):
        asm, atoms = self._atomistic(n=2)
        atoms[0]["id"] = 100
        atoms[1]["id"] = 200
        adapter_mod.RDKitAdapter(internal=asm)
        assert atoms[0].get("id") == 100
        assert atoms[1].get("id") == 200
        # MP_ID mirrors the pre-set id.
        assert int(atoms[0].get(adapter_mod.MP_ID)) == 100
        assert int(atoms[1].get(adapter_mod.MP_ID)) == 200
