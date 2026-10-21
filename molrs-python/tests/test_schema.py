"""FFI smoke for ``molrs.core.schema``: the block-name constants and the relation
lookup cross the seam as plain strings."""

from __future__ import annotations

import molrs
import numpy as np


def test_block_names_are_the_rust_constants() -> None:
    assert molrs.core.schema.ATOMS == "atoms"
    assert molrs.core.schema.TOPOLOGY == (
        molrs.core.schema.BONDS,
        molrs.core.schema.ANGLES,
        molrs.core.schema.DIHEDRALS,
        molrs.core.schema.IMPROPERS,
    )
    assert molrs.core.schema.block(molrs.core.schema.PAIRS) is not None


def test_groups_and_meta_keys_are_projected() -> None:
    assert [str(key) for key in molrs.core.keys.IMAGES] == ["ix", "iy", "iz"]
    assert [str(key) for key in molrs.core.keys.AXIS] == ["axis_x", "axis_y", "axis_z"]
    assert molrs.core.keys.ATOM_TYPE_LABELS == "atom_type_labels"
    assert molrs.core.keys.UNITS == "units"
    assert molrs.core.keys.ATOM_TYPE_LABELS not in {
        spec.key for spec in molrs.core.schema.columns
    }


def test_relation_endpoints_reads_the_vocabulary_or_the_columns() -> None:
    rel = molrs.core.schema.relation_endpoints
    assert rel("bonds", []) == [("atomi", "atoms"), ("atomj", "atoms")]
    assert rel("links", ["atomi", "atomj"]) == [("atomi", "atoms"), ("atomj", "atoms")]
    assert rel("cell", []) == []
    assert rel("members", ["ibead", "atom"]) == [("ibead", "atoms")]
    assert rel("members", ["ibead", "atom"], {"atom": "/frame/atoms"}) == [
        ("ibead", "atoms"),
        ("atom", "/frame/atoms"),
    ]
    assert molrs.core.schema.block("members").declared_endpoints == ["atom"]


def test_the_topology_conventions_are_canonical() -> None:
    dtypes = {spec.key: spec.dtype for spec in molrs.core.schema.columns}
    assert dtypes["formal_charge"] == "i64"
    assert dtypes["chain"] == "string" and "chain_id" not in dtypes
    assert dtypes["fx"] == "float"
    assert [str(key) for key in molrs.core.keys.FORCES] == ["fx", "fy", "fz"]
    for name in ["constraints", "drudes", "members", "virtual_sites"]:
        assert molrs.core.schema.block(name) is not None


def test_numpy_dtype_is_the_dtype_a_column_is_stored_at() -> None:
    assert molrs.core.schema.column("formal_charge").numpy_dtype == "int64"
    for spec in molrs.core.schema.columns:
        width = int(spec.shape[4:-1]) if spec.shape.startswith("vec(") else None
        shape = (3,) if width is None else (3, width)
        block = molrs.core.Block()
        if spec.dtype == "string":
            assert spec.numpy_dtype == "str"
            block[spec.key] = ["a", "b", "c"]
            assert block.dtype(spec.key) == "string"
            continue
        values = np.ones(shape, dtype=bool) if spec.dtype == "bool" else np.ones(shape)
        block[spec.key] = values
        assert block[spec.key].dtype == np.dtype(spec.numpy_dtype), spec.key


def test_key_hash_agrees_with_equal_string():
    from molrs.core import keys

    assert keys.X == "x"
    assert hash(keys.X) == hash("x")
    assert {"x": 1}[keys.X] == 1
    assert len({keys.X, "x"}) == 1
