"""Compile local reaction environments before editing the assembled world.

The compiler sees the complete set of selected bindings while the monomer
templates are still intact.  For every prospective junction it materialises a
small view of the *planned product*, types that view once per distinct rooted
environment, and records only scalar per-atom annotations.  Execution can then
apply the reaction to the real world in one batch and replay those annotations;
no growing polymer is ever handed back to a typifier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import molrs

from molpy.builder.assembly._selector import Binding
from molpy.core import fields
from molpy.core.atomistic import Atomistic

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


type Scalar = str | int | float | bool


@dataclass(frozen=True)
class _AddedAtomRef:
    """An atom created by one global binding, in reaction creation order."""

    binding: int
    ordinal: int


type _AtomRef = int | _AddedAtomRef


@dataclass(frozen=True)
class _AtomPatch:
    atom: _AtomRef
    annotations: tuple[tuple[str, Scalar], ...]


@dataclass(frozen=True)
class AssemblyPlan:
    """A compiled batch: selected bindings plus cacheable per-atom write-back."""

    bindings: tuple[Binding, ...]
    atom_patches: tuple[_AtomPatch, ...]

    def write_atoms(
        self,
        graph: Atomistic,
        created_sets: list[list[int]],
    ) -> None:
        """Replay only typifier-owned scalar fields onto the assembled graph."""
        added: dict[_AddedAtomRef, int] = {}
        for binding_index, created in enumerate(created_sets):
            for ordinal, handle in enumerate(created):
                added[_AddedAtomRef(binding_index, ordinal)] = handle

        atoms = {atom.handle: atom for atom in graph.atoms}
        for patch in self.atom_patches:
            handle = (
                patch.atom if isinstance(patch.atom, int) else added.get(patch.atom)
            )
            if handle is None:
                raise RuntimeError(
                    "compiled assembly expected reaction-created atom "
                    f"{patch.atom}, but execution did not return it"
                )
            atom = atoms.get(handle)
            if atom is None:
                raise RuntimeError(
                    f"compiled assembly refers to atom {handle}, but the reaction "
                    "deleted it during execution"
                )
            atom.update(**dict(patch.annotations))


@dataclass(frozen=True)
class _CachedPatch:
    """Annotations addressed by canonical position in a rooted local product."""

    key_graph: Atomistic
    root_positions: tuple[int, ...]
    atoms: tuple[tuple[int, tuple[tuple[str, Scalar], ...]], ...]


class LocalEnvironmentCache:
    """Memoise typifier output by rooted, chemically labelled local product."""

    _IDENTITY_FIELDS = frozenset(
        {
            "id",
            "mol_id",
            "res_id",
            "res_name",
            "site",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "fx",
            "fy",
            "fz",
        }
    )

    def __init__(self, typifier: molrs.Typifier) -> None:
        self._typifier = typifier
        self._buckets: dict[int, list[_CachedPatch]] = {}

    def patch(
        self,
        product: Atomistic,
        roots: set[int],
        targets: set[int],
    ) -> tuple[tuple[int, tuple[tuple[str, Scalar], ...]], ...]:
        """Return ``(local_handle, annotations)`` for this product's targets."""
        key_graph = self._key_graph(product, roots)
        canonical = key_graph.canonical_order()
        canonical_index = {handle: i for i, handle in enumerate(canonical)}
        root_positions = tuple(sorted(canonical_index[h] for h in roots))
        bucket = self._buckets.setdefault(key_graph.structural_hash(), [])

        for cached in bucket:
            if cached.root_positions == root_positions and key_graph.is_isomorphic(
                cached.key_graph
            ):
                return tuple(
                    (canonical[position], annotations)
                    for position, annotations in cached.atoms
                )

        captured = self._capture(product, canonical_index, targets)
        bucket.append(
            _CachedPatch(
                key_graph=key_graph,
                root_positions=root_positions,
                atoms=captured,
            )
        )
        return tuple(
            (canonical[position], annotations) for position, annotations in captured
        )

    def _capture(
        self,
        product: Atomistic,
        canonical_index: Mapping[int, int],
        targets: set[int],
    ) -> tuple[tuple[int, tuple[tuple[str, Scalar], ...]], ...]:
        before_atoms = list(product.atoms)
        before = {atom.handle: dict(atom.data) for atom in before_atoms}
        typed_raw = self._typifier.typify(product)
        typed = (
            typed_raw
            if isinstance(typed_raw, Atomistic)
            else Atomistic.adopt(typed_raw)
        )
        after_atoms = list(typed.atoms)
        if len(after_atoms) != len(before_atoms):
            raise ValueError(
                f"{type(self._typifier).__name__}.typify changed the atom count "
                f"from {len(before_atoms)} to {len(after_atoms)}; a typifier may "
                "annotate a graph but may not change its structure"
            )
        after = {atom.handle: dict(atom.data) for atom in after_atoms}
        if set(after) != set(before):
            # Native typifiers preserve handles today.  Positional recovery would
            # hide a broken graph-transform contract, so fail loudly instead.
            raise ValueError(
                f"{type(self._typifier).__name__}.typify did not preserve atom handles"
            )

        owned = self._scalar_delta(before, after)
        captured: list[tuple[int, tuple[tuple[str, Scalar], ...]]] = []
        for handle in targets:
            annotations = tuple(
                sorted(
                    (key, value)
                    for key in owned
                    if (value := self._scalar(after[handle].get(key))) is not None
                )
            )
            if fields.TYPE.key in owned and not any(
                key == fields.TYPE.key for key, _ in annotations
            ):
                raise ValueError(
                    "typifier left a planned product atom untyped at canonical "
                    f"position {canonical_index[handle]}"
                )
            captured.append((canonical_index[handle], annotations))
        captured.sort(key=lambda entry: entry[0])
        return tuple(captured)

    @classmethod
    def _scalar_delta(
        cls,
        before: Mapping[int, Mapping[str, object]],
        after: Mapping[int, Mapping[str, object]],
    ) -> frozenset[str]:
        owned: set[str] = set()
        for handle, post in after.items():
            pre = before[handle]
            for key, value in post.items():
                if cls._scalar(value) is not None and pre.get(key) != value:
                    owned.add(key)
        return frozenset(owned)

    @staticmethod
    def _scalar(value: object) -> Scalar | None:
        # bool is intentionally accepted: molrs has a native boolean column.
        return value if isinstance(value, (str, int, float, bool)) else None

    @classmethod
    def _key_graph(cls, product: Atomistic, roots: set[int]) -> Atomistic:
        """Encode all chemical scalar labels and rootedness into cache-only charge.

        molrs' graph hash already covers element, connectivity, bond order,
        aromaticity and charge.  A custom typifier may additionally inspect
        formal charge or another scalar atom label, so a cache key cannot safely
        use the structural hash alone.  The copy below folds every non-identity
        scalar label into a deterministic exactly-representable float.  It is
        never shown to the typifier.
        """
        keyed = product.copy()
        for atom in keyed.atoms:
            labels = tuple(
                sorted(
                    (key, value)
                    for key, raw in atom.data.items()
                    if key not in cls._IDENTITY_FIELDS
                    and (value := cls._scalar(raw)) is not None
                )
            )
            payload = repr((atom.handle in roots, labels)).encode()
            # 52 bits fit exactly in an IEEE-754 f64 integer mantissa.
            code = int.from_bytes(sha256(payload).digest()[:8], "little") & (
                (1 << 52) - 1
            )
            atom[fields.CHARGE] = float(code)
        return keyed


class AssemblyCompiler:
    """Compile one selector result into local product environments and patches."""

    def __init__(
        self,
        reaction: molrs.Reaction,
        typifier: molrs.Typifier | None,
        reach: int | None,
    ) -> None:
        self._reaction = reaction
        self._reach = reach
        self.cache = LocalEnvironmentCache(typifier) if typifier is not None else None

    def compile(
        self,
        world: Atomistic,
        bindings: list[Binding],
        labels: Mapping[int, str],
    ) -> AssemblyPlan:
        """Compile every possible junction before mutating ``world``."""
        frozen_bindings = tuple(dict(binding) for binding in bindings)
        if self.cache is None:
            return AssemblyPlan(frozen_bindings, ())

        assert self._reach is not None
        adjacency = self._planned_adjacency(world, bindings)
        atoms_by_handle = {atom.handle: atom for atom in world.atoms}
        atoms_by_residue: dict[int, set[int]] = {}
        for atom in atoms_by_handle.values():
            residue = atom.get(fields.RES_ID)
            if residue is not None:
                atoms_by_residue.setdefault(int(residue), set()).add(atom.handle)

        merged: dict[_AtomRef, dict[str, Scalar]] = {}
        for central_index, binding in enumerate(bindings):
            handles = self._motif_handles(
                binding,
                adjacency,
                atoms_by_handle,
                atoms_by_residue,
            )
            patches = self._compile_motif(
                world,
                handles,
                bindings,
                labels,
                central_index,
            )
            for atom_ref, atom_values in patches:
                current = merged.setdefault(atom_ref, {})
                for key, value in atom_values:
                    previous = current.get(key)
                    if previous is not None and previous != value:
                        raise RuntimeError(
                            "overlapping compiled environments disagree for atom "
                            f"{atom_ref}, field {key!r}: {previous!r} != {value!r}; "
                            "increase reach so every target has complete context"
                        )
                    current[key] = value

        atom_patches = tuple(
            _AtomPatch(atom, tuple(sorted(annotations.items())))
            for atom, annotations in merged.items()
        )
        return AssemblyPlan(frozen_bindings, atom_patches)

    def _compile_motif(
        self,
        world: Atomistic,
        handles: set[int],
        bindings: list[Binding],
        labels: Mapping[int, str],
        central_index: int,
    ) -> tuple[tuple[_AtomRef, tuple[tuple[str, Scalar], ...]], ...]:
        bare, old_to_local = world.induced_subgraph(sorted(handles))
        motif = Atomistic.adopt(bare)
        preexisting = frozenset(atom.handle for atom in motif.atoms)
        local_to_old = {local: old for old, local in old_to_local.items()}

        local_bindings: list[Binding] = []
        global_indices: list[int] = []
        for index, binding in enumerate(bindings):
            if set(binding.values()) <= handles:
                local_bindings.append(
                    {label: old_to_local[handle] for label, handle in binding.items()}
                )
                global_indices.append(index)
        try:
            central_local = global_indices.index(central_index)
        except ValueError as exc:  # pragma: no cover - guarded by motif centres
            raise RuntimeError(
                "central binding is absent from its compiled motif"
            ) from exc

        local_labels = {
            old_to_local[handle]: value
            for handle, value in labels.items()
            if handle in old_to_local
        }
        touched_sets, created_sets = self._reaction.apply_many_detailed(
            motif, local_bindings, local_labels, refresh=False
        )
        motif.generate_topology(
            gen_angle=True,
            gen_dihedral=True,
            clear_existing=True,
        )
        molrs.perceive_aromaticity(motif)

        ref_of_local: dict[int, _AtomRef] = {
            local: old for local, old in local_to_old.items() if local in preexisting
        }
        for local_index, created in enumerate(created_sets):
            global_index = global_indices[local_index]
            for ordinal, handle in enumerate(created):
                ref_of_local[handle] = _AddedAtomRef(global_index, ordinal)

        roots = set(touched_sets[central_local])
        targets = self._targets(motif, roots, self._reach or 0)
        # A raw atom-level cut has open valences.  A residue-expanded motif is a
        # union of complete user monomers and needs no synthetic completion.
        residue_complete = all(
            atom.get(fields.RES_ID) is not None for atom in motif.atoms
        )
        if not residue_complete:
            completed = Atomistic.adopt(molrs.Perceive().find_hydrogens(motif))
            motif = completed

        local_patches = self.cache.patch(motif, roots, targets)
        output: list[tuple[_AtomRef, tuple[tuple[str, Scalar], ...]]] = []
        for handle, atom_values in local_patches:
            atom_ref = ref_of_local.get(handle)
            if atom_ref is None:
                raise RuntimeError(
                    "typifier write-back selected a completion atom; the compiled "
                    "context shell is smaller than the declared reach"
                )
            output.append((atom_ref, atom_values))
        return tuple(output)

    def _motif_handles(
        self,
        binding: Binding,
        adjacency: Mapping[int, set[int]],
        atoms_by_handle: Mapping[int, object],
        atoms_by_residue: Mapping[int, set[int]],
    ) -> set[int]:
        # One reach is the write-back radius and one is the typifier's context
        # shell.  At least one hop is needed to carry an unmapped leaving atom.
        radius = max(2 * (self._reach or 0), 1)
        selected = self._ball(binding.values(), adjacency, radius)

        residues: set[int] = set()
        for handle in selected:
            atom = atoms_by_handle[handle]
            residue = atom.get(fields.RES_ID)  # type: ignore[attr-defined]
            if residue is None:
                return selected
            residues.add(int(residue))
        return set().union(*(atoms_by_residue[residue] for residue in residues))

    def _planned_adjacency(
        self, world: Atomistic, bindings: Iterable[Binding]
    ) -> dict[int, set[int]]:
        adjacency = {atom.handle: set() for atom in world.atoms}
        for bond in world.bonds:
            a, b = (atom.handle for atom in bond.endpoints)
            adjacency[a].add(b)
            adjacency[b].add(a)
        for binding in bindings:
            for map_a, map_b in self._reaction.forming_bonds:
                a, b = binding[map_a], binding[map_b]
                adjacency[a].add(b)
                adjacency[b].add(a)
        return adjacency

    @staticmethod
    def _ball(
        centers: Iterable[int], adjacency: Mapping[int, set[int]], radius: int
    ) -> set[int]:
        distance = {handle: 0 for handle in centers}
        queue = deque(distance)
        while queue:
            handle = queue.popleft()
            if distance[handle] >= radius:
                continue
            for neighbor in adjacency[handle]:
                if neighbor not in distance:
                    distance[neighbor] = distance[handle] + 1
                    queue.append(neighbor)
        return set(distance)

    @staticmethod
    def _targets(graph: Atomistic, roots: set[int], reach: int) -> set[int]:
        targets: set[int] = set()
        for root in roots:
            targets.update(
                handle for handle, _ in graph.topo_distances(root, max_hops=reach)
            )
        return targets
