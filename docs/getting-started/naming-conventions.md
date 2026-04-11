## Naming Conventions in MolPy

### What the convention is

MolPy adopts a strict and explicit naming convention that operates at two levels: namespace organization and field naming. At the namespace level, molecular data is organized into semantic groupings such as `atoms`, `bonds`, `angles`, and `dihedrals`. Each namespace corresponds to a physical or topological category and acts as a first-level key in the data hierarchy. At the field level, individual properties are accessed as second-level keys under their parent namespace, using names that distinguish between integer atom indices and object references.

MolPy intentionally supports two complementary representations of molecular topology, each optimized for different stages of a molecular modeling workflow. The Frame and Block representation serves as the data interchange layer, optimized for serialization, storage, numerical processing, and interoperability with external simulation engines. The Entity representation serves as the interactive construction layer, optimized for structure building, graph traversal, and chemistry-aware manipulation. Although both representations describe the same physical topology, they operate at different abstraction levels and therefore use distinct naming schemes to avoid semantic confusion.

At the Frame and Block level, topology is expressed purely in terms of integer atom indices. These indices are always 0-based and refer to rows in the atoms block. For this representation, MolPy uses the field names `atomi`, `atomj`, `atomk`, and `atoml`. For example, a bond stored in a frame is represented as:

```python
frame["bonds"] = Block({
    "type": ["C-H"],
    "atomi": [0],
    "atomj": [1],
})
```

Here, `atomi` and `atomj` are integers that refer to positions in the atoms block. A strict rule applies: `atomi`, `atomj`, `atomk`, and `atoml` must always store integers and must never store Atom objects.

### Frame Schema with Namespaces

The following sections define the complete schema for Frame and Block structures. Each namespace groups related fields according to their physical meaning and usage patterns. Fields are accessed using string keys for both namespace and field, such as `frame["atoms"]` to retrieve the atoms Block, or `frame["atoms"]["x"]` to access positions directly.

#### Atomic Properties (`atoms`)

The `atoms` namespace contains per-atom properties, including atomic numbers, positions, charges, and optional identifiers. All arrays within this Block have length `N`, where `N` is the number of atoms. Atomic positions are stored as three separate 1D arrays (`x`, `y`, `z`) of length `N`, which is the standard convention used by MolPy readers and expected by downstream code (e.g., potential energy calculations).

| Field | Type | Description |
|-------|------|-------------|
| `id` | int array | Atom IDs (1-indexed, optional) |
| `mol_id` | int array | Molecule IDs (1-indexed, optional) |
| `number` | int array | Atomic numbers (optional) |
| `element` | string array | Element symbols (optional) |
| `type` | int or string array | Atom types (optional) |
| `mass` | float array | Atomic masses in amu (optional) |
| `charge` | float array | Partial charges in elementary charge units (optional) |
| `x` | float array (N) | Atomic x-coordinates |
| `y` | float array (N) | Atomic y-coordinates |
| `z` | float array (N) | Atomic z-coordinates |
| `vx` | float array (N) | Atomic x-velocities (optional) |
| `vy` | float array (N) | Atomic y-velocities (optional) |
| `vz` | float array (N) | Atomic z-velocities (optional) |
| `res_id` | int array | Residue IDs (optional) |
| `res_name` | string array | Residue names (optional) |

Format-specific aliases such as LAMMPS `q` and `mol` exist only at the I/O boundary. Readers canonicalize them to `charge` and `mol_id`; writers localize them back when required by the target format.

#### Bond Topology (`bonds`)

The `bonds` namespace stores bond connectivity using separate index arrays for source and target atoms. This design simplifies indexing and aligns with the naming convention used in Entity-level representations. All arrays have length `E`, where `E` is the number of bonds.

| Field | Type | Description |
|-------|------|-------------|
| `atomi` | int array | Bond source atom indices |
| `atomj` | int array | Bond target atom indices |
| `type` | string or int array | Bond types (optional) |

#### Angle Topology (`angles`)

The `angles` namespace represents three-body interactions. Atom indices are stored separately as `atomi`, `atomj`, and `atomk`, where `atomj` is the center atom. All arrays have length `A`, where `A` is the number of angles.

| Field | Type | Description |
|-------|------|-------------|
| `atomi` | int array | First atom index |
| `atomj` | int array | Center atom index |
| `atomk` | int array | Third atom index |
| `type` | string or int array | Angle types (optional) |

#### Dihedral Topology (`dihedrals`)

The `dihedrals` namespace represents four-body torsional interactions. Atom indices are stored separately as `atomi`, `atomj`, `atomk`, and `atoml`, following the same convention as the `angles` namespace. All arrays have length `D`, where `D` is the number of dihedrals.

| Field | Type | Description |
|-------|------|-------------|
| `atomi` | int array | First atom index |
| `atomj` | int array | Second atom index |
| `atomk` | int array | Third atom index |
| `atoml` | int array | Fourth atom index |
| `type` | string or int array | Dihedral types (optional) |

#### Improper Topology (`impropers`)

The `impropers` namespace represents improper dihedral interactions, typically used to enforce planarity or chirality constraints. The index naming follows the same convention as `dihedrals`.

| Field | Type | Description |
|-------|------|-------------|
| `atomi` | int array | First atom index |
| `atomj` | int array | Second atom index |
| `atomk` | int array | Third atom index |
| `atoml` | int array | Fourth atom index |
| `type` | string or int array | Improper types (optional) |

### Namespace Naming Conventions

Namespace names use lowercase and typically use plural forms (such as `atoms`, `bonds`, `angles`, `dihedrals`, `impropers`) to indicate that they contain collections of related items. This convention makes namespace keys self-documenting and consistent across the codebase. Namespaces are accessed using string keys: `frame["atoms"]` retrieves the atoms Block, which can then be indexed again with field names such as `frame["atoms"]["x"]` for positions.

Field names within a namespace use lowercase with underscore separators for multi-word names. Topology index fields use the pattern `atomi`, `atomj`, `atomk`, `atoml` to indicate their role (source, target, center, endpoints) while maintaining a consistent suffix-free naming that distinguishes them from entity-level object references.

### Entity-Level Topology: Object References

At the Entity level, MolPy's topology objects (Bond, Angle, Dihedral, Improper) operate directly on Atom instances. These objects are used during molecular construction, editing, and chemical reasoning. For entity-level topology, MolPy uses the field names `itom`, `jtom`, `ktom`, and `ltom`, which correspond to the Frame-level `atomi`, `atomj`, `atomk`, and `atoml` but store object references instead of integer indices. For example:

```python
bond = mol.def_bond(atom1, atom2)
print(bond.itom)  # Atom object
print(bond.jtom)  # Atom object
```

In this context, `itom` and `jtom` are explicit references to Atom objects, not indices. The naming is intentionally short because these fields are accessed frequently during structure manipulation, and the `tom` suffix signals that the value is an object rather than a numeric identifier. A corresponding strict rule applies: `itom`, `jtom`, `ktom`, and `ltom` must always store Atom references and must never store integers.

This parallel naming scheme ensures that the semantic role of each atom (first, second, center, etc.) remains consistent across both representations, while the naming itself (`atomi` versus `itom`) makes the representation explicit at every use site.

### Why the convention exists

The namespace and naming convention addresses two complementary problems: data organization and type safety. At the organization level, namespaces eliminate ambiguity about where a field belongs and what category of data it represents. Without explicit namespaces, field names must encode their semantic category through prefixes or suffixes, leading to inconsistent conventions such as `atom_z` versus `z_atom` versus `atomic_number` versus `element`. Namespaces make the category explicit and separate it from the field identity, so atomic numbers are always accessed as `frame["atoms"]["number"]` regardless of context.

At the type safety level, MolPy deliberately avoids using the same field name for both indices and references. While names such as `atom_i` or `atom1` are common in other libraries, they tend to blur the distinction between "position in a table" and "object in memory". By using `atomi` and `atomj` at the Frame level and `itom` and `jtom` at the Entity level, the code makes this distinction visible at the point of use and enforceable through type checking or runtime validation.

This separation prevents subtle bugs where a field silently changes meaning depending on context. Such bugs become especially problematic when adding serialization, caching, multiprocessing, or cross-language bindings, since the interchange format must operate on indices while the in-process API operates on object references. The Frame-level design is fully serializable to JSON, Arrow, and HDF5, and it is language-agnostic, making it straightforward to implement equivalent data structures in Rust, C++, or TypeScript. It aligns naturally with the data layouts expected by MD engines such as LAMMPS, which operate on indexed atom tables rather than pointer-based object graphs.

At the same time, the Entity-level design supports the fluent and expressive APIs needed for interactive molecular construction, where graph traversal and chemical reasoning require direct access to atom properties without repeated index lookups. By keeping the two layers separate and enforcing explicit conversion at boundaries, MolPy avoids the common pitfall of "universal" data structures that try to serve both purposes but end up being awkward for both.

### How to convert between representations

MolPy provides explicit conversion paths between these two representations. When converting from Entity to Frame (for example, via `Atomistic.to_frame()`), each `itom`, `jtom`, and so on is replaced by the corresponding atom index, and the results are stored in namespace-organized Blocks. The resulting frame uses `atomi`, `atomj`, and so on exclusively within their respective namespace blocks. Conceptually, this works as:

```python
# Entity to Frame conversion for bonds
atomi_list = [atoms.index(bond.itom) for bond in bonds]
atomj_list = [atoms.index(bond.jtom) for bond in bonds]

frame["bonds"] = Block({
    "atomi": atomi_list,
    "atomj": atomj_list,
})
```

When converting from Frame to Entity, each index is resolved back to an Atom object using the atoms container. The namespace structure in the Frame guides which Entity types to construct:

```python
# Frame to Entity conversion for bonds
for i in range(len(frame["bonds"]["atomi"])):
    atomi = frame["bonds"]["atomi"][i]
    atomj = frame["bonds"]["atomj"][i]
    bond = Bond(itom=atoms[atomi], jtom=atoms[atomj])
```

These conversions must be explicit and localized at the boundary between Frame and Entity layers. Mixing the two representations inside the same object is not allowed. The namespace structure in Frame ensures that all related fields (such as bond indices and bond types) are kept together during conversion, simplifying the logic and reducing the chance of misalignment errors.


### Implementation guidelines for developers and agents

When refactoring or extending the MolPy codebase, treat this naming scheme as a hard invariant. Any topology stored inside a Frame or Block must use `atomi`, `atomj`, `atomk`, and `atoml`. Any topology stored inside an Entity object must use `itom`, `jtom`, `ktom`, and `ltom`. If existing code uses ambiguous names (such as `atom_i`, `atom1`, or reuses `itom` to store indices), split the representation and introduce an explicit conversion step at the layer boundary. Lightweight runtime checks or type hints are encouraged at constructors to ensure misuse is caught early.

This convention slightly increases verbosity at conversion boundaries, but it pays off in clarity, correctness, and long-term maintainability. It allows MolPy to scale to larger systems and workflows without semantic drift, support serialization and cross-language backends cleanly, and remain friendly to both interactive chemistry workflows and high-throughput numerical pipelines. Most importantly, it makes the meaning of topology fields obvious from the name alone, which is a critical property for a library intended to be extended by both humans and automated agents.
