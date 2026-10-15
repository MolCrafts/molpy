"""molrs — Rust-backed molecular simulation primitives.

**The top level is the subsystems, plus the core data classes a user handles
directly.** Those classes are the same objects as :mod:`molrs.core`'s
(``molrs.Frame is molrs.core.Frame``): ``Frame``,
``Block``, ``Trajectory``, ``Box``, ``MolGraph``, ``Atomistic``,
``CoarseGrain``, the entity classes (``Atom``, ``Bond``, ``Angle``,
``Dihedral``, ``Improper``, ``Bead``, ``CgBond``, ``Port``, ``VirtualSite``,
``DrudeParticle``, ``MasslessSite``), ``Element`` and ``Topology``.

No function, algorithm, region or unit preset is promoted. Everything else
lives in the module named after its Rust owner, so that path and the Rust
path are the same words:

* :mod:`molrs.core` — the core data model: the column store and the frame
  (``Block``, ``Frame``, ``Trajectory``), space (``Box``, neighbour search,
  regions, meshes, point paths), the molecular graph (``MolGraph``,
  ``Atomistic``, ``CoarseGrain`` and their live views), elements and the
  unit engine; the vocabularies :mod:`molrs.core.keys`,
  :mod:`molrs.core.schema` and :mod:`molrs.core.constants`.
* :mod:`molrs.op` — pure numeric base: weighted superposition, centroids.
* :mod:`molrs.perceive` — chemical perception: rings, aromaticity,
  hydrogens, stereochemistry, SMARTS matching and reactions, coarse-grained
  bead-pattern matching.
* :mod:`molrs.io` — every file reader and writer: structure, trajectory and
  force-field files, SMILES and CGsmiles text, ``*.mrec`` records, frame
  bytes.
* :mod:`molrs.ff` — force fields, one submodule per Rust owner
  (``forcefield``, ``potential``, ``typifier``, ``charge``, ``ir``,
  ``params``, ``clpol_scaling``).
* :mod:`molrs.optimize` — geometry optimizers.
* :mod:`molrs.md` — in-process molecular dynamics: the integrators and the
  ``MdDriver``; it integrates potentials, it defines none.
* :mod:`molrs.conformer` — 3D conformer generation.
* :mod:`molrs.builder` — structure builders, site-graph assembly,
  coarse-graining.
* :mod:`molrs.compute` — trajectory analysis.
* :mod:`molrs.stream` — live Frame streaming (the transport).

A name that is not one of those handled classes has exactly one spelling —
``molrs.io.smiles.SmilesIr`` and nothing else — so there is one thing to
learn, document, and grep for.
"""

from . import (
    builder,
    compute,
    conformer,
    core,
    ff,
    io,
    md,
    op,
    optimize,
    perceive,
    stream,
)
from ._native import __version__ as __version__

# FFI ABI handshake, read by name by downstream handle-bridge extensions
# (molpack) at their import time.
from ._native import _ffi_abi_token  # noqa: F401

# The core data classes a user handles directly. Same objects as molrs.core.
from .core import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Block,
    Bond,
    Box,
    CgBond,
    CoarseGrain,
    Dihedral,
    DrudeParticle,
    Element,
    Frame,
    Improper,
    MasslessSite,
    MolGraph,
    Port,
    Topology,
    Trajectory,
    VirtualSite,
)

__all__ = [
    "builder",
    "compute",
    "conformer",
    "core",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "stream",
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
    "Bond",
    "Box",
    "CgBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "Frame",
    "Improper",
    "MasslessSite",
    "MolGraph",
    "Port",
    "Topology",
    "Trajectory",
    "VirtualSite",
]
