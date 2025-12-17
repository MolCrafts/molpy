"""External integration layer for MolPy.

This module provides:
- **Adapters**: Bidirectional synchronization between MolPy's internal data structures
  and external library representations (e.g., RDKit, OpenMM).
- **Wrappers**: Minimal shells around external binaries and CLIs (e.g., antechamber, tleap).

**Architecture:**
- Adapter: Keeps MolPy ↔ external data structures in sync
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)
- Compute: Domain logic and operation orchestration (uses adapters and wrappers)

**Example Usage:**

Adapters:
    >>> from molpy.external import RDKitAdapter, Generate3D
    >>> adapter = RDKitAdapter(internal=atomistic)
    >>> generate_3d = Generate3D(add_hydrogens=True, embed=True)
    >>> adapter = generate_3d(adapter)
    >>> updated_atomistic = adapter.get_internal()

Wrappers:
    >>> from molpy.external import AntechamberWrapper, TLeapWrapper
    >>> from pathlib import Path
    >>>
    >>> # Antechamber wrapper
    >>> ante = AntechamberWrapper(name="antechamber", workdir=Path("tmp_ante"))
    >>> proc = ante.run_raw(["-i", "lig.mol2", "-fi", "mol2", "-o", "out.mol2", "-fo", "mol2"])
    >>>
    >>> # TLeap wrapper
    >>> tleap = TLeapWrapper(name="tleap", workdir=Path("tmp_tleap"))
    >>> script = "source leaprc.gaff\nquit\n"
    >>> proc = tleap.run_script(script_text=script)
"""

from .antechamber_wrapper import AntechamberWrapper

# Adapter exports
from .base import Adapter

# RDKit is an optional dependency. Keep imports lazy/guarded so that
# `import molpy` works without rdkit installed.
try:  # pragma: no cover
    from .rdkit_adapter import MP_ID, RDKitAdapter
    from .rdkit_compute import Generate3D, OptimizeGeometry

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing
    _HAS_RDKIT = False
    MP_ID = None  # type: ignore[assignment]
    RDKitAdapter = None  # type: ignore[assignment]
    Generate3D = None  # type: ignore[assignment]
    OptimizeGeometry = None  # type: ignore[assignment]

from .tleap_wrapper import TLeapWrapper

# Wrapper exports
from .wrapper import Wrapper

__all__ = [
    "Adapter",
    "Wrapper",
    "AntechamberWrapper",
    "TLeapWrapper",
]

if _HAS_RDKIT:
    __all__ += [
        "RDKitAdapter",
        "MP_ID",
        "Generate3D",
        "OptimizeGeometry",
    ]
