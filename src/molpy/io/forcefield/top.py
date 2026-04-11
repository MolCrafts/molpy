import re
from pathlib import Path

from molpy.core.forcefield import ForceField


class GromacsTopReader:
    """Utility to read a Gromacs .top/.itp topology file into a dictionary.

    The returned mapping contains a key for every section header encountered
    (e.g. ``atomtypes``, ``moleculetype``), mapping to the raw **content lines**
    (with inline comments stripped) that appear under that section in the order
    they occur.
    """

    _SECTION_RE = re.compile(r"^\s*\[\s*([\w-]+)\s*]\s*$")
    _INCLUDE_RE = re.compile(r"^\s*#\s*include\s+(?:\"|<)([^\">]+)(?:\"|>)")

    def __init__(self, file: str | Path, include: bool = False):
        self.file = Path(file)
        self.include = include

    # NOTE: `forcefield` is optional and only used to help resolve #include paths
    def read(
        self,
        forcefield: ForceField,
        *,
        strip_comments: bool = True,
        recursive: bool = True,
    ) -> ForceField:
        """Parse the topology file.

        Parameters
        ----------
        forcefield:
            Optional object providing a ``base_dir`` or ``path`` attribute used
            to resolve relative ``#include`` statements that point inside the
            force-field directory (e.g. ``ff/amber14sb.ff/ions.itp``).
        strip_comments:
            Whether to remove text following a ``;`` (Gromacs comment
            delimiter). Leading/trailing whitespace is always removed.
        recursive:
            If *True* (default) ``#include``d files are parsed recursively. The
            content of an included file is *merged into* the dictionary being
            built; if the same section appears multiple times its content is
            *extended* in occurrence order.

        Returns
        -------
        dict[str, list[str]]
            Mapping from *lower‑case* section name to list of raw content lines.
        """
        result: dict[str, list[str]] = {}
        visited: set[Path] = set()
        self._parse_file(
            self.file, result, visited, forcefield, strip_comments, recursive
        )

        # parse atom sections
        self._parse_atom_section(result["atoms"], forcefield)
        # parse bond sections
        self._parse_bond_section(result["bonds"], forcefield)
        # parse angle sections
        self._parse_angle_section(result["angles"], forcefield)
        # parse dihedral sections
        self._parse_dihedral_section(result["dihedrals"], forcefield)

        # parse pair sections
        self._parse_pair_section(result["pairs"], forcefield)

        return forcefield

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _parse_file(
        self,
        path: Path,
        store: dict[str, list[str]],
        visited: set[Path],
        forcefield: ForceField | None,
        strip_comments: bool,
        recursive: bool,
    ) -> None:
        if path in visited:
            return  # Prevent infinite include loops
        visited.add(path)

        cwd = path.parent
        with path.open("r", encoding="utf-8") as fh:
            current_section: str | None = None
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                # Handle comments stripping
                if strip_comments:
                    line = line.split(";", 1)[0]
                line = line.strip()
                if not line:
                    continue  # Skip blank/comment lines

                # Pre‑processor directives
                if line.startswith("#"):
                    # Handle #include
                    m_inc = self._INCLUDE_RE.match(line)
                    if self.include and m_inc and recursive:
                        inc_file = Path(m_inc.group(1))
                        resolved = self._resolve_include(inc_file, cwd, forcefield)
                        if resolved and resolved.exists():
                            self._parse_file(
                                resolved,
                                store,
                                visited,
                                forcefield,
                                strip_comments,
                                recursive,
                            )
                        else:
                            raise FileNotFoundError(
                                f"Could not resolve include '{inc_file}' from '{path}'."
                            )
                    # Other pre‑processor directives (#ifdef, #define, etc.)
                    # are safely ignored for this simple reader.
                    continue

                # Section header
                m_sec = self._SECTION_RE.match(line)
                if m_sec:
                    current_section = m_sec.group(1).lower()
                    store.setdefault(current_section, [])
                    continue

                # Normal content line
                if current_section is None:
                    # Lines appearing before the first section are gathered
                    # under a pseudo‑section named '__preamble__'.
                    current_section = "__preamble__"
                    store.setdefault(current_section, [])
                store[current_section].append(line)

    # --------------------------------------------------------------------- #
    def _resolve_include(
        self, inc: Path, cwd: Path, forcefield: ForceField | None
    ) -> Path:
        """Return an absolute path for an included file.

        Search order:
            1. If *inc* is absolute, return it directly.
            2. Relative to the directory of the file that contains the include.
            3. Inside ``forcefield.base_dir`` or ``forcefield.path`` if given.
        """
        if inc.is_absolute():
            return inc
        candidate = cwd / inc
        if candidate.exists():
            return candidate
        # Try forcefield dir
        if forcefield is not None:
            for attr in ("base_dir", "path"):
                base = getattr(forcefield, attr, None)
                if base:
                    base_path = Path(base) / inc
                    if base_path.exists():
                        return base_path
        # Fallback: return path relative to cwd even if it doesn't exist
        return candidate

    def _parse_atom_section(self, lines: list[str], ff: ForceField) -> None:
        """Parse the [atomtypes] section of a Gromacs topology file.

        Parameters
        ----------
        lines : list[str]
            Lines of the [atomtypes] section.
        ff : object
            Force field object to populate with atom types.

        Returns
        -------
        None
        """
        from molpy.core.forcefield import AtomStyle

        atomstyle = ff.def_style(AtomStyle("full"))

        header = [
            "nr",
            "name",
            "resnr",
            "residu",
            "atom",
            "cgnr",
            "charge",
            "mass",
            "typeB",
            "chargeB",
            "massB",
        ]
        atomtypes = []
        for line in map(lambda l: l.split(), lines):
            data = dict(zip(header, line))
            at = atomstyle.def_type(data.pop("name"), **data)
            atomtypes.append(at)

        self.atomtypes = atomtypes

    def _parse_bond_section(self, lines: list[str], ff: ForceField) -> None:
        """Parse the [bondtypes] section of a GROMACS topology file."""

        func_types = {
            "1": "harmonic",  # kb (kJ mol‑1 nm‑2)  r0 (nm)
            "2": "G96",  # same params as harmonic but G96 functional form
            "3": "morse",  # r0  De  alpha
            "4": "cubic",  # r0  k2  k3  k4  (rare)
        }

        param_specs = {
            "harmonic": ("r0", "k"),
            "G96": ("r0", "k"),
            "morse": ("r0", "De", "alpha"),
            "cubic": ("r0", "k2", "k3", "k4"),
        }

        for raw in lines:
            # strip comments and whitespace
            raw = raw.split(";")[0].strip()
            if not raw:
                continue  # blank or comment line

            cols = raw.split()
            i, j, funct = cols[:3]
            params = list(map(float, cols[3:]))

            style_name = func_types.get(funct)
            if style_name is None:
                raise ValueError(f"Unknown bond funct '{funct}' in line: {raw}")

            from molpy.core.forcefield import BondStyle

            bondstyle = ff.def_style(BondStyle(style_name))

            itype = self.atomtypes[int(i) - 1]
            jtype = self.atomtypes[int(j) - 1]

            # Use a canonical order for the style name (e.g., CA-CB)
            name = f"{itype.name}-{jtype.name}"

            param_names = param_specs[style_name]
            param_dict = {n: v for n, v in zip(param_names, params)}

            # Register the bond type in the force‑field object
            bondstyle.def_type(itype, jtype, name=name, **param_dict)

    def _parse_angle_section(self, lines: list[str], ff: ForceField) -> None:
        """Parse the [angletypes] section of a GROMACS topology file."""

        func_types = {
            "1": "harmonic",  # theta0  k
            "2": "G96",  # theta0  k  (G96 quadratic)
            "3": "quartic",  # c0 c1 c2 c3
            "4": "ub",  # theta0 k  r0 k_ub  (Urey–Bradley)
        }

        param_specs = {
            "harmonic": ("theta0", "k"),
            "G96": ("theta0", "k"),
            "quartic": ("c0", "c1", "c2", "c3"),
            "ub": ("theta0", "k", "r0", "k_ub"),
        }

        for raw in lines:
            raw = raw.split(";")[0].strip()
            if not raw:
                continue

            cols = raw.split()
            i, j, k, funct = cols[:4]
            params = list(map(float, cols[4:]))

            style_name = func_types.get(funct)
            if style_name is None:
                raise ValueError(f"Unknown angle funct '{funct}' in line: {raw}")

            from molpy.core.forcefield import AngleStyle

            anglestyle = ff.def_style(AngleStyle(style_name))

            itype = self.atomtypes[int(i) - 1]
            jtype = self.atomtypes[int(j) - 1]
            ktype = self.atomtypes[int(k) - 1]

            name = f"{itype.name}-{jtype.name}-{ktype.name}"
            param_names = param_specs[style_name]
            param_dict = {n: v for n, v in zip(param_names, params)}

            anglestyle.def_type(itype, jtype, ktype, name=name, **param_dict)

    def _parse_dihedral_section(self, lines: list[str], ff: ForceField) -> None:
        """Parse the [dihedraltypes] section of a GROMACS topology file."""

        func_types = {
            "1": "periodic",  # phi0  k  multiplicity
            "2": "rb",  # c0 c1 c2 c3 c4 c5  (Ryckaert–Bellemans)
            "3": "harmonic",  # psi0  k  (improper-like, but some force fields use it for proper)
        }

        param_specs = {
            "periodic": ("phi0", "k", "n"),
            "rb": ("c0", "c1", "c2", "c3", "c4", "c5"),
            "harmonic": ("psi0", "k"),
        }

        for raw in lines:
            raw = raw.split(";")[0].strip()
            if not raw:
                continue

            cols = raw.split()
            i, j, k, l, funct = cols[:5]
            params = list(map(float, cols[5:]))

            style_name = func_types.get(funct)
            if style_name is None:
                raise ValueError(f"Unknown dihedral funct '{funct}' in line: {raw}")

            from molpy.core.forcefield import DihedralStyle

            dihstyle = ff.def_style(DihedralStyle(style_name))

            itype = self.atomtypes[int(i) - 1]
            jtype = self.atomtypes[int(j) - 1]
            ktype = self.atomtypes[int(k) - 1]
            ltype = self.atomtypes[int(l) - 1]

            name = f"{itype.name}-{jtype.name}-{ktype.name}-{ltype.name}"
            param_names = param_specs[style_name]
            param_dict = {n: v for n, v in zip(param_names, params)}

            dihstyle.def_type(itype, jtype, ktype, ltype, name=name, **param_dict)

    def _parse_pair_section(self, lines: list[str], ff: ForceField) -> None:
        """Parse the [pairtypes] or [nonbond_params] section."""

        func_types = {
            "1": "lj12-6",
            "2": "buckingham",
        }
        param_specs = {
            "lj12-6": ("c6", "c12"),
            "buckingham": ("A", "B", "C"),
        }

        for raw in lines:
            raw = raw.split(";")[0].strip()
            if not raw:
                continue
            cols = raw.split()
            i, j, funct = cols[:3]
            params = list(map(float, cols[3:]))

            style_name = func_types.get(funct)
            if style_name is None:
                raise ValueError(f"Unknown pair funct '{funct}' in line: {raw}")
            param_names = param_specs[style_name]

            from molpy.core.forcefield import PairStyle

            pairstyle = ff.def_style(PairStyle(style_name))
            itype = self.atomtypes[int(i) - 1]
            jtype = self.atomtypes[int(j) - 1]

            pairstyle.def_type(
                itype,
                jtype,
                name=f"{itype.name}-{jtype.name}",
                **{n: v for n, v in zip(param_names, params)},
            )


# ============================================================================
# Writer
# ============================================================================

_BOND_FUNC: dict[str, str] = {"harmonic": "1", "G96": "2", "morse": "3", "cubic": "4"}
_ANGLE_FUNC: dict[str, str] = {"harmonic": "1", "G96": "2", "quartic": "3", "ub": "4"}
_DIHEDRAL_FUNC: dict[str, str] = {"periodic": "1", "rb": "2", "harmonic": "3"}
_PAIR_FUNC: dict[str, str] = {"lj12-6": "1", "buckingham": "2"}

_BOND_PARAMS: dict[str, tuple[str, ...]] = {
    "harmonic": ("r0", "k"),
    "G96": ("r0", "k"),
    "morse": ("r0", "De", "alpha"),
    "cubic": ("r0", "k2", "k3", "k4"),
}
_ANGLE_PARAMS: dict[str, tuple[str, ...]] = {
    "harmonic": ("theta0", "k"),
    "G96": ("theta0", "k"),
    "quartic": ("c0", "c1", "c2", "c3"),
    "ub": ("theta0", "k", "r0", "k_ub"),
}
_DIHEDRAL_PARAMS: dict[str, tuple[str, ...]] = {
    "periodic": ("phi0", "k", "n"),
    "rb": ("c0", "c1", "c2", "c3", "c4", "c5"),
    "harmonic": ("psi0", "k"),
}
_PAIR_PARAMS: dict[str, tuple[str, ...]] = {
    "lj12-6": ("c6", "c12"),
    "buckingham": ("A", "B", "C"),
}


class GromacsForceFieldWriter:
    """Write a ForceField to GROMACS ``.top`` / ``.itp`` format.

    The output is roundtrip-compatible with :class:`GromacsTopReader`.

    Args:
        filepath: Destination path.
        precision: Number of decimal digits for floating-point values.
    """

    def __init__(self, filepath: str | Path, precision: int = 6) -> None:
        self._file = Path(filepath)
        self._prec = precision

    def write(self, forcefield: ForceField) -> None:
        """Serialize *forcefield* to GROMACS topology format."""
        from molpy.core.forcefield import (
            AngleStyle,
            AngleType,
            AtomStyle,
            AtomType,
            BondStyle,
            BondType,
            DihedralStyle,
            DihedralType,
            PairStyle,
            PairType,
        )

        atom_types = forcefield.get_types(AtomType)
        # Build index mapping (1-based, matching reader convention)
        at_index: dict[int, int] = {id(at): i + 1 for i, at in enumerate(atom_types)}

        with open(self._file, "w", encoding="utf-8") as f:
            f.write("; Generated by MolPy\n\n")

            # [atoms]
            if atom_types:
                f.write("[ atoms ]\n")
                f.write("; nr  name  resnr  residu  atom  cgnr  charge  mass\n")
                for idx, at in enumerate(atom_types, 1):
                    kw = at.params.kwargs
                    nr = kw.get("nr", str(idx))
                    name = at.name
                    resnr = kw.get("resnr", "0")
                    residu = kw.get("residu", "LIG")
                    atom = kw.get("atom", name)
                    cgnr = kw.get("cgnr", "1")
                    charge = self._fmt(float(kw.get("charge", 0.0)))
                    mass = self._fmt(float(kw.get("mass", 0.0)))
                    f.write(
                        f"  {nr:>4}  {name:<5} {resnr:>5}  {residu:<5} {atom:<5} {cgnr:>4}  {charge}  {mass}\n"
                    )
                f.write("\n")

            # [bonds]
            bond_types = forcefield.get_types(BondType)
            if bond_types:
                f.write("[ bonds ]\n")
                f.write("; i  j  func  params...\n")
                for bt in bond_types:
                    i = at_index.get(id(bt.itom), 0)
                    j = at_index.get(id(bt.jtom), 0)
                    # Find style name
                    style_name = self._find_style_name(forcefield, BondStyle, bt)
                    funct = _BOND_FUNC.get(style_name, "1")
                    params = self._extract_params(bt, _BOND_PARAMS.get(style_name, ()))
                    f.write(f"  {i}  {j}  {funct}  {params}\n")
                f.write("\n")

            # [angles]
            angle_types = forcefield.get_types(AngleType)
            if angle_types:
                f.write("[ angles ]\n")
                f.write("; i  j  k  func  params...\n")
                for at in angle_types:
                    i = at_index.get(id(at.itom), 0)
                    j = at_index.get(id(at.jtom), 0)
                    k = at_index.get(id(at.ktom), 0)
                    style_name = self._find_style_name(forcefield, AngleStyle, at)
                    funct = _ANGLE_FUNC.get(style_name, "1")
                    params = self._extract_params(at, _ANGLE_PARAMS.get(style_name, ()))
                    f.write(f"  {i}  {j}  {k}  {funct}  {params}\n")
                f.write("\n")

            # [dihedrals]
            dihedral_types = forcefield.get_types(DihedralType)
            if dihedral_types:
                f.write("[ dihedrals ]\n")
                f.write("; i  j  k  l  func  params...\n")
                for dt in dihedral_types:
                    i = at_index.get(id(dt.itom), 0)
                    j = at_index.get(id(dt.jtom), 0)
                    k = at_index.get(id(dt.ktom), 0)
                    l_ = at_index.get(id(dt.ltom), 0)
                    style_name = self._find_style_name(forcefield, DihedralStyle, dt)
                    funct = _DIHEDRAL_FUNC.get(style_name, "1")
                    params = self._extract_params(
                        dt, _DIHEDRAL_PARAMS.get(style_name, ())
                    )
                    f.write(f"  {i}  {j}  {k}  {l_}  {funct}  {params}\n")
                f.write("\n")

            # [pairs]
            pair_types = forcefield.get_types(PairType)
            if pair_types:
                f.write("[ pairs ]\n")
                f.write("; i  j  func  params...\n")
                for pt in pair_types:
                    i = at_index.get(id(pt.itom), 0)
                    j = at_index.get(id(pt.jtom), 0)
                    style_name = self._find_style_name(forcefield, PairStyle, pt)
                    funct = _PAIR_FUNC.get(style_name, "1")
                    params = self._extract_params(pt, _PAIR_PARAMS.get(style_name, ()))
                    f.write(f"  {i}  {j}  {funct}  {params}\n")
                f.write("\n")

    def _fmt(self, v: float | int) -> str:
        if isinstance(v, float):
            return f"{v:.{self._prec}f}"
        return str(v)

    def _extract_params(self, typ: object, param_names: tuple[str, ...]) -> str:
        kw = typ.params.kwargs  # type: ignore[attr-defined]
        vals = [kw.get(name, 0.0) for name in param_names]
        return "  ".join(self._fmt(v) for v in vals)

    @staticmethod
    def _find_style_name(forcefield: ForceField, style_class: type, typ: object) -> str:
        """Find the style name that contains *typ*."""
        from molpy.core.forcefield import Type

        for style in forcefield.get_styles(style_class):
            if typ in style.types.bucket(Type):
                return style.name
        return "harmonic"
