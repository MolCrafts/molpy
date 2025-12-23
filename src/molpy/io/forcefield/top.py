import re
from pathlib import Path

from molpy.core.forcefield import ForceField


class GromacsTopReader:
    """Utility to read a Gromacs .top/.itp topology file into a dictionary.

    The returned mapping contains a key for every section header encountered
    (e.g. ``atomtypes``, ``moleculetype``), mapping to the raw **content lines**
    (with inline comments stripped) that appear under that section in the order
    they occur.

    Example
    -------
    >>> reader = GromacsTopReader('topol.top')
    >>> sections = reader.read()
    >>> sections['system']
    ['My simulation box']
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
