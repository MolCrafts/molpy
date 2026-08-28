"""Every molpy submodule imports. Enumerated, not a hand list."""

from __future__ import annotations

import importlib
import pkgutil
import warnings

import molpy


def test_walk_packages_imports() -> None:
    failures: list[str] = []
    for mod in pkgutil.walk_packages(molpy.__path__, molpy.__name__ + "."):
        if mod.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(mod.name)
        except ModuleNotFoundError as exc:
            # Optional extras (rdkit, openbabel, …) are not a hard import.
            if exc.name in {"rdkit", "openbabel"}:
                continue
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — the point is to surface any miss
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "import failures:\n" + "\n".join(failures)


def test_import_molpy_and_md_are_silent() -> None:
    import sys

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        importlib.reload(molpy)
        sys.modules.pop("molpy.md", None)
        sys.modules.pop("molrs.md", None)
        importlib.import_module("molpy.md")
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert not fw
