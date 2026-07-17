"""Builder-wide unit-test layout and orthogonality gates."""

import ast
from pathlib import Path


class TestBuilderTestLayout:
    SOURCE = Path(__file__).resolve().parents[2] / "src" / "molpy" / "builder"
    TESTS = Path(__file__).resolve().parent

    @classmethod
    def _test_file(cls, source: Path) -> Path | None:
        relative = source.relative_to(cls.SOURCE)
        if relative.parts[0] in {"assembly", "nanostructure"}:
            return None  # Each package owns a stricter local gate.
        if relative.parts[:2] == ("polymer", "ambertools"):
            return None
        if relative.parts[0] == "polymer":
            return cls.TESTS / f"test_polymer_{source.stem}.py"
        return cls.TESTS / f"test_{source.stem.removeprefix('_')}.py"

    def test_root_and_polymer_modules_have_mirrored_test_files(self):
        for source in self.SOURCE.rglob("*.py"):
            if source.name == "__init__.py":
                continue
            test_file = self._test_file(source)
            if test_file is not None:
                assert test_file.exists(), f"{source} lacks {test_file}"

    def test_each_production_class_has_its_own_test_class(self):
        for source in self.SOURCE.rglob("*.py"):
            if source.name == "__init__.py":
                continue
            test_file = self._test_file(source)
            if test_file is None:
                continue
            production = {
                node.name
                for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef)
            }
            tests = {
                node.name
                for node in ast.parse(test_file.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            }
            expected = {f"Test{name.removeprefix('_')}" for name in production}
            assert expected <= tests, f"{source.name}: {sorted(expected - tests)}"

    def test_unit_modules_contain_no_e2e_markers(self):
        for test_file in self.TESTS.glob("test_*.py"):
            if test_file == Path(__file__):
                continue
            text = test_file.read_text(encoding="utf-8")
            assert "pytest.mark.integration" not in text
            assert "pytest.mark.slow" not in text
