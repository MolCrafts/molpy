"""Architecture checks for the assembly unit-test package."""

import ast
from pathlib import Path


class TestAssemblyTestLayout:
    SOURCE = (
        Path(__file__).resolve().parents[3] / "src" / "molpy" / "builder" / "assembly"
    )
    TESTS = Path(__file__).resolve().parent

    def test_test_modules_mirror_source_modules(self):
        source_modules = {
            path.stem.removeprefix("_")
            for path in self.SOURCE.glob("_*.py")
            if path.name != "__init__.py"
        }
        test_modules = {
            path.stem.removeprefix("test_")
            for path in self.TESTS.glob("test_*.py")
            if path.name != "test_init.py"
        }
        assert test_modules == source_modules

    def test_each_production_class_has_its_own_test_class(self):
        for source in self.SOURCE.glob("_*.py"):
            if source.name == "__init__.py":
                continue
            module = source.stem.removeprefix("_")
            production_classes = {
                node.name
                for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef)
            }
            test_file = self.TESTS / f"test_{module}.py"
            test_classes = {
                node.name
                for node in ast.parse(test_file.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            }
            expected = {f"Test{name.removeprefix('_')}" for name in production_classes}
            assert expected <= test_classes, (
                f"{source.name} is missing {sorted(expected - test_classes)} in "
                f"{test_file.name}"
            )

    def test_unit_mirror_contains_no_integration_markers(self):
        for test_file in self.TESTS.glob("test_*.py"):
            if test_file == Path(__file__):
                continue
            text = test_file.read_text(encoding="utf-8")
            assert "pytest.mark.integration" not in text
            assert "pytest.mark.slow" not in text
