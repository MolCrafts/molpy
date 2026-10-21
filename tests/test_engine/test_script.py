"""Script: an immutable (name, text, path) value."""

import dataclasses

import pytest

from molpy.engine import Script


def test_is_immutable():
    script = Script.from_text("input", "units real\n")
    with pytest.raises(dataclasses.FrozenInstanceError):
        script.text = "x"  # type: ignore[misc]


def test_from_text_dedents_a_literal_and_ends_in_one_newline():
    script = Script.from_text(
        "input",
        """
        units real
        run 0


        """,
    )
    assert script.text == "\nunits real\nrun 0\n"
    assert Script.from_text("empty", "").text == ""


def test_from_path_keeps_the_file_text(tmp_path):
    path = tmp_path / "in.lmp"
    path.write_text("  units real\nrun 0", encoding="utf-8")
    script = Script.from_path(path)
    assert (script.name, script.text, script.path) == (
        "in",
        "  units real\nrun 0",
        path,
    )


def test_save_writes_without_changing_the_script(tmp_path):
    script = Script.from_text("input", "units real\n")
    target = script.save(tmp_path / "deep" / "in.lmp")
    assert target.read_text(encoding="utf-8") == "units real\n"
    assert script.path is None


def test_save_needs_a_path():
    with pytest.raises(ValueError, match="no path"):
        Script.from_text("input", "x").save()


def test_preview_numbers_lines_and_counts_the_rest():
    script = Script.from_text("input", "\n".join(f"l{i}" for i in range(12)))
    assert script.preview(max_lines=2) == " 1 | l0\n 2 | l1\n... (10 more lines)"
    assert Script.from_text("e", "").preview() == "(empty script)"


def test_only_the_value_api_remains():
    for gone in ("from_url", "append", "insert", "format", "move", "rename", "reload"):
        assert not hasattr(Script, gone)
