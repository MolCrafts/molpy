"""Tests for the native MolTemplate parser (IR level)."""

from __future__ import annotations

from pathlib import Path

from molpy.parser.moltemplate import (
    ClassDef,
    Document,
    NewStmt,
    WriteBlock,
    WriteOnceBlock,
    parse_file,
    parse_string,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestTokenizer:
    def test_parse_empty(self):
        doc = parse_string("")
        assert isinstance(doc, Document)
        assert doc.statements == []

    def test_parse_comment_only(self):
        doc = parse_string("# just a comment\n# another\n")
        assert doc.statements == []


class TestClassDef:
    def test_parse_simple_class(self):
        doc = parse_string("Foo {\n}\n")
        assert len(doc.statements) == 1
        cls = doc.statements[0]
        assert isinstance(cls, ClassDef)
        assert cls.name == "Foo"
        assert cls.bases == []

    def test_parse_inherits(self):
        doc = parse_string("Bar inherits Foo, Baz {\n}\n")
        cls = doc.statements[0]
        assert isinstance(cls, ClassDef)
        assert cls.name == "Bar"
        assert cls.bases == ["Foo", "Baz"]


class TestWriteBlock:
    def test_write_once_captures_lines(self):
        src = """
Foo {
  write_once("Data Masses") {
    @atom:O  15.9994
    @atom:H   1.008
  }
}
"""
        doc = parse_string(src)
        cls = doc.statements[0]
        assert isinstance(cls, ClassDef)
        block = cls.statements[0]
        assert isinstance(block, WriteOnceBlock)
        assert block.section == "Data Masses"
        assert any("@atom:O" in ln for ln in block.body_lines)
        assert any("@atom:H" in ln for ln in block.body_lines)

    def test_write_captures_atoms(self):
        src = """
Foo {
  write("Data Atoms") {
    $atom:O  $mol:. @atom:O  -0.834  0.0000  0.0000  0.0000
  }
}
"""
        doc = parse_string(src)
        cls = doc.statements[0]
        assert isinstance(cls, ClassDef)
        block = cls.statements[0]
        assert isinstance(block, WriteBlock)
        assert block.section == "Data Atoms"
        assert any("$atom:O" in ln for ln in block.body_lines)


class TestNew:
    def test_parse_new_no_transform(self):
        doc = parse_string("w1 = new TIP3P\n")
        new = doc.statements[0]
        assert isinstance(new, NewStmt)
        assert new.instance_name == "w1"
        assert new.class_name == "TIP3P"
        assert new.transforms == []

    def test_parse_new_with_move(self):
        doc = parse_string("w2 = new TIP3P.move(3.0, 0.0, 0.0)\n")
        new = doc.statements[0]
        assert isinstance(new, NewStmt)
        assert len(new.transforms) == 1
        assert new.transforms[0].op == "move"
        assert new.transforms[0].args == [3.0, 0.0, 0.0]

    def test_parse_new_with_chain(self):
        doc = parse_string("w3 = new TIP3P.move(1,2,3).rot(45, 0, 0, 1).scale(1.5)\n")
        new = doc.statements[0]
        assert isinstance(new, NewStmt)
        assert [t.op for t in new.transforms] == ["move", "rot", "scale"]
        assert new.transforms[1].args == [45.0, 0.0, 0.0, 1.0]


class TestTip3pFixture:
    def test_parse_tip3p(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        # Expect: one class def + two `new` statements at top level
        kinds = [type(s).__name__ for s in doc.statements]
        assert "ClassDef" in kinds
        assert kinds.count("NewStmt") == 2
        cls = next(s for s in doc.statements if isinstance(s, ClassDef))
        sections = {
            s.section
            for s in cls.statements
            if isinstance(s, (WriteBlock, WriteOnceBlock))
        }
        assert {
            "Data Masses",
            "In Charges",
            "In Settings",
            "Data Atoms",
            "Data Bonds",
            "Data Angles",
        } <= sections
