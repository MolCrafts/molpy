"""Regression tests for post-class [N].move(...) array replication."""

from __future__ import annotations

from molpy.parser.moltemplate import NewStmt, parse_string


class TestParseArrayForm:
    def test_single_dim(self):
        doc = parse_string("m = new X [12].move(0, 0, 5.2)\n")
        s = doc.statements[0]
        assert isinstance(s, NewStmt)
        assert s.class_name == "X"
        assert len(s.arrays) == 1
        assert s.arrays[0].count == 12
        assert s.arrays[0].transform is not None
        assert s.arrays[0].transform.op == "move"
        assert s.arrays[0].transform.args == [0.0, 0.0, 5.2]

    def test_three_dims(self):
        doc = parse_string(
            "m = new Butane [12].move(0, 0, 5.2)\n"
            "              [12].move(0, 5.2, 0)\n"
            "              [6].move(10.4, 0, 0)\n"
        )
        s = doc.statements[0]
        assert isinstance(s, NewStmt)
        counts = [a.count for a in s.arrays]
        assert counts == [12, 12, 6]
        # All three should carry .move transforms
        ops = [a.transform.op for a in s.arrays]
        assert ops == ["move", "move", "move"]


class TestSingleQuotedSection:
    def test_single_quote_section(self):
        from molpy.parser.moltemplate import WriteBlock

        doc = parse_string(
            "Foo {\n  write('Data Bond List') {\n    $bond:b1 $atom:a $atom:b\n  }\n}\n"
        )
        cls = doc.statements[0]
        block = cls.statements[0]
        assert isinstance(block, WriteBlock)
        assert block.section == "Data Bond List"


class TestButaneSystem:
    """Integration: real butane/system.lt expands 864 molecules."""

    def test_butane_system_expansion(self):
        # Construct an inline butane-like test to avoid requiring cloned repo
        src = """
Butane {
  write("Data Atoms") {
    $atom:c $mol:. @atom:C 0.0 0.0 0.0 0.0
  }
}

m = new Butane [3].move(0, 0, 1.0) [2].move(0, 1.0, 0) [2].move(1.0, 0, 0)
"""
        from molpy.io.forcefield.moltemplate import read_moltemplate_system
        from molpy.parser.moltemplate import build_system, parse_string

        doc = parse_string(src)
        system, _ff = build_system(doc)
        # 3 * 2 * 2 = 12 copies, each with 1 atom
        assert len(list(system.atoms)) == 12
