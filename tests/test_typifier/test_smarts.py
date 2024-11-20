import lark
import pytest
import molpy as mp

# from molpy.typifier.exceptions import molpy.typifierError
# from molpy.typifier.forcefield import Forcefield
from molpy.typifier.smarts import SmartsParser
from molpy.typifier.smarts_graph import SMARTSGraph

# from molpy.typifier.topology_graph import TopologyGraph


class TestSMARTS:

    @pytest.fixture(scope="session")
    def smarts_parser(self):
        return SmartsParser()

    @pytest.fixture(scope="session")
    def rule_match(self, smarts_parser):
        def _rule_match(top, smart, result):
            rule = SMARTSGraph(name="test", parser=smarts_parser, smarts_string=smart)
            assert bool(list(rule.find_matches(top))) is result

        return _rule_match

    @pytest.fixture(scope="session")
    def rule_match_count(self, smarts_parser):
        def _rule_match_count(top, smart, count):
            rule = SMARTSGraph(
                name="test",
                parser=smarts_parser,
                smarts_string=smart,
            )
            assert len(list(rule.find_matches(top))) is count

        return _rule_match_count

    def test_ast(self, smarts_parser):
        ast = smarts_parser.parse("O([H&X1])(H)")
        assert ast.data == "start"
        assert ast.children[0].data == "atom"
        assert ast.children[0].children[0].data == "atom_symbol"
        assert str(ast.children[0].children[0].children[0]) == "O"

    @pytest.mark.parametrize(
        "pattern", ["[#6][#1](C)H", "[O;X2]([C;X4](F)(*)(*))[C;X4]"]
    )
    def test_parse(self, pattern, smarts_parser):
        assert smarts_parser.parse(pattern)

    def test_uniqueness(self, rule_match, test_data_path):
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/uniqueness_test.mol2")
            .frame.to_struct()
            .get_topology()
        )
        rule_match(mol2, "[#6]1[#6][#6][#6][#6][#6]1", False)
        rule_match(mol2, "[#6]1[#6][#6][#6][#6]1", False)
        rule_match(mol2, "[#6]1[#6][#6][#6]1", True)

    def test_ringness(self, rule_match, test_data_path):
        ring_mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/ring.mol2")
            .frame.to_struct()
            .get_topology()
        )

        rule_match(ring_mol2, "[#6]1[#6][#6][#6][#6][#6]1", True)

        not_ring_mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/not_ring.mol2")
            .frame.to_struct()
            .get_topology()
        )

        rule_match(not_ring_mol2, "[#6]1[#6][#6][#6][#6][#6]1", False)

    def test_fused_ring(self, smarts_parser, test_data_path):
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/fused.mol2")
            .frame.to_struct()
            .get_topology()
        )
        rule = SMARTSGraph(
            name="test",
            parser=smarts_parser,
            smarts_string="[#6]12[#6][#6][#6][#6][#6]1[#6][#6][#6][#6]2",
        )

        match_indices = list(rule.find_matches(mol2))
        assert 3 in match_indices
        assert 4 in match_indices
        assert len(match_indices) == 2

    def test_ring_count(self, smarts_parser, test_data_path):
        # Two rings
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/fused.mol2")
            .frame.to_struct()
            .get_topology()
        )
        rule = SMARTSGraph(name="test", parser=smarts_parser, smarts_string="[#6;R2]")

        match_indices = list(rule.find_matches(mol2))
        for atom_idx in (3, 4):
            assert atom_idx in match_indices
        assert len(match_indices) == 2

        rule = SMARTSGraph(name="test", parser=smarts_parser, smarts_string="[#6;R1]")
        match_indices = list(rule.find_matches(mol2))
        for atom_idx in (0, 1, 2, 5, 6, 7, 8, 9):
            assert atom_idx in match_indices
        assert len(match_indices) == 8

        # One ring
        ring = (
            mp.io.read_mol2(test_data_path / "data/mol2/ring.mol2")
            .frame.to_struct()
            .get_topology()
        )

        rule = SMARTSGraph(name="test", parser=smarts_parser, smarts_string="[#6;R1]")
        match_indices = list(rule.find_matches(ring))
        for atom_idx in range(6):
            assert atom_idx in match_indices
        assert len(match_indices) == 6

    def test_precedence_ast(self, smarts_parser):
        ast1 = smarts_parser.parse("[C,H;O]")
        ast2 = smarts_parser.parse("[O;H,C]")
        assert ast1.children[0].children[0].data == "weak_and_expression"
        assert ast2.children[0].children[0].data == "weak_and_expression"

        assert ast1.children[0].children[0].children[0].data == "or_expression"
        assert ast2.children[0].children[0].children[1].data == "or_expression"

        ast1 = smarts_parser.parse("[C,H&O]")
        ast2 = smarts_parser.parse("[O&H,C]")
        assert ast1.children[0].children[0].data == "or_expression"
        assert ast2.children[0].children[0].data == "or_expression"

        assert ast1.children[0].children[0].children[1].data == "and_expression"
        assert ast2.children[0].children[0].children[0].data == "and_expression"

    def test_precedence(self, rule_match_count, test_data_path):
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/ethane.mol2")
            .frame.to_struct()
            .get_topology()
        )

        checks = {
            "[C,O;C]": 2,
            "[C&O;C]": 0,
            "[!C;O,C]": 0,
            "[!C&O,C]": 2,
        }

        for smart, result in checks.items():
            rule_match_count(mol2, smart, result)

    def test_pf6(self, rule_match_count, test_data_path):
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/pf6.mol2")
            .frame.to_struct()
            .get_topology()
        )

        checks = {
            r"P": 1,
            r"FP([!%F1])([!%F1])([!%F1])([!%F1])F": 2,
            r"FP([!%F1;!%F2])([!%F1;!%F2])([%F1])([%F1])F": 2,
            r"FP([%F1])([%F1])([%F2])([%F2])F": 2,
        }

        for smart, result in checks.items():
            rule_match_count(mol2, smart, result)

    def test_not_ast(self, smarts_parser):
        checks = {
            "[!C;!H]": "weak_and_expression",
            "[!C&H]": "and_expression",
            "[!C;H]": "weak_and_expression",
            "[!C]": "not_expression",
        }

        for smart, grandchild in checks.items():
            ast = smarts_parser.parse(smart)
            assert ast.children[0].children[0].data == grandchild

        illegal_nots = ["[!CH]", "[!C!H]"]
        for smart in illegal_nots:
            with pytest.raises(lark.UnexpectedInput):
                smarts_parser.parse(smart)

    def test_not(self, rule_match_count, test_data_path):
        mol2 = mp.io.read_mol2(
            test_data_path / "data/mol2/ethane.mol2"
        ).frame.to_struct().get_topology()

        checks = {
            "[!O]": 8,
            "[!#5]": 8,
            "[!C]": 6,
            "[!#6]": 6,
            "[!C&!H]": 0,
            "[!C;!H]": 0,
        }
        for smart, result in checks.items():
            rule_match_count(mol2, smart, result)

    def test_hexa_coordinated(self, test_data_path):
        ff = mp.io.read_xml_forcefield(
            test_data_path / "forcefield/xml/pf6.xml"
        ).forcefield
        mol2 = mp.io.read_mol2(test_data_path / "data/mol2/pf6.mol2").frame.to_struct()

        typifier = mp.SmartsTypifier(ff)
        pf6 = typifier.typify(mol2)

        types = [a["type"] for a in pf6["atoms"]]
        assert types.count("P") == 1
        assert types.count("F1") == 2
        assert types.count("F2") == 2
        assert types.count("F3") == 2

        # assert len(pf6["bonds"]) == 6
        # assert all(bond["type"] for bond in pf6["bonds"])

        # assert len(pf6["angles"]) == 15
        # assert all(angle["type"] for angle in pf6["angles"])

    def test_optional_names_bad_syntax(self):
        bad_optional_names = ["_C", "XXX", "C"]
        with pytest.raises(Exception):
            SmartsParser(optional_names=bad_optional_names)

    def test_optional_names_good_syntax(self):
        good_optional_names = ["_C", "_CH2", "_CH"]
        SmartsParser(optional_names=good_optional_names)

    def test_optional_name_parser(self):
        optional_names = ["_C", "_CH2", "_CH"]
        S = SmartsParser(optional_names=optional_names)
        ast = S.parse("_CH2_C_CH")
        symbols = [a.children[0] for a in ast.find_data("atom_symbol")]
        for name in optional_names:
            assert name in symbols
