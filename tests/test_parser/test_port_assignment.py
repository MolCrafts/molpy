"""Test port assignment in BigSMILES to Atomistic conversion.

This tests the fix for the bug where ports were assigned to wrong atoms.
For BigSMILES like {[][$]OCCO[$][]}, ports should be on the O atoms (first and last),
not sequentially on O, C.

Note: BigSMILES v1.1 requires both terminal descriptors to be explicit.
"""

import pytest

from molpy.parser.smiles import parse_bigsmiles, bigsmilesir_to_polymerspec


class TestPortAssignment:
    """Test that ports are correctly assigned to atoms based on BigSMILES notation."""

    def test_ports_on_terminal_oxygen_atoms(self):
        """Test that {[][$]OCCO[$][]} assigns ports to both O atoms, not O and C.

        This was a bug: descriptors without position_hint were assigned sequentially
        to atoms[0], atoms[1] instead of atoms[0], atoms[-1].
        """
        bigsmiles = "{[][$]OCCO[$][]}"
        ir = parse_bigsmiles(bigsmiles)
        spec = bigsmilesir_to_polymerspec(ir)
        monomers = spec.all_monomers()

        assert len(monomers) == 1
        monomer = monomers[0]

        # Find atoms with ports
        port_atoms = [a for a in monomer.atoms if a.get("port")]

        # Should have exactly 2 ports
        assert len(port_atoms) == 2, f"Expected 2 ports, got {len(port_atoms)}"

        # Both ports should be on O atoms, not C
        for atom in port_atoms:
            assert (
                atom.get("symbol") == "O"
            ), f"Port should be on O atom, got {atom.get('symbol')}"

    def test_ports_on_first_and_last_atoms(self):
        """Test first/last convention for port assignment."""
        bigsmiles = "{[][$]CC[$][]}"
        ir = parse_bigsmiles(bigsmiles)
        spec = bigsmilesir_to_polymerspec(ir)
        monomers = spec.all_monomers()

        assert len(monomers) == 1
        monomer = monomers[0]
        atoms = list(monomer.atoms)

        # First and last atoms should have ports
        assert atoms[0].get("port") == "$"
        assert atoms[-1].get("port") == "$"

    def test_three_port_monomer(self):
        """Test 3-port monomer like {[][$]OCC(CO[$])(CO[$])[]}."""
        bigsmiles = "{[][$]OCC(CO[$])(CO[$])[]}"
        ir = parse_bigsmiles(bigsmiles)
        spec = bigsmilesir_to_polymerspec(ir)
        monomers = spec.all_monomers()

        assert len(monomers) == 1
        monomer = monomers[0]

        # Find atoms with ports
        port_atoms = [a for a in monomer.atoms if a.get("port")]

        # Should have exactly 3 ports
        assert len(port_atoms) == 3, f"Expected 3 ports, got {len(port_atoms)}"

    def test_single_port_monomer(self):
        """Test single port monomer like {[][$]CCO[]}.

        Note: Per BigSMILES v1.1, both terminals are required.
        Empty terminal [] means no external connection.
        """
        bigsmiles = "{[][$]CCO[]}"
        ir = parse_bigsmiles(bigsmiles)
        spec = bigsmilesir_to_polymerspec(ir)
        monomers = spec.all_monomers()

        assert len(monomers) == 1
        monomer = monomers[0]
        atoms = list(monomer.atoms)

        # First atom should have port
        port_atoms = [a for a in atoms if a.get("port")]
        assert len(port_atoms) == 1
        assert atoms[0].get("port") == "$"

    def test_directed_descriptors(self):
        """Test directed descriptors like {[][<]CC[>][]}."""
        bigsmiles = "{[][<]CC[>][]}"
        ir = parse_bigsmiles(bigsmiles)
        spec = bigsmilesir_to_polymerspec(ir)
        monomers = spec.all_monomers()

        assert len(monomers) == 1
        monomer = monomers[0]
        atoms = list(monomer.atoms)

        # Check ports are assigned
        port_atoms = [a for a in atoms if a.get("port")]
        assert len(port_atoms) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
