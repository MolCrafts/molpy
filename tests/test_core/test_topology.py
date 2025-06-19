"""
Test suite for molpy.core.topology module.
"""

import pytest
from molpy.core.topology import Topology


class TestTopology:
    def test_topology_creation(self):
        """Test basic topology creation."""
        topo = Topology()
        assert isinstance(topo, Topology)
        
    def test_topology_has_required_methods(self):
        """Test that topology has required methods."""
        topo = Topology()
        # igraph Graph functionality should be available
        assert hasattr(topo, 'add_vertex')
        assert hasattr(topo, 'add_edge')
        assert hasattr(topo, 'vs')  # vertices
        assert hasattr(topo, 'es')  # edges
        
        # Custom properties should be available
        assert hasattr(topo, 'n_atoms')
        assert hasattr(topo, 'n_bonds')
        assert hasattr(topo, 'n_angles')
        assert hasattr(topo, 'n_dihedrals')
