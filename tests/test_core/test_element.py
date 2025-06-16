import pytest
import molpy as mp
from molpy.core.element import Element


class TestElement:
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Initialize elements before each test"""
        Element.initialize()
    
    def test_initialize(self):
        """Test element initialization"""
        Element.initialize()
        assert len(Element._elements) > 0
        assert "hydrogen" in Element._elements
        assert "carbon" in Element._elements
        assert "oxygen" in Element._elements
    
    def test_get_element_by_name(self):
        """Test getting element by name"""
        h = Element("hydrogen")
        assert h.name == "hydrogen"
        assert h.symbol == "H"
        assert h.number == 1
        assert abs(h.mass - 1.007947) < 0.001
    
    def test_get_element_by_symbol(self):
        """Test getting element by symbol"""
        c = Element("C")
        assert c.name == "carbon"
        assert c.symbol == "C"
        assert c.number == 6
        assert abs(c.mass - 12.01078) < 0.001
    
    def test_get_element_by_number(self):
        """Test getting element by atomic number"""
        o = Element(8)
        assert o.name == "oxygen"
        assert o.symbol == "O"
        assert o.number == 8
        assert abs(o.mass - 15.99943) < 0.001
    
    def test_element_not_found(self):
        """Test error when element not found"""
        with pytest.raises(KeyError, match="Element not found"):
            Element("nonexistent")
        
        with pytest.raises(KeyError, match="Element not found"):
            Element(999)
    
    def test_element_repr(self):
        """Test element string representation"""
        h = Element("H")
        assert repr(h) == "<Element H>"
        
        c = Element("carbon")
        assert repr(c) == "<Element C>"
    
    def test_get_symbols_method(self):
        """Test get_symbols class method"""
        inputs = ["hydrogen", "C", 8, "N"]
        expected = ["H", "C", "O", "N"]
        result = Element.get_symbols(inputs)
        assert result == expected
    
    def test_get_symbols_with_mixed_types(self):
        """Test get_symbols with mixed input types"""
        inputs = [1, "carbon", "O", 7]
        expected = ["H", "C", "O", "N"]
        result = Element.get_symbols(inputs)
        assert result == expected
    
    def test_deuterium_special_case(self):
        """Test deuterium handling"""
        d = Element("deuterium")
        assert d.name == "deuterium"
        assert d.symbol == "D"
        assert d.number == 1  # Same as hydrogen
        assert abs(d.mass - 2.01355321270) < 0.001
    
    def test_element_dataclass_properties(self):
        """Test element dataclass properties"""
        elem = Element.Element(1, "test", "T", 1.0)
        assert elem.number == 1
        assert elem.name == "test"
        assert elem.symbol == "T"
        assert elem.mass == 1.0
        assert repr(elem) == "<Element T>"
    
    def test_multiple_element_access(self):
        """Test accessing multiple elements"""
        elements = ["H", "C", "N", "O"]
        for elem_symbol in elements:
            elem = Element(elem_symbol)
            assert elem.symbol == elem_symbol
            assert elem.number > 0
            assert elem.mass > 0
