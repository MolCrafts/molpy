from dataclasses import dataclass
from typing import Union, List

__all__ = ["Element"]

daltons = 1


@dataclass
class ElementData:
    """Data class representing the properties of a chemical element."""
    number: int
    name: str
    symbol: str
    mass: float

    def __repr__(self) -> str:
        return f"<Element {self.symbol}>"


class Element:
    """
    The `Element` class represents chemical elements and provides functionality to retrieve
    element information by name, symbol, or atomic number. It also includes a method to
    initialize a predefined set of elements and their properties.

    This class acts as a factory that returns ElementData instances based on the input
    identifier (name, symbol, or atomic number).

    Attributes:
        _elements (dict[str, ElementData]): A dictionary storing predefined elements, where
            keys are element names and values are `ElementData` instances.
        _symbol_to_element (dict[str, ElementData]): A dictionary mapping symbols to elements.
        _number_to_element (dict[int, ElementData]): A dictionary mapping atomic numbers to elements.

    Methods:
        __new__(self, name_or_symbol_or_number: str | int) -> ElementData:
            Retrieves an `ElementData` instance based on its name, symbol, or atomic number.
            Raises a `KeyError` if the element is not found.

        initialize(cls):
            Class method to initialize the element dictionaries with predefined elements
            and their properties, such as atomic number, name, symbol, and atomic mass.

        get_symbols(cls, identifiers: list[str | int]) -> list[str]:
            Class method that takes a list of element names, symbols, or atomic numbers
            and returns a list of their corresponding symbols.

        get_atomic_number(cls, symbol: str) -> int:
            Class method that returns the atomic number for a given element symbol.
    """

    _elements: dict[str, ElementData] = {}
    _symbol_to_element: dict[str, ElementData] = {}
    _number_to_element: dict[int, ElementData] = {}

    def __new__(cls, name_or_symbol_or_number: Union[str, int]) -> "ElementData":
        """
        Create an ElementData instance based on name, symbol, or atomic number.
        
        Args:
            name_or_symbol_or_number: Element identifier (name, symbol, or atomic number)
            
        Returns:
            ElementData instance
            
        Raises:
            KeyError: If element is not found
        """
        if isinstance(name_or_symbol_or_number, int):
            if name_or_symbol_or_number in cls._number_to_element:
                return cls._number_to_element[name_or_symbol_or_number]
        elif isinstance(name_or_symbol_or_number, str):
            # Try name first
            if name_or_symbol_or_number in cls._elements:
                return cls._elements[name_or_symbol_or_number]
            # Then try symbol
            if name_or_symbol_or_number in cls._symbol_to_element:
                return cls._symbol_to_element[name_or_symbol_or_number]
                
        raise KeyError(f"Element not found: {name_or_symbol_or_number}")

    @classmethod
    def get_symbols(cls, identifiers: List[Union[str, int]]) -> List[str]:
        """Convert list of element identifiers to symbols."""
        return [cls(e).symbol for e in identifiers]
    
    @classmethod
    def get_atomic_number(cls, symbol: str) -> int:
        """Get atomic number by element symbol."""
        if symbol in cls._symbol_to_element:
            return cls._symbol_to_element[symbol].number
        raise KeyError(f"Element with symbol '{symbol}' not found")

    @classmethod
    def initialize(cls):
        """Initialize the element databases with predefined elements."""
        elements_data = [
            ElementData(1, "hydrogen", "H", 1.007947 * daltons),
            # ElementData(1, "deuterium", "D", 2.01355321270 * daltons),
            ElementData(2, "helium", "He", 4.003 * daltons),
            ElementData(3, "lithium", "Li", 6.9412 * daltons),
            ElementData(4, "beryllium", "Be", 9.0121823 * daltons),
            ElementData(5, "boron", "B", 10.8117 * daltons),
            ElementData(6, "carbon", "C", 12.01078 * daltons),
            ElementData(7, "nitrogen", "N", 14.00672 * daltons),
            ElementData(8, "oxygen", "O", 15.99943 * daltons),
            ElementData(9, "fluorine", "F", 18.99840325 * daltons),
            ElementData(10, "neon", "Ne", 20.17976 * daltons),
            ElementData(11, "sodium", "Na", 22.989769282 * daltons),
            ElementData(12, "magnesium", "Mg", 24.30506 * daltons),
            ElementData(13, "aluminum", "Al", 26.98153868 * daltons),
            ElementData(14, "silicon", "Si", 28.08553 * daltons),
            ElementData(15, "phosphorus", "P", 30.9737622 * daltons),
            ElementData(16, "sulfur", "S", 32.0655 * daltons),
            ElementData(17, "chlorine", "Cl", 35.4532 * daltons),
            ElementData(18, "argon", "Ar", 39.9481 * daltons),
            ElementData(19, "potassium", "K", 39.09831 * daltons),
            ElementData(20, "calcium", "Ca", 40.0784 * daltons),
            ElementData(21, "scandium", "Sc", 44.9559126 * daltons),
            ElementData(22, "titanium", "Ti", 47.8671 * daltons),
            ElementData(23, "vanadium", "V", 50.94151 * daltons),
            ElementData(24, "chromium", "Cr", 51.99616 * daltons),
            ElementData(25, "manganese", "Mn", 54.9380455 * daltons),
            ElementData(26, "iron", "Fe", 55.8452 * daltons),
            ElementData(27, "cobalt", "Co", 58.9331955 * daltons),
            ElementData(28, "nickel", "Ni", 58.69342 * daltons),
            ElementData(29, "copper", "Cu", 63.5463 * daltons),
            ElementData(30, "zinc", "Zn", 65.4094 * daltons),
            ElementData(31, "gallium", "Ga", 69.7231 * daltons),
            ElementData(32, "germanium", "Ge", 72.641 * daltons),
            ElementData(33, "arsenic", "As", 74.921602 * daltons),
            ElementData(34, "selenium", "Se", 78.963 * daltons),
            ElementData(35, "bromine", "Br", 79.9041 * daltons),
            ElementData(36, "krypton", "Kr", 83.7982 * daltons),
            ElementData(37, "rubidium", "Rb", 85.46783 * daltons),
            ElementData(38, "strontium", "Sr", 87.621 * daltons),
            ElementData(39, "yttrium", "Y", 88.905852 * daltons),
            ElementData(40, "zirconium", "Zr", 91.2242 * daltons),
            ElementData(41, "niobium", "Nb", 92.906382 * daltons),
            ElementData(42, "molybdenum", "Mo", 95.942 * daltons),
            ElementData(43, "technetium", "Tc", 98 * daltons),
            ElementData(44, "ruthenium", "Ru", 101.072 * daltons),
            ElementData(45, "rhodium", "Rh", 102.905502 * daltons),
            ElementData(46, "palladium", "Pd", 106.421 * daltons),
            ElementData(47, "silver", "Ag", 107.86822 * daltons),
            ElementData(48, "cadmium", "Cd", 112.4118 * daltons),
            ElementData(49, "indium", "In", 114.8183 * daltons),
            ElementData(50, "tin", "Sn", 118.7107 * daltons),
            ElementData(51, "antimony", "Sb", 121.7601 * daltons),
            ElementData(52, "tellurium", "Te", 127.603 * daltons),
            ElementData(53, "iodine", "I", 126.904473 * daltons),
            ElementData(54, "xenon", "Xe", 131.2936 * daltons),
            ElementData(55, "cesium", "Cs", 132.90545192 * daltons),
            ElementData(56, "barium", "Ba", 137.3277 * daltons),
            ElementData(57, "lanthanum", "La", 138.905477 * daltons),
            ElementData(58, "cerium", "Ce", 140.1161 * daltons),
            ElementData(59, "praseodymium", "Pr", 140.907652 * daltons),
            ElementData(60, "neodymium", "Nd", 144.2423 * daltons),
            ElementData(61, "promethium", "Pm", 145 * daltons),
            ElementData(62, "samarium", "Sm", 150.362 * daltons),
            ElementData(63, "europium", "Eu", 151.9641 * daltons),
            ElementData(64, "gadolinium", "Gd", 157.253 * daltons),
            ElementData(65, "terbium", "Tb", 158.925352 * daltons),
            ElementData(66, "dysprosium", "Dy", 162.5001 * daltons),
            ElementData(67, "holmium", "Ho", 164.930322 * daltons),
            ElementData(68, "erbium", "Er", 167.2593 * daltons),
            ElementData(69, "thulium", "Tm", 168.934212 * daltons),
            ElementData(70, "ytterbium", "Yb", 173.043 * daltons),
            ElementData(71, "lutetium", "Lu", 174.9671 * daltons),
            ElementData(72, "hafnium", "Hf", 178.492 * daltons),
            ElementData(73, "tantalum", "Ta", 180.947882 * daltons),
            ElementData(74, "tungsten", "W", 183.841 * daltons),
            ElementData(75, "rhenium", "Re", 186.2071 * daltons),
            ElementData(76, "osmium", "Os", 190.233 * daltons),
            ElementData(77, "iridium", "Ir", 192.2173 * daltons),
            ElementData(78, "platinum", "Pt", 195.0849 * daltons),
            ElementData(79, "gold", "Au", 196.9665694 * daltons),
            ElementData(80, "mercury", "Hg", 200.592 * daltons),
            ElementData(81, "thallium", "Tl", 204.38332 * daltons),
            ElementData(82, "lead", "Pb", 207.21 * daltons),
            ElementData(83, "bismuth", "Bi", 208.980401 * daltons),
            ElementData(84, "polonium", "Po", 209 * daltons),
            ElementData(85, "astatine", "At", 210 * daltons),
            ElementData(86, "radon", "Rn", 222.018 * daltons),
            ElementData(87, "francium", "Fr", 223 * daltons),
            ElementData(88, "radium", "Ra", 226 * daltons),
            ElementData(89, "actinium", "Ac", 227 * daltons),
            ElementData(90, "thorium", "Th", 232.038062 * daltons),
            ElementData(91, "protactinium", "Pa", 231.035882 * daltons),
            ElementData(92, "uranium", "U", 238.028913 * daltons),
            ElementData(93, "neptunium", "Np", 237 * daltons),
            ElementData(94, "plutonium", "Pu", 244 * daltons),
            ElementData(95, "americium", "Am", 243 * daltons),
            ElementData(96, "curium", "Cm", 247 * daltons),
            ElementData(97, "berkelium", "Bk", 247 * daltons),
            ElementData(98, "californium", "Cf", 251 * daltons),
            ElementData(99, "einsteinium", "Es", 252 * daltons),
            ElementData(100, "fermium", "Fm", 257 * daltons),
            ElementData(101, "mendelevium", "Md", 258 * daltons),
            ElementData(102, "nobelium", "No", 259 * daltons),
            ElementData(103, "lawrencium", "Lr", 262 * daltons),
            ElementData(104, "rutherfordium", "Rf", 261 * daltons),
            ElementData(105, "dubnium", "Db", 262 * daltons),
            ElementData(106, "seaborgium", "Sg", 266 * daltons),
            ElementData(107, "bohrium", "Bh", 264 * daltons),
            ElementData(108, "hassium", "Hs", 269 * daltons),
            ElementData(109, "meitnerium", "Mt", 268 * daltons),
            ElementData(110, "darmstadtium", "Ds", 281 * daltons),
            ElementData(111, "roentgenium", "Rg", 272 * daltons),
            ElementData(112, "ununbium", "Uub", 285 * daltons),
            ElementData(113, "ununtrium", "Uut", 284 * daltons),
            ElementData(114, "ununquadium", "Uuq", 289 * daltons),
            ElementData(115, "ununpentium", "Uup", 288 * daltons),
            ElementData(116, "ununhexium", "Uuh", 292 * daltons),
        ]
        
        # Clear existing dictionaries
        cls._elements.clear()
        cls._symbol_to_element.clear()
        cls._number_to_element.clear()
        
        # Populate dictionaries
        for element in elements_data:
            cls._elements[element.name] = element
            cls._symbol_to_element[element.symbol] = element
            cls._number_to_element[element.number] = element

Element.initialize()
