from dataclasses import dataclass

__all__ = ["Element"]

daltons = 1


class Element:
    """
    The `Element` class represents chemical elements and provides functionality to retrieve
    element information by name, symbol, or atomic number. It also includes a method to
    initialize a predefined set of elements and their properties.

    Attributes:
        _elements (dict[str, Element]): A dictionary storing predefined elements, where
            keys are element names and values are `Element` instances.

    Methods:
        __new__(self, name_or_symbol_or_number: str | int) -> "Element":
            Retrieves an `Element` instance based on its name, symbol, or atomic number.
            Raises a `KeyError` if the element is not found.

        initialize(cls):
            Class method to initialize the `_elements` dictionary with predefined elements
            and their properties, such as atomic number, name, symbol, and atomic mass.

        get_symbols(cls, maybe_atomic_number: list[str | int]) -> list[str]:
            Class method that takes a list of element names, symbols, or atomic numbers
            and returns a list of their corresponding symbols.

    Inner Classes:
        Element:
            A dataclass representing the properties of a chemical element.

            Attributes:
                number (int): The atomic number of the element.
                name (str): The name of the element.
                symbol (str): The chemical symbol of the element.
                mass (float): The atomic mass of the element.

            Methods:
                __repr__(self) -> str:
                    Returns a string representation of the element in the format
                    `<Element {symbol}>`.
    """

    @dataclass
    class Element:
        number: int
        name: str
        symbol: str
        mass: float

        def __repr__(self) -> str:
            return f"<Element {self.symbol}>"

    _elements: dict[str, Element] = {}

    def __new__(self, name_or_symbol_or_number: str | int) -> "Element":
        if isinstance(name_or_symbol_or_number, int):  # number
            result = next(
                (
                    e
                    for e in self._elements.values()
                    if e.number == name_or_symbol_or_number
                ),
                None,
            )
        elif isinstance(name_or_symbol_or_number, str):  # name or symbol
            if name_or_symbol_or_number in self._elements:
                result = self._elements[name_or_symbol_or_number]
            else:
                result = next(
                    (
                        e
                        for e in self._elements.values()
                        if e.symbol == name_or_symbol_or_number
                    ),
                    None,
                )
        if result is None:
            raise KeyError(f"Element not found: {name_or_symbol_or_number}")

        return result

    @classmethod
    def initialize(cls):

        cls._elements = dict(
            hydrogen=cls.Element(1, "hydrogen", "H", 1.007947 * daltons),
            deuterium=cls.Element(1, "deuterium", "D", 2.01355321270 * daltons),
            helium=cls.Element(2, "helium", "He", 4.003 * daltons),
            lithium=cls.Element(3, "lithium", "Li", 6.9412 * daltons),
            beryllium=cls.Element(4, "beryllium", "Be", 9.0121823 * daltons),
            boron=cls.Element(5, "boron", "B", 10.8117 * daltons),
            carbon=cls.Element(6, "carbon", "C", 12.01078 * daltons),
            nitrogen=cls.Element(7, "nitrogen", "N", 14.00672 * daltons),
            oxygen=cls.Element(8, "oxygen", "O", 15.99943 * daltons),
            fluorine=cls.Element(9, "fluorine", "F", 18.99840325 * daltons),
            neon=cls.Element(10, "neon", "Ne", 20.17976 * daltons),
            sodium=cls.Element(11, "sodium", "Na", 22.989769282 * daltons),
            magnesium=cls.Element(12, "magnesium", "Mg", 24.30506 * daltons),
            aluminum=cls.Element(13, "aluminum", "Al", 26.98153868 * daltons),
            silicon=cls.Element(14, "silicon", "Si", 28.08553 * daltons),
            phosphorus=cls.Element(15, "phosphorus", "P", 30.9737622 * daltons),
            sulfur=cls.Element(16, "sulfur", "S", 32.0655 * daltons),
            chlorine=cls.Element(17, "chlorine", "Cl", 35.4532 * daltons),
            argon=cls.Element(18, "argon", "Ar", 39.9481 * daltons),
            potassium=cls.Element(19, "potassium", "K", 39.09831 * daltons),
            calcium=cls.Element(20, "calcium", "Ca", 40.0784 * daltons),
            scandium=cls.Element(21, "scandium", "Sc", 44.9559126 * daltons),
            titanium=cls.Element(22, "titanium", "Ti", 47.8671 * daltons),
            vanadium=cls.Element(23, "vanadium", "V", 50.94151 * daltons),
            chromium=cls.Element(24, "chromium", "Cr", 51.99616 * daltons),
            manganese=cls.Element(25, "manganese", "Mn", 54.9380455 * daltons),
            iron=cls.Element(26, "iron", "Fe", 55.8452 * daltons),
            cobalt=cls.Element(27, "cobalt", "Co", 58.9331955 * daltons),
            nickel=cls.Element(28, "nickel", "Ni", 58.69342 * daltons),
            copper=cls.Element(29, "copper", "Cu", 63.5463 * daltons),
            zinc=cls.Element(30, "zinc", "Zn", 65.4094 * daltons),
            gallium=cls.Element(31, "gallium", "Ga", 69.7231 * daltons),
            germanium=cls.Element(32, "germanium", "Ge", 72.641 * daltons),
            arsenic=cls.Element(33, "arsenic", "As", 74.921602 * daltons),
            selenium=cls.Element(34, "selenium", "Se", 78.963 * daltons),
            bromine=cls.Element(35, "bromine", "Br", 79.9041 * daltons),
            krypton=cls.Element(36, "krypton", "Kr", 83.7982 * daltons),
            rubidium=cls.Element(37, "rubidium", "Rb", 85.46783 * daltons),
            strontium=cls.Element(38, "strontium", "Sr", 87.621 * daltons),
            yttrium=cls.Element(39, "yttrium", "Y", 88.905852 * daltons),
            zirconium=cls.Element(40, "zirconium", "Zr", 91.2242 * daltons),
            niobium=cls.Element(41, "niobium", "Nb", 92.906382 * daltons),
            molybdenum=cls.Element(42, "molybdenum", "Mo", 95.942 * daltons),
            technetium=cls.Element(43, "technetium", "Tc", 98 * daltons),
            ruthenium=cls.Element(44, "ruthenium", "Ru", 101.072 * daltons),
            rhodium=cls.Element(45, "rhodium", "Rh", 102.905502 * daltons),
            palladium=cls.Element(46, "palladium", "Pd", 106.421 * daltons),
            silver=cls.Element(47, "silver", "Ag", 107.86822 * daltons),
            cadmium=cls.Element(48, "cadmium", "Cd", 112.4118 * daltons),
            indium=cls.Element(49, "indium", "In", 114.8183 * daltons),
            tin=cls.Element(50, "tin", "Sn", 118.7107 * daltons),
            antimony=cls.Element(51, "antimony", "Sb", 121.7601 * daltons),
            tellurium=cls.Element(52, "tellurium", "Te", 127.603 * daltons),
            iodine=cls.Element(53, "iodine", "I", 126.904473 * daltons),
            xenon=cls.Element(54, "xenon", "Xe", 131.2936 * daltons),
            cesium=cls.Element(55, "cesium", "Cs", 132.90545192 * daltons),
            barium=cls.Element(56, "barium", "Ba", 137.3277 * daltons),
            lanthanum=cls.Element(57, "lanthanum", "La", 138.905477 * daltons),
            cerium=cls.Element(58, "cerium", "Ce", 140.1161 * daltons),
            praseodymium=cls.Element(59, "praseodymium", "Pr", 140.907652 * daltons),
            neodymium=cls.Element(60, "neodymium", "Nd", 144.2423 * daltons),
            promethium=cls.Element(61, "promethium", "Pm", 145 * daltons),
            samarium=cls.Element(62, "samarium", "Sm", 150.362 * daltons),
            europium=cls.Element(63, "europium", "Eu", 151.9641 * daltons),
            gadolinium=cls.Element(64, "gadolinium", "Gd", 157.253 * daltons),
            terbium=cls.Element(65, "terbium", "Tb", 158.925352 * daltons),
            dysprosium=cls.Element(66, "dysprosium", "Dy", 162.5001 * daltons),
            holmium=cls.Element(67, "holmium", "Ho", 164.930322 * daltons),
            erbium=cls.Element(68, "erbium", "Er", 167.2593 * daltons),
            thulium=cls.Element(69, "thulium", "Tm", 168.934212 * daltons),
            ytterbium=cls.Element(70, "ytterbium", "Yb", 173.043 * daltons),
            lutetium=cls.Element(71, "lutetium", "Lu", 174.9671 * daltons),
            hafnium=cls.Element(72, "hafnium", "Hf", 178.492 * daltons),
            tantalum=cls.Element(73, "tantalum", "Ta", 180.947882 * daltons),
            tungsten=cls.Element(74, "tungsten", "W", 183.841 * daltons),
            rhenium=cls.Element(75, "rhenium", "Re", 186.2071 * daltons),
            osmium=cls.Element(76, "osmium", "Os", 190.233 * daltons),
            iridium=cls.Element(77, "iridium", "Ir", 192.2173 * daltons),
            platinum=cls.Element(78, "platinum", "Pt", 195.0849 * daltons),
            gold=cls.Element(79, "gold", "Au", 196.9665694 * daltons),
            mercury=cls.Element(80, "mercury", "Hg", 200.592 * daltons),
            thallium=cls.Element(81, "thallium", "Tl", 204.38332 * daltons),
            lead=cls.Element(82, "lead", "Pb", 207.21 * daltons),
            bismuth=cls.Element(83, "bismuth", "Bi", 208.980401 * daltons),
            polonium=cls.Element(84, "polonium", "Po", 209 * daltons),
            astatine=cls.Element(85, "astatine", "At", 210 * daltons),
            radon=cls.Element(86, "radon", "Rn", 222.018 * daltons),
            francium=cls.Element(87, "francium", "Fr", 223 * daltons),
            radium=cls.Element(88, "radium", "Ra", 226 * daltons),
            actinium=cls.Element(89, "actinium", "Ac", 227 * daltons),
            thorium=cls.Element(90, "thorium", "Th", 232.038062 * daltons),
            protactinium=cls.Element(91, "protactinium", "Pa", 231.035882 * daltons),
            uranium=cls.Element(92, "uranium", "U", 238.028913 * daltons),
            neptunium=cls.Element(93, "neptunium", "Np", 237 * daltons),
            plutonium=cls.Element(94, "plutonium", "Pu", 244 * daltons),
            americium=cls.Element(95, "americium", "Am", 243 * daltons),
            curium=cls.Element(96, "curium", "Cm", 247 * daltons),
            berkelium=cls.Element(97, "berkelium", "Bk", 247 * daltons),
            californium=cls.Element(98, "californium", "Cf", 251 * daltons),
            einsteinium=cls.Element(99, "einsteinium", "Es", 252 * daltons),
            fermium=cls.Element(100, "fermium", "Fm", 257 * daltons),
            mendelevium=cls.Element(101, "mendelevium", "Md", 258 * daltons),
            nobelium=cls.Element(102, "nobelium", "No", 259 * daltons),
            lawrencium=cls.Element(103, "lawrencium", "Lr", 262 * daltons),
            rutherfordium=cls.Element(104, "rutherfordium", "Rf", 261 * daltons),
            dubnium=cls.Element(105, "dubnium", "Db", 262 * daltons),
            seaborgium=cls.Element(106, "seaborgium", "Sg", 266 * daltons),
            bohrium=cls.Element(107, "bohrium", "Bh", 264 * daltons),
            hassium=cls.Element(108, "hassium", "Hs", 269 * daltons),
            meitnerium=cls.Element(109, "meitnerium", "Mt", 268 * daltons),
            darmstadtium=cls.Element(110, "darmstadtium", "Ds", 281 * daltons),
            roentgenium=cls.Element(111, "roentgenium", "Rg", 272 * daltons),
            ununbium=cls.Element(112, "ununbium", "Uub", 285 * daltons),
            ununtrium=cls.Element(113, "ununtrium", "Uut", 284 * daltons),
            ununquadium=cls.Element(114, "ununquadium", "Uuq", 289 * daltons),
            ununpentium=cls.Element(115, "ununpentium", "Uup", 288 * daltons),
            ununhexium=cls.Element(116, "ununhexium", "Uuh", 292 * daltons),
        )

    @classmethod
    def get_symbols(cls, maybe_atomic_number: list[str | int]) -> list[str]:
        return [cls(int(e)).symbol for e in maybe_atomic_number]
    
    @classmethod
    def get_atomic_number(cls, symber: str) -> int:
        return cls._elements[symber].number


Element.initialize()
