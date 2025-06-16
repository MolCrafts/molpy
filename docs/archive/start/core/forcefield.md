# ForceField

`ForceField` class is designed to manage potential functions in molecular simulations. It provides a unified framework for defining, organizing, querying, and exporting interaction parameters such as bonds, angles, dihedrals, and nonbonded terms.

With a modular and extensible interface, this class helps users build and manipulate force field parameters in a clear and maintainable way. It is particularly useful for tasks such as parameter fitting, model calibration, and format conversion, serving as a central component in simulation workflows.

Let's start with an empty `ForceField` object, and define a few potentials manually:
```python

ff = mp.ForceField()
full_style = ff.def_atomstyle("full")
full_style.def_type("H", mass=1.008, charge=0.417)
full_style.def_type("O", mass=15.9994, charge=-0.834)
# Define artificial harmonic bond potential
harmonic_bond = ff.def_bondstyle("hamonic")
harmonic_bond.def_type("H", "O", k=1.0, r0=1.0)

# Define a lj pair
lj = ff.def_pairstyle("lj", mix="geometric")
lj.def_type("H", "H", epsilon=0.1, sigma=1.0)
lj.def_type("O", "O", epsilon=0.1, sigma=1.0)
```

The `ForceField` class manages a collection of interaction styles, referred to as `Style` objects. Each `Style` can contain multiple `Type` instances, representing specific parameter sets for different interaction patterns.

A `Style` can have both global parameters and parameters specific to each `Type`. These parameters can be provided as ordered arguments or as keyword arguments. Global parameters of a `Style` are stored in a dictionary, while `Type` parameters are stored in a list to preserve their order. Despite the list-based storage, you can still access or set parameters by name using keywords, thanks to the underlying mapping.

Internally, `AtomType` is identified solely by its name. Many-body interaction types hold references to `AtomType` instances; any modification to an `AtomType`’s parameters will automatically be reflected in all associated many-body types. The same design applies to `Style` and its associated `Type` instances — they share references rather than duplicating parameter data.

This design ensures consistency and reduces redundancy when updating force field parameters across different interaction terms. It also provides flexibility for parameter manipulation, whether you're assigning values by position or by name.

Many of the `ForceField` methods can be used to query the parameters of the force field. For example, you can use `bondstyle = ff.get_bondstyle(name)` to retrieve the atom style for a specific pair of atom types, or `bondstyle.get_bond_style("H", "O")` to get the bond style for a specific pair of atom types. You can also use `ff.get_bond_style("H", "O").get_param("k")` to get the value of a specific parameter for a specific bond style.

## Type label is mandatory
The design of the `ForceField` borrow from [LAMMPS type label](https://docs.lammps.org/Howto_type_labels.html). Using type labels instead of numeric types can be advantageous in various scenarios, not only for readability but also for generality. For example, when typifying different molecules and then merge them, using type labels allows for a more flexible and extensible approach. This is particularly useful in cases where the number of atom types may change or when dealing with complex systems that require a more descriptive representation of atom types.