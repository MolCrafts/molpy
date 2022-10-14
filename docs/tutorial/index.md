# Tutorial

This section will introduce the basic usage of `MolPy`. 

`MolPy`'s hierarchy is clear and the most commonly used components are distribute via top module. You can use it in a `NumPy` style. We can start to use `molpy` by create a `system` and load the data from a file.

```python
import numpy as np
import molpy as mp

system = mp.System()  # create a model
system.load_data('/path/to/data', 'lammps')  # load exist data
assert system.n_atoms == 10  
assert system.n_bonds == 9
```

You can add atoms to the system and establish bonds between them. To ensure flexibility and extensibility, any type of python object can be used as an attribute of atom or bond. 

```python
system.add_atom(id=10, 
                type='C', 
                mass=12.011, 
                position=np.array([0.0, 1.0, 0.0]))  # add an atom
system.add_atoms(id=[11, 12], 
                 type=['H', 'H'], 
                 mass=[1.008, 1.008], 
                 position=np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))  # add multiple atoms
system.add_bonds([(10, 11), (11, 12)], type=['CH', 'CH'])  # add bond by atom index in system
assert system.n_atoms == 13
assert system.n_bonds == 10
```

You can easily access any atom or bond in the system by index or id, and an `Atom` or `Bond` object will be returned. 

```python
c = system.get_atom(10)  # get atom by index
h1 = system.get_atom_by_index(11)  # get atom by index

assert isinstance(c, mp.Atom)
assert h1['type'] == 'H'
```

For a large molecular dynamics model, you can also use `numpy` style to slice or index the system. 

```python
import numpy.testing as npt

state = system.state  # convert object-oriented data to numpy array
atoms = state.atoms  # get all atoms info.
npt.assert_equal(atoms['id'], np.arange(13)) 
```

`MolPy` also support classical forcefield. It can store forcefield parameters and calculate energy and force of molecular. You can load `XML` file which compatiable with `OpenMM` or add parameters manually. 

```python

ff = system.forcefield
ff.def_atom(typeName='C', typeClass='CH2', charge=-0.12)
ff.def_atom(typeName='H', typeClass='CH2', charge=0.06)
ff.def_bond(ff.BondStyle.harmonic, 'C', 'H', length=1.53, k=800)
```

As you can see, we try to keep you from manually creating the various components in `molpy` as much as possible.