# molpy
A data structure used to describe molecules in computational chemistry, just like numpy in data science

## Installation

The package haven't uploaded to the conda yet. The way you experience is download source code

```
git clone https://github.com/Roy-Kid/molpy
```
If you don't want to set environment variables, create a .py file in the root of molpy, alongside molpy subfolder.
Or, you can add path manually.

```
import sys
sys.path.append('path/to/molpy')
import molpy as mp
```

## Quick Start

Just like numpy, you should import molpy at first

```python
import numpy as np
import molpy as mp
```

`Molpy` has two core classes used to describe molecules, one is `Atom`, the other is `Group`, and the others are for these two classes. First of all, you need to understand that each molecule is composed of Many atoms are bonded together, so the whole molecule forms a connected graph. Each atom will store the atoms adjacent to it. A bunch of atoms cannot be scattered randomly, the `Group` class will be their container.

If you build a model manually, you should operate it from bottom to top

```python
# define atoms in H2O
H1 = Atom('H1')
H2 = Atom('H2')
O  = Atom('O')

# define topology connection
O.bondto(H1)
O.bondto(H2)

# define molecule
H2O = Group('H2O')

# add atoms to the molecule
H2O.addAtoms([H1, H2, O])
H2O.addBond(O, H1)
H2O.addBond(O, H2)
```
It's very troublesome, and we won't do so. This just points out the relationship between the underlying logic and these two key classes. We will provide a series of factory functions to help you get rid of this cumbersome work. For example, we can read the model information directly from various molecular dynamics file formats

```python
lactic = mp.fromPDB('lactic.pdb')
polyester = mp.fromLAMMPS('pe.data')
benzene = mp.fromSMILS('c1ccccc1')
```
For quantum mechanics, each atom has its element. Therefore, the `element` attribute of `atom` is a very special class, which provides standard element information. When you set its element symbol or name, it will be automatically converted to an instance of the element class

```python
O.element = 'O'  # Element symbols can be automatically promoted to element class
>>> O.element
>>> < Element oxygen >
```

For molecular simulation, each atom has its atomic type. `atomType` is set by `forcefield` and shared globally. For example, two hydrogen atoms of a water molecule should be of the same type. You don't want to modify the parameters of one hydrogen without changing the other

```python
# initialize a forcefield
ff = ForceField('tip3p')
# define atomType, return handle and assign it to H1
H1.atomType = ff.defAtomType('H2O', charge=0.3*mp.unit.coulomb)
>>> H1.properties
>>> {
    'name': 'H1',
    'atomType': 'H2O',
    'element': 'H'
}
```
As you can see, we also have a built-in unit system (powered by [pint](https://github.com/hgrecco/pint)) to realize the functions of unit conversion and simplification. Similarly, this operation does not need manual operation. We provide a template patching mechanism in `forcefield`. After defining a template in advance, we can directly transfer all attributes from the template to the molecule

Not only do atoms attach this attribute, the bonding between atoms, bond angle and dihedral angle are also determined by the corresponding parameters. Chemical bonds have been generated when defining the topology

```python
>>> H2O.getBond(H1, O)
>>> < Bond H1-O >
>>> atom, btom = < Bond H1-O >
>>> atom
>>> < Atom H1 >
>>> assert H1.bondto(O) == O.bondto(H1) == H2O.getBond(H1, O)
>>> True
```

Through topology search, key angles and dihedral angles can be generated

```python
>>> H2O.searchAngles()
>>> [< Angle H1-O-H2 >]
```

For molecular graph neural networks, we can also give a `covalentmap` to describe the topological distance within molecules`

```python
atomlist, covalentMap = H2O.getCovalentMap()
>>> atomlist
>>> [< Atom H1 >, < Atom H2 >, < Atom O >]
>>> covalentMap 
>>> [[0 2 1]
     [2 0 1]
     [1 2 0]]
```

## roadmap:
### Core work
1. Data structure: describe the data structure of molecules
1. Molecular modeling: give the definition of atom and bonding to generate a molecule
1. Molecular splicing: reuse molecular fragments to generate macromolecules
1. Hierarchy: quickly index fragments in molecules
1. Topology search: generate key angle, dihedral angle and other information when the bonding information is known
1. Serialization: return language independent data structures and call other tools
1. Force field distribution: determine the atomic type according to the atomic chemical environment
1. Template matching: match the molecule with the template in the force field
1. Structural optimization: find configurations with lower molecular energy by gradient descent
1. Packing: lay molecules tightly in the simulation box

### Peripheral work
*Data input and output: read in and output files in other formats
*Call other programs: call other QM / mm programs directly
*Script core structure: human friendly script API and storage
*Script input and output: generate scripts required by different software
*Analysis module construction: Prefabricated molecular structure analysis tool
*Analysis module extension: it is easier to add analysis function plug-ins
###Icing on the cake
*Interface