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

There are many ways to build up a molecule, for example, you can write it manually from bottom up:

```python
# define atoms in H2O
H1 = Atom('H1')
H2 = Atom('H2')
O  = Atom('O')
# define molecule
H2O = Group('H2O')
# add atoms to the molecule
H2O.addAtoms([H1, H2, O])
H2O.addBond(O, H1)
H2O.addBond(O, H2)
```
Very cumbersome, right? It's just the underlying logic and I will provide more methods to do this job. 
You can assign any property to the atom and retrieve it any time

```python
O.element = 'O'  # Element symbols can be automatically promoted to element class
>>> O.element
>>> < Element oxygen >
```
I fully use the python characteristics of dynamic language
```python
O.anyProperty = 'test'
>>> o.properties
>>> {'_uuid': 140232393903408,
 '_name': 'O',
 '_container': [],
 '_itemType': 'Atom',
 '_bondInfo': {},
 '_element': < Element oxygen >,
 'anyProperty': 'test'}
```
Those names of properties which start with underscore are predefined, in order to avoid conflicts, do not use it.

Also, I use [pint](https://github.com/hgrecco/pint) as unit system.

```python
O.velocity = 1.2 unit.meter/unit.second  # Just a demonstration
```

There are two core classes of the whole molpy: `Atom` and `Group`. All Atom classes form a graph data structure through atomic connections

```python
In [9]: O.bondedAtoms
Out[9]: dict_keys([< Atom H1 >, < Atom H2 >])

In [10]: O.bonds
Out[10]: 
{< Atom H1 >: < Bond < Atom O >-< Atom H1 > >,
 < Atom H2 >: < Bond < Atom H2 >-< Atom O > >}
```
So you can easily traverse the entire structure, for example, get the covalentMap
> covalentMap describes topological distance in molecule. Distance of O->H1 is 1, so (0, 1) in matrix is 1. The order of the matrix is the order in which the atoms are added

```python
In [12]: H2O.getCovalentMap()
Out[12]: 
array([[0, 2, 1],
       [2, 0, 1],
       [1, 1, 0]])
```
`Group` can be regarded as an organization for atoms, or a container of Atoms. One atom can be added to many groups. It's like a group in StarCraft or Red Alert2. 