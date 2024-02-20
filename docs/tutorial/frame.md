# Frame

The core of molpy is `Frame` class. It is the data structure that molpy uses to store and manipulate molecular data. The `Frame` stores a variety of information, including atomic, topological, and geometric information. Each frame can completely describe the state of a system at a moment in time like a snapshot. 

::: NOTE:

`System` only hold ONE frame at a time, and switch between frames. If you want to store multiple frames, you need to use `Trajectory` class.

Here are two types of `Frame`. The first one is called `DynamicFrame`, which is flexible aims to edit the model. The second one is called `StaticFrame`, which is immutable and is used for lookup and indexing. The two types of `Frame` are interchangeable. 

## DynamicFrame

`DynamicFrame` is a collection of `Dict`.

For per-atom data, `molpy` create an `Atom` object. Each `Atom` has an unique ID, and used as the key of the `dict`. `Atom` object also is a `dict`, so the hiechary is:

```python
frame = DynamicFrame()
frame._atoms = {}
frame._atoms[atom_id] = Atom(
    attribA = 1  # int or float
    attribB = 'A' # str
    attribC =  [1, 2, 3] # list or np.array
)
```
::: NOTE

Those objects such as `Atom`, `Bond` etc. is called `Item`. Again, all those trivial classes don't even need to be created by us manually. 

To record topological information, `DynamicFrame` initiaizes a `Topology` object. `Topology` is a graph structure and each node is the atom ID of the `Atom` item. The edge is the bond ID of the `Bond` item. Similarly, we can take the unique three-body or four-body ID from the topology as the angle and dihedral. `DynamicFrame` has dict of `Bond`, `Angle`, and `Dihedral`(maybe `Improper` in the future). The key is the ID of the item and the value is the item itself. So when you add a bond to the frame, following things will happen:

``` python
# add a bond
dframe.add_bond(1, 2, attribA=1) # add a bond between atom 1 and 2
atom1 = dframe.atoms[1]  # pick up the second atom
atom2 = dframe.atoms[2]
bond = mp.Bond(atoms1, atoms2, attribA=1)  # create a bond
dframe._bonds[bond.id] = bond  # register the bond
dframe._topo.add_bond(atom1.id, atom2.id, bond.id)  # add to topo

# retrieve the bond by
# bond = frame.get_bond(1, 2)
atom1 = frame.atoms[1]
atom2 = frame.atoms[2]
bond_id = frame._topo.get_bond(atom1.id, atom2.id)
bond = frame._bonds[bond_id]
```
The `DynamicFrame` is memory inefficiency but easy to CRUD. But sometime we don't need to insert or delete items frequently. For example, we need to get the all coordinates or atom types as an array. or create a mask to retrieve the atoms that satisfy some conditions. In this case, we can use `StaticFrame` to speed up the process.

## StaticFrame

`StaticFrame` is a collection of `np.array`(more specific, [structured array](https://numpy.org/doc/stable/user/basics.rec.html)). It is immutable and aligned and is used for lookup and indexing. The `StaticFrame` is memory efficient but hard to CRUD. 

Although the design pattern tells us that both classes derived from the `Frame` class should implement a method like `add_atom`, we don't want to implement it in the `StaticFrame`. Because the `StaticFrame` is immutable, we don't encourage you to CRUD with `StaticFrame`(even create it manually). So far, you can use a class method to create, or convert from `DynamicFrame`. 

``` python
sframe = dframe.to_static()
# or you can do:
sframe = mp.StaticFrame.from_dict({
    'charge': np.ones(n_atoms), 
    'xyz': np.random.random((n_atoms, ndim)), 
})
```

Our focus is on how to access the data. For atomic info, we may want to fetch data by row(a set of atom) or by column(an attribute of atoms)

``` python
# get a subset of atoms
atoms = sframe[: 3]  # get first three atoms
"""
array([(1., [0.48002889, 0.65715661, 0.53987016]),
       (1., [0.852534  , 0.44212008, 0.76396292]),
       (1., [0.73687783, 0.30907199, 0.26108508])],
      dtype=[('charge', '<f8'), ('xyz', '<f8', (3,))])
"""
# get attribute of atoms
charge = sframe['xyz']
"""
array([[0.48002889, 0.65715661, 0.53987016],
       [0.852534  , 0.44212008, 0.76396292],
       [0.73687783, 0.30907199, 0.26108508],
       [0.66011495, 0.29811501, 0.0325941 ],
       [0.80849894, 0.95918574, 0.60295134]])
"""
# get attributes of atoms
sframe[['charge', 'xyz']]
"""
array([[1.        , 0.48002889, 0.65715661, 0.53987016],
       [1.        , 0.852534  , 0.44212008, 0.76396292],
       [1.        , 0.73687783, 0.30907199, 0.26108508],
       [1.        , 0.66011495, 0.29811501, 0.0325941 ],
       [1.        , 0.80849894, 0.95918574, 0.60295134]])
"""
```
