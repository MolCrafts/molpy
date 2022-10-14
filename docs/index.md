# Welcome to MolPy

`MolPy` aims to build a package to manipulate and analyze molecular model used in computational chemistry.

The core of `MolPy` is a flexible and efficient data structure to manage atomic and topological information of molecular. You can quickly add, delete, modify and quert the information of molecular, also can slice or index data with the help of numpy.

`MolPy` also support classical forcefield. It can store forcefield parameters and calculate energy and force of molecular. The goal at this version is to automatically match `bond` `angle` and other topological data through defined atomic types.

You can also use `MolPy` to build molecular dynamics simulation with the original input and output modules. 

Around this data structure, `MolPy` provides a set of crude tools to optimize the model. It WILL provide `MD`, `MC` and molecule packing tools in the future.

## Installation

`MolPy` does not provide pip or conda installation. You should git clone the code and install it manually. 

