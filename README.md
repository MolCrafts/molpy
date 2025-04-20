# molpy

MolPy is a data structure and toolkits for molecular simulation. It's provide a simple and flexible way to build your own molecular system and access module and trajectory. When you use this elegant and well-designed python package, you can fully focus on your own research and leave tedious work to molpy, such as file parsing, data management, optimize structure etc.

This package depends on few external packages
* [numpy](https://github.com/numpy/numpy)
* [chemfiles.py](https://github.com/chemfiles/chemfiles.py)
* [python-igraph](https://github.com/igraph/python-igraph)

> :laughing: This project is still under active development. Any suggestions and feature requests are welcome.

## roadmap

  - [x] static and dynamic data structure;
  - [x] read and write data via Chemfiles;
  - [x] trilinic box;
  - [ ] celllist & neighorlist;
  - [ ] potential function;
  - [x] forcefield;
  - [ ] optimizer(minimizer, MD etc.);
  - [ ] modelling;
  - [ ] typification;
  - [ ] SMARTS/SMILES expression;
  - [ ] interactivate visualization API;
  - [ ] plugin system;
  - [ ] documentation;

  After all functions are implemented and stable, we will abstract the core part of the package to high-performance C++, and compile to other language.

## ecosystem

### potential training

We are developing a universial potential training platform [https://github.com/MolCrafts/molpot]

### visualization

We **will** also provide an interactive visualization package [molvis](https://github.com/Roy-Kid/molvis). In this work, we use production-level game engine from Microsoft [Babylonjs](https://www.babylonjs.com/) to visualize system. The highlight is you can use Ipython to manipulate the system on-the-fly and see the result immediately. It's very useful for debugging and research. 