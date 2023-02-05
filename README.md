# molpy

MolPy is a data structure and toolkits for molecular dynamics simulation. It's provide a simple and flexible way to build your own molecular dynamics simulation and access your trajectory. When you use this elegant and well-designed python package, you can fully focus on your own research and leave tedious work to molpy, such as file parsing, data management, visualization etc.

This package depends on few external package except Numpy and [Chemfiles](https://chemfiles.org/chemfiles.py/latest/index.html#), but it may rely on higher version of Python.

> :laughing: This project is still under active development. If you have any suggestions, please feel free to contact me.

We also provide an interactive visualization package [molvis](https://github.com/Roy-Kid/molvis). In this work, we use production-level game engine from Microsoft [Babylonjs](https://www.babylonjs.com/) to visualize system. The highlight is you can use Ipython to manipulate the system on-the-fly and see the result immediately. It's very useful for debugging and research. 

## roadmap

### 0.0.1
  - [x] refactor core data structure;
  - [x] static and dynamic data structure;
  - [x] read and write data via Chemfiles;
  - [x] trilinic box;
  - [x] forcefield;
  - [ ] modelling;
  - [ ] neighorlist;
  - [ ] interactivate visualization API;
  - [ ] minimizer;
  - [ ] simple example;
  - [ ] document;

### 0.0.2
  - [ ] parallel dataloader;
  - [ ] graph neural network support;
  - [ ] SMARTS expression;
  - [ ] potential function generator;
  - [ ] MD engine API;
