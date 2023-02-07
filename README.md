# molpy

MolPy is a data structure and toolkits for molecular dynamics simulation. It's provide a simple and flexible way to build your own molecular dynamics simulation and access your trajectory. When you use this elegant and well-designed python package, you can fully focus on your own research and leave tedious work to molpy, such as file parsing, data management, visualization etc.

This package depends on few external package except Numpy and [Chemfiles](https://chemfiles.org/chemfiles.py/latest/index.html#), but it may rely on higher version of Python.

> :laughing: This project is still under active development. If you have any suggestions, please feel free to contact me.

We also provide an interactive visualization package [molvis](https://github.com/Roy-Kid/molvis). In this work, we use production-level game engine from Microsoft [Babylonjs](https://www.babylonjs.com/) to visualize system. The highlight is you can use Ipython to manipulate the system on-the-fly and see the result immediately. It's very useful for debugging and research. 

## roadmap

### 0.0.1: novice

    In this version, we will complete the core data structure of the project, determine the style of the API, and fully test the correctness of the code. 

  - [x] refactor core data structure;
  - [x] static and dynamic data structure;
  - [x] read and write data via Chemfiles;
  - [x] trilinic box;
  - [x] forcefield;
  - [x] typification;
  - [ ] modelling;
  - [ ] neighorlist;
  - [ ] interactivate visualization API;
  - [ ] minimizer;
  - [ ] simple example;
  - [ ] document;

### 0.0.2: assist

    Based on the novice version, we will add some useful tools to assist the user to apply the package to their own research and do what other packages can't do.

  - [ ] parallel dataloader;
  - [ ] graph neural network support;
  - [ ] SMARTS/SMILES expression;
  - [ ] potential function generator;
  - [ ] MD engine API;

### 0.0.3: ringmaster

    If possible, in this version, we will seamlessly connect with molecular simulation package. For example, we can provide a simple way to submit the simulation to the cluster and monitor the progress of the simulation. Once molpy script build, use it in any simulation package.