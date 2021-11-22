# Welcome to molpy

> author: Roy Kid

> contact: lijichen365@126.com

> date: 2021-11-22

> version: 0.0.2

`molpy`want to build a data structure used to describe molecules in computational chemistry, just like numpy in data science

## Introduction

`molpy`是一套提供给量子化学/分子动力学的分子数据结构, 希望能做到计算化学的numpy. 此外`molpy`提供了分子建模, 拓扑搜索, 力场匹配等功能. `molpy`架构清晰简洁, 核心结构为`Atom`, `Group`, 及`Molecule`类, 通过把`atom`视为node, `bond`看作edge, 将分子描述为图结构. `molpy`支持模板匹配建立拓扑结构(gmx/openmm-style), 搜索angle/dihedral(moltemplate-style), 力场匹配(将atom/bond/angle等参数附加到结构上), 大大方便您构建复杂体系模型. `molpy`合理规划了命名空间, 可以迅速地索引任何一级的原子('atom7@group5@molecule3')并且进行操作(atom.move((x,y,z))). 每一个数据类在建立之后都可以通过复制来实现模块化建模, 代码复用使您的脚本更加简介. `molpy`提供了丰富的输入输出接口, 可以将体系输出成各种QM/MM计算工具. `molpy`非常灵活, 极其容易拓展, 并且企图将优秀软件包的思想汇聚, 让您一站式从建模到模拟到数据处理, 无需考虑目标软件, 仅需专注于科学本身. 

## Quick start

使用这个工具非常容易, 首先像numpy一样导入

```python
import numpy as np
import molpy as mp
```

与python的思想一样, `molpy`将所有的信息储存为一个python实例, 其基类为`Item`(下文中`Item`泛指一切`molpy`对象, 其中首字母大写指代类, 小写则是实例). 每一个分子都是由很多原子(`Atom`)键接(`atom.bondto(btom)`)而成, 每个原子将储存着与它相邻的原子(`atom._bondInfo`), 整个分子形成一个相互连通的图(`Group/Molecule`). 为了理解这种层级结构, 我们手动新建一个水分子

```python
# 定义一个原子
H1 = Atom('H1')
H2 = Atom('H2')
O = Atom('O')

# 定义键接关系
O.bondto(H1)
O.bondto(H2)

# 定义一个容器
H2O = Group('H2O')
H2O.addAtom(H1)
H2O.addAtom(H2)
H2O.addAtom(O)
```
`Group`中储存了原子和拓扑信息, 同时也是一级命名空间. `Group`中的`Atom.name`不允许有重复, 否则难以查找. 每一个`atom`可以动态附加各种原子信息, 例如`atom.velocity`等. 有一些属性是特殊的, 例如`atom.atomType`只能接受`atomType`实例, 这个实例全局共享, 具有相同的`atomType`的原子的属性都会同步更改. `Atom`的`element`属性也是一个非常特别的类, 它提供了标准的元素信息. 当你设置它的元素符号或者名称, 他将自动转化为元素类的实例, 只其中提供了原子序数, 相对原子质量等各种参数. 

```python
O.element = 'O'
>>> O.element
>>> <Element oxygen>
```

为了保持对各种信息的追踪, 我们通常用`forcefield`类来管理各种类型

```python
ff = mp.ForceField('h2o')
H = ff.defAtomType('H', mass=1.001, charge=0.3, element='H')
O = ff.defAtomType('O', mass=15.9994, charge=-0.6, element='O)
OHbond = ff.defBondType('OH', style='harmonic', r0=0.99, k=1.12)
HOHangle = ff.defAngleType('HOH', style='harmonic', theta0=108, k=0.8)
```
> 在以后的版本中我们提供单位模块, 方便单位的转换和换算. 
我们现在有了水的各个原子信息, 和力场信息. 针对拓扑结构, `molpy`支持两种风格的构建: moltemplate 和 openmm 风格

moltemplate风格指, 我们手动提供原子间的键接信息, `molpy`负责搜索`angle/dihedral/improper`:

```python
H2O.addBondByName('O', 'H1')
H2O.addBondByName('O', 'H2')
# 或者根据加入原子的顺序
# H2O.addBondByIndex(0, 2)
# H2O.addBondByIndex(1, 2)
```

openmm风格指, 我们提供一个分子的模板, `molpy`根据模板补全拓扑信息:

```python
ff.registTemplate(H2OT)  # H2OT 是一个well-defined的分子, 定义方法和H2O相同;
template = ff.match(H2O) # 搜索到匹配的模板, 
ff.patch(H2O, template)  # 通过模板补全分子.
```

这样, 我们可以将体系中所有分子的拓扑信息补充完整. 接下来是将力场中的信息复制到相对应的对象中, 例如`atomType`, `bondType`等

```python
ff.render(H2O)
>>> H2O.natoms 
>>> 3
>>> H2O.atoms[0].name
>>> 'H1'
>>> H2O.atoms[2].mass
>>> 1.001
>>> H2O.getBonds()
>>> [<Bond H1-O>, <Bond H2-O>]
>>> H2O.getAngles()
>>> [<Angle H1-O-H2>, ]
```

对于分子的图神经网络, 我们还可以给出描述分子内拓扑距离的`covalentMap`

```python
atomlist, covalentMap = H2O.getCovalentMap()
>>> atomlist
>>> [< Atom H1 >, < Atom H2 >, < Atom O >]
>>> covalentMap 
>>> [[0 2 1]
     [2 0 1]
     [1 2 0]]
```

为了输出到MD软件所需要的格式, 我们需要提供一个`System`类来储存包括模拟盒子, 周期性边界等信息. 同时, 以上的工作也可以通过封装好的命令自动完成

```python
system = mp.System("PE with Np=120")
system.cell = mp.Cell(3, "ppp", xlo=0, xhi=100, ylo=0, yhi=100, zlo=0, zhi=100)
system.forcefield = ff = mp.ForceField("LJ ff", unit="LJ")

system.addMolecule(H2O)  # 如果不是Molecule类, 会自动提升
system.complete()
toLAMMPS('H2O.data', system, atom_style="full")

```

## roadmap:

### 核心工作
1. 数据结构: 描述分子的数据结构 √
1. 分子建模: 给出原子和键接定义生成一个分子 √
1. 分子拼接: 复用分子片段生成大分子 √
1. 层级结构: 快速索引分子中的片段 √
1. 拓扑搜索: 已知键接信息生成键角二面角等信息 √
1. 序列化  : 返回与语言无关的数据结构共其他工具调用 
1. 力场分配: 根据原子化学环境判断原子类型 × (不考虑实现)
1. 模板匹配: 将分子和力场中模板相匹配 √
1. 结构优化: 梯度下降地寻找分子能量较低的构型 × (优先实现)
1. packing: 将分子密铺在模拟盒子中 × (优先实现)

### 外围工作

* 数据输入输出: 读入输出其他格式的文件
* 调用其他程序: 直接调用其他QM/MM程序
* 脚本核心结构: 人类友好的脚本API和存储
* 脚本输入输出: 生成不同软件需要的脚本
* 分析模块构建: 预制的分子结构分析工具
* 分析模块扩展: 更容易增加分析功能插件

### 锦上添花

* 界面