# Welcome to molpy

> author: Roy Kid

> contact: lijichen365@126.com

> date: 2021-11-22

> version: 0.0.2

`molpy`want to build a data structure used to describe molecules in computational chemistry, just like numpy in data science

## Introduction

`molpy`是一套提供给量子化学/分子动力学的分子数据结构, 希望能做到计算化学的numpy. 此外`molpy`提供了分子建模, 拓扑搜索, 力场匹配等功能. `molpy`架构清晰简洁, 核心结构为`Atom`, `Group`, 及`Molecule`类, 通过把`atom`视为node, `bond`看作edge, 将分子描述为图结构. `molpy`支持模板匹配建立拓扑结构(gmx/openmm-style), 搜索angle/dihedral(moltemplate-style), 力场匹配(将atom/bond/angle等参数附加到结构上), 大大方便您构建复杂体系模型. `molpy`合理规划了命名空间, 可以迅速地索引任何一级的原子('atom7@group5@molecule3')并且进行操作(atom.move((x,y,z))). 每一个数据类在建立之后都可以通过复制来实现模块化建模, 代码复用使您的脚本更加简介. `molpy`提供了丰富的输入输出接口, 可以将体系输出成各种QM/MM计算工具. `molpy`非常灵活, 极其容易拓展, 并且企图将优秀软件包的思想汇聚, 让您一站式从建模到模拟到数据处理, 无需考虑目标软件, 仅需专注于科学本身. 

## Quick start

像numpy一样导入
```python
import numpy as np
import molpy as mp
```
像搭建乐高®一样搭建分子
```python
# 定义一个原子
H1 = Atom('H1')
H2 = Atom('H2')
O = Atom('O')
# 定义一个容器
H2O = Group('H2O')
H2O.addAtoms([H1, H2, O])
# 定义键接关系
H2O.addBondByName('H1', 'O')
H2O.addBondByName('H2', 'O')
```
自动搜索拓扑结构
```python
H2O.searchAngles()
H2O.searchDihedrals()
```
一次定义参数, 直接应用全局
```python
ff = ForceField('SPCE')
H = ff.defAtomType('H', mass=1.001, charge=0.3, element='H')
O = ff.defAtomType('O', mass=15.9994, charge=-0.6, element='O)
OHbond = ff.defBondType('OH', style='harmonic', r0=0.99, k=1.12)
HOHangle = ff.defAngleType('HOH', style='harmonic', theta0=108, k=0.8)

ff.render(H2O)
```
完全基于面向对象编程, 信息立等可取
```python
>>> H2O.getAtomByName('H1')
<Atom H1>
>>> H2O.getAtomByName('H1').position
array([1, 2, 3])
>>> H2O.bonds
[<Bond H1-O>, <Bond H2-O>]
>>> H2O.bonds[0].length
1.08
```
可对任意分子/基团进行空间调整
```python
H1.move((1,2,3))       # 平移一个矢量
H2O.rot(132, 1, 0, 0)  # 欧拉角旋转
```
使用Python构建您的分子就是如此简单!

## Quick Installation 

暂时还没有推送到conda和pip上, 因此需要手动clone仓库
在自己的代码文件开头添加路径
```python
import sys
sys.path.append("/home/roy/work/molpy")
import molpy as mp
```

## 设计理念

**命名空间**: 每一级都是独立命名空间, 因此可以按照层级和名字查找任意一个对象

**面向对象**: 完全面向对象, 所有信息直接附加在python实例上

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

* 数据输入输出: 读入输出其他格式的文件  (逐步完善中)
* 调用其他程序: 直接调用其他QM/MM程序  (准备封装dpdispatcher)
* 脚本核心结构: 人类友好的脚本API和存储  (脚本部分独立为ChemRosetta)
* 脚本输入输出: 生成不同软件需要的脚本  
* 分析模块构建: 预制的分子结构分析工具  (增加Frame*类处理轨迹)
* 分析模块扩展: 更容易增加分析功能插件

### 锦上添花

* 界面