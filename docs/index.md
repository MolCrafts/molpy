# Welcome to molpy

> author: Roy Kid

> contact: lijichen365@126.com

> date: 2021-10-23

> version: 0.0.1

`molpy`  want to build a data structure used to describe molecules in computational chemistry, just like numpy in data science

## Quick start

使用这个工具非常容易, 首相像numpy一样导入

```python
import numpy as np
import molpy as mp
```

`molpy`有两个核心的类用来描述分子, 一个是`Atom`, 另一个是`Group`, 其他的都是为这两个类服务的. 首先你需要理解, 每一个分子都是由很多原子键接而成, 因此整个分子形成一个相互连通的图. 每个原子将储存着与它相邻的原子. 一堆原子不能零散地乱放, `Group`类将是他们的容器. 

如果手动地建立一个模型, 应该从底向上地操作

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
非常麻烦, 我们也不会这么做. 这只是指出了底层的逻辑, 和这两个关键的类之间的联系. 我们将提供一系列的生成函数来帮助你拜托这种繁琐的工作. 例如我们可以直接从各种分子动力学文件中读取模型信息

```python
lactic = mp.fromPDB('lactic.pdb')
polyester = mp.fromLAMMPS('pe.data')
benzene = mp.fromSMILS('c1ccccc1')
```
针对量化计算, 每个原子都有它的元素. 因此`Atom`的`element`属性是一个非常特别的类, 它提供了标准的元素信息. 当你设置它的元素符号或者名称, 他将自动转化为元素类的实例
```python
O.element = 'O'
>>> O.element
>>> <Element oxygen>
```

针对分子模拟, 每个原子都有它的原子类型. `atomType`是由`forcefield`设定并全局共享的. 例如一个水分子的两个氢原子应该是类型相同的. 你不会希望修改一个氢的参数而另一个不发生变化. 
```python
# 实例化一个力场类
ff = ForceField('tip3p')
# 定义原子类, 返回实例并赋给原子
H1.atomType = ff.defAtomType('H2O', charge=0.3*mp.unit.)
>>> H1.properties
>>> {
    'name': 'H1',
    'atomType': 'H2O',
    'element': 'H'
}
```
你可以看到我们这里也内置了单位系统(由pint提供), 实现了单位换算化简等功能. 同样, 这个操作也不会需要手动操作. 我们在`forcefield`中提供了模板匹配机制, 提前定义好一个模板, 就可以直接把全部属性从模板上转移到分子中.

不仅原子上附加这属性, 原子之间的键接, 键角和二面角也由相应的参数. 化学键在定义拓扑结构的时候已经生成
```python
>>> H2O.getBond(H1, O)
>>> < Bond H1-O >
>>> atom, btom = < Bond H1-O >
>>> atom
>>> < Atom H1 >
>>> assert H1.bondto(O) == O.bondto(H1) == H2O.getBond(H1, O)
>>> True
```

通过拓扑搜索, 可以生成键角和二面角

```python
>>> H2O.searchAngles()
>>> [< Angle H1-O-H2 >]
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

## roadmap:

### 核心工作
1. 数据结构: 描述分子的数据结构
1. 分子建模: 给出原子和键接定义生成一个分子
1. 分子拼接: 复用分子片段生成大分子
1. 层级结构: 快速索引分子中的片段
1. 拓扑搜索: 已知键接信息生成键角二面角等信息
1. 序列化  : 返回与语言无关的数据结构共其他工具调用
1. 力场分配: 根据原子化学环境判断原子类型
1. 模板匹配: 将分子和力场中模板相匹配
1. 结构优化: 梯度下降地寻找分子能量较低的构型
1. packing: 将分子密铺在模拟盒子中

### 外围工作

* 数据输入输出: 读入输出其他格式的文件
* 调用其他程序: 直接调用其他QM/MM程序
* 脚本核心结构: 人类友好的脚本API和存储
* 脚本输入输出: 生成不同软件需要的脚本
* 分析模块构建: 预制的分子结构分析工具
* 分析模块扩展: 更容易增加分析功能插件

### 锦上添花

* 界面