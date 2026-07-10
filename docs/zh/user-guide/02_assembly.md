# 组装

从单体长出一条聚合物链、把弛豫好的熔体交联成网络、把一条链首尾闭合成大环——这是三件不同的活。你在研究的不同阶段才会用到它们，它们回答的也是不同的问题。它们的共同点不在于活本身，而在于 MolPy 要你把哪些话说清楚才肯动手。

## 三件不同的活，同样的三项输入

无论你在搭什么，MolPy 都只问同样三件事。

**哪些原子允许反应。** 你在分子上按名字把它们标出来。

**反应做了什么。** 你把它写下来，用一条 reaction SMARTS，只写一次。

**哪些被标记的位点真的配上对。** 链把每个单体和下一个配对。交联剂把空间上恰好靠近的位点配对。大环把某个已经连通的东西的两端配对。

三件活里只有第三项输入不同，而 MolPy 把这一点落到了实处：干活的是同一个 assembler，你只需要递给它一个 **Selector**，它只回答这一个问题。`TopologySelector` 按链上的相邻关系配对，`RandomSelector` 在距离截断之内配对。你也可以写一个按你喜欢的方式配对的。这个选择下游的一切——摆放各个片段、编辑图、修复每根新键周围的力场类型——都是同一条你永远不必碰的代码路径。

## 组装就是粘贴、编辑和修复，而且全都是局部的

**assembler 把分子粘贴进同一个世界，在它的 `select` 钩子指定的地方施加你的反应，再修复每根新键邻域内的力场类型。**

这句话里真正干活的词是*邻域*。组装过程中没有任何一步会走遍整个体系。往链上加第一千个单体的代价，和加第二个时一样，因为被检查的原子只有正在形成的那根键附近的那些。

> **底层原理：为什么没有任何东西被全局重算**
>
> 力场根据一个原子的周围环境判定它的类型，范围只到几根键之外——对 GAFF 大约两根键，对带芳环模式的力场大约三根键。所以形成一根键，只可能改变距它这个范围以内的原子的类型。更远的地方可以证明不受影响。
>
> 因此 MolPy 在每根新键周围切出一个球，把球内的原子重新类型化，只把这些原子写回去。切出的球比它写回的原子范围更宽，因为处在写回区边缘的原子仍然需要让自己完整的周围环境落在视野里，才能被正确地类型化。切多宽是从力场自己的模式里读出来的；它从来不是一个可调设置。
>
> 只有两个后果会渗进 API，也只有两个。如果一条模式追问"这个原子在*任何*环上吗？"，那这个力场根本没有有界的邻域——闭合一根键就可能让上千个原子落到某个环上——这样的 typifier 会被直接拒绝。而 MolPy 无法自省的 typifier，比如 AmberTools 包装器，必须由你告诉它看多远。

## assembler 不是什么

它**不是反应引擎**。它从不解析 SMARTS，从不匹配模式，从不改写一根键。这些全部由 `Reaction` 完成。assembler 决定*哪些*位点反应；反应决定它们反应时*发生什么*。

它**不是端口系统**。这里没有 `<` 和 `>`，没有头和尾，没有哪个连接器对象来裁定羟基可以碰上羧基。位点无序也无方向，reaction SMARTS 是唯一写下化学的地方。

它**不是 typifier**。它调用 typifier，并且拒绝那些它用不了的。

## 重复单元就是标了几个原子的普通分子

没有 `RepeatUnit` 类，没有 `Junction`，没有 `Port`。重复单元就是一个普通分子，只不过其中几个原子被命名了。你把可以反应的那几个标出来，别的原子一概不动。

一个环氧乙烷重复单元是一个真实的、封端的分子：乙二醇。它的两个羟基氧是位点；羟基氢是反应将要移除的封端。

```python
from molpy.core import fields
from molpy.parser import parse_smiles, smilesir_to_atomistic

eo = smilesir_to_atomistic(parse_smiles("OCCO"))   # ethylene glycol
eo.atoms[0][fields.SITE] = "a"                     # one hydroxyl oxygen
eo.atoms[3][fields.SITE] = "b"                     # the other
```

这里没有任何地方说"头"或"尾"，`a` 和 `b` 也不携带方向——它们只是反应可以引用的标签。线性链是一个键恰好连成一条路径的世界。四臂交联剂是一个标了四个原子的分子。大环不过是在两个已经连通的原子之间再加一根键。上面这个 `eo` 原封不动地服务于全部三种情形。

## 电荷在模板上被冻结，于是守恒是白送的

AM1-BCC 电荷来自对整个分子的自洽求解。"片段的电荷"这种东西根本不存在，所以也就没有任何诚实的办法，靠观察新连接处周围的原子来算出它的电荷。这不是 MolPy 的局限；这就是"非局域"的含义。

出路是在组装期间根本不计算电荷。在封端好的重复单元上求解一次——那时分子小、闭壳层，`sqm` 才有意义。然后把每个封端的电荷折叠到它所封的那个位点原子上。折叠之后，每个封端携带的电荷恰好是零。

再看反应此时做了什么：它删掉的是电荷为零的原子。**净电荷守恒，是因为被移除的东西根本不带电**——而不是因为事后有个修正项把损失重新分摊了。守恒不再是一条启发式规则，它变成了一个记账恒等式。

MolPy 不会替你掩饰一个忘了冻结的模板。如果反应即将删掉一个带电原子，`assemble` 会抛错，并且直说：

```python
builder.build("{[#EO]|3}")
# ValueError: assembly changed the net charge by -0.6 e: the reaction deleted atoms
# that carry charge. Freeze the monomer templates first so each cap's charge folds
# onto its site atom.
```

AMBER 的 prep 文件按残基冻结电荷，也是同一个道理。同样的物理，各自独立地得出。

## 你可以猜一个数值，绝不能猜一个身份

组装需要给它即将形成的那根键一个长度，而此刻还没有任何力场就这根键被咨询过。于是它猜：两个原子共价半径之和。

这个猜测是正当的，理由很窄。键长是一个**连续**量，确实没有先验可查，而且下游的几何优化会把它拉到正确的值上。如果那次优化没能收敛，你得到的是一个错误，而不是一个悄无声息地绷着劲的结构。

拿原子的元素对比一下。如果 MolPy 不知道它，就假定是碳，下游没有任何一步会察觉。键长会被弛豫掉；身份不会。所以缺失元素会抛错，缺失原子类型会抛错，而未知的键长得到的是一个具名常量，外加一句注释，注明由哪个优化器负责把它收敛掉。

**猜数值，绝不猜身份。** 判据就是：后面是否有某一步会把这个猜测收敛掉。

## 相同的连接处只被类型化一次

千聚体链上每一个 EO–EO 连接处的局部化学都一模一样。逐个类型化，就是对同一个片段重复一千遍完全相同的过程。

所以 MolPy 切出的每个邻域都按其结构做哈希，它给出的类型缓存在这个哈希下。第二个连接处命中缓存。第八百个也是，下一条链的第一个连接处同样如此。类型化的遍数跟踪的是*不同*化学环境的数目，而不是你形成的键的数目。

缓存活在 assembler 上，不在单次调用上。把同一个 assembler 复用到一百条链上，整个熔体里 EO–EO 连接处只被类型化恰好一次。

## 长出一条链

`PolymerBuilder` 持有一个单体库，并且讲 CGSmiles。递给它一个记号串，它就为每种重复单元各盖出一份副本，把相邻的那些键接起来，然后把聚合物交还给你。每份粘贴进去的副本都拿到一个残基 id 和残基名——重复单元*就是*残基，而这个身份一路留存到 PDB 或 prmtop 里。

AmberTools 是个黑盒；antechamber 不会告诉 MolPy 它看多远，所以你告诉它一次，以键为单位。GAFF 原子类型由一到两根键的环境决定，因此 `reach=2`。MolPy 能读懂的 typifier，比如 `OPLSAATypifier`，会从自己的模式里推出这个数，也就不接受这样的参数。

```python
import molpy as mp
from molpy.builder import MonomerLibrary, PolymerBuilder, ResiduePlacer
from molpy.builder.ambertools import AmberTools
from molpy.typifier import AmberToolsTypifier

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
gaff = AmberToolsTypifier(AmberTools(), reach=2)
gaff.typify(eo)

builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, typifier=gaff)
chain = builder.build("{[#EO]|1000}")
# -> Atomistic, 1000 EO residues, junction types already correct
```

这条反应读作：一个带氢的 `a` 位点氧，加上一个带氢、且连在某个碳上的 `b` 位点氧，二者成为一座醚桥。左边出现、右边没有再出现的原子就是离去基团。builder 里没有任何东西知道"脱水"这个词；是 SMARTS 说出了它，而 `%a` 和 `%b` 谓词把它绑定到你标记的那些原子上。

## 交联用的是同一台机器

`PolymerBuilder` *就是*一个 assembler——它不过是给 assembler 添了一个库和一套记号。把这两样剥掉，剩下的就是 assembler 本身，而交联需要的正是这个：一张你已经有了的图，加上一条决定哪些位点配对的规则。

```python
from molpy.builder import GraphAssembler, RandomSelector

gel = GraphAssembler(ether, typifier=gaff).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)
```

`RandomSelector` 把彼此相距 6 Å 以内的位点对打乱，然后逐对消耗，直到 80 % 的位点已经反应。它只提供一条配对规则，别的什么都不管——图编辑、重新类型化和缓存，跟建链时是同一份代码。

这里没有传 `placer`，因为熔体的坐标已经有意义了，绝不能被扰动。`PolymerBuilder` 默认会摆放，因为新鲜的模板副本会互相重叠着落下。这是关于*你的输入*的判断，而不是关于你伸手取了哪个类的判断，所以它才是一个参数。

## 一个端到端的网络

建链、装填、交联，然后弛豫——交联键正是那些长度靠猜的键。

```python
import molpy as mp
from molpy.optimize import LBFGS, ForceFieldPotential

builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, typifier=gaff)
melt = mp.pack.Packmol().pack([builder.build("{[#EO]|50}")] * 100, density=0.9)

gel = GraphAssembler(ether, typifier=gaff).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)

frame = gel.to_frame()
LBFGS(ForceFieldPotential(gaff.forcefield)).run(frame, fmax=0.05, steps=200)
mp.io.write_lammps_system("gel", frame, gaff.forcefield)
```

一个 `builder` 建出全部一百条链，所以整个熔体里 EO–EO 连接处只被类型化一次——哪怕这些链长短不一也一样，因为缓存的键是局部结构，不是拓扑。交联处的连接则是每种不同环境类型化一次。这两个数目都不随链数增长，而这是一百条五十聚体的类型化量还算得动的唯一原因。

`LBFGS` 那一行，就是猜出来的键长消失的地方。它不是可有可无的润色：交联键是按共价半径之和的间距形成的，而在这一行之前，没有任何一步问过力场这个距离应该是多少。

## assembler 什么时候会拒绝

三种拒绝，都很响亮，而且都发生在任何一根键形成之前。

递给它一个 typifier，而它的力场追问环归属却不给尺寸——写 `[R]` 而不是 `[r6]`——构造就会失败，并且指名那条冒犯的模式。这样的力场没有有界邻域，所以没有诚实的办法用它去重新类型化一个连接处。MolPy 不会悄悄退化成改为重新类型化整个体系；正是那种回退，曾经让长链慢得不成样子。

递给它两个共享同一个原子的反应位点，`assemble` 会抛错，而不是把一次编辑叠加到另一次编辑已经作废的句柄上。

递给它一个封端仍然带电的重复单元——你忘了 `freeze`——它会抛错，而不是悄悄把净电荷泄漏进你的体系。

如果 `select` 找不到任何可反应的东西，这不算错误。截断可能太紧，或者目标转化率已经达到。你会得到一条警告，指明候选数和截断值，而你的世界原封不动地回到你手上。

## 写你自己的配对规则

你永远不去继承 assembler。你写一个 `Selector`，它只回答本页开头的第三个问题，别的什么都不管。它收到这个世界，以及反应匹配到的位点（按反应物分组），然后产出它想要成键的那些对。

```python
from molpy.builder import Selector
from molpy.core.atomistic import Atomistic

class NearestNeighborSelector(Selector):
    def select(self, world: Atomistic, occurrences: list[list[dict[int, int]]]):
        a_sites, b_sites = occurrences
        for occ_a in a_sites:
            occ_b = self._nearest(world, occ_a, b_sites)
            yield {**occ_a, **occ_b}       # {map_number: atom handle}
```

匹配已经发生过了——assembler 只做一次，线性时间——所以 selector 从不扫描体系。它只做决定。邻域提取、重新类型化、缓存、电荷检查以及互不重叠的保证，全都白送，因为它们没有一个依赖于你如何挑选这些对。

这就是本页三件活其实是同一套算法的含义：你可能想改的那部分，恰好是你唯一能改的部分。

## 另请参见

- **力场类型化** —— 哪些 typifier 可以用在组装过程中，以及如何为黑盒 typifier 声明 `reach`。
- **几何优化** —— 把猜出来的键长收敛掉的那一步。
- **构建交联凝胶** —— 上面这套流程，配上装填与平衡。
- `molpy.Reaction` —— reaction SMARTS 的语义、离去基团，以及 `%label` 谓词。
