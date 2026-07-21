# 选择器 (Selector)

把"距离这个点 3 A 以内的所有碳原子"写成可组合的表达式，而不是手写循环。

## 无需循环，直接选原子

分析工作中经常要回答这类问题："找出所有碳原子"、"距离某个点 3 A 以内的原子"，或者"盒子左半边的重原子"。用 if 语句手写循环当然可以，但代码一长就很难维护，真正的意图也淹没在细节里。

**选择器是可组合的谓词，对 `Block` 的列生成布尔掩码。** 用 `&`、`|`、`~` 运算符把它们拼起来，不用手动写循环就能构建复杂的查询。

每个选择器都遵守同一套协议：直接调用它，传入一个 block，得到过滤后的 block；调用 `.mask()` 得到布尔数组。

## 按属性筛选

最简单的选择器按单列的值过滤。`ElementSelector` 匹配元素符号，`AtomTypeSelector` 匹配类型编号。

```python
import molpy as mp
from molpy.core.selector import (
    ElementSelector, AtomTypeSelector,
    CoordinateRangeSelector, DistanceSelector,
)
import numpy as np

atoms = mp.Block({
    "element": ["C", "C", "H", "H", "O", "N"],
    "type":    [1, 1, 2, 2, 3, 4],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
})

carbons = ElementSelector("C")(atoms)
print(carbons.nrows)         # 2
print(carbons["element"])    # ['C', 'C']

type2 = AtomTypeSelector(2)(atoms)
print(type2["element"])      # ['H', 'H']
```

## 几何选择器

`CoordinateRangeSelector` 按某一轴的坐标范围过滤。`DistanceSelector` 按到参考点的距离过滤。两者都要求 block 有 `x`、`y`、`z` 列。

```python
right_half = CoordinateRangeSelector("x", min_value=2.5)(atoms)
print(right_half["element"])   # ['H', 'O', 'N']

near_origin = DistanceSelector(center=[0.0, 0.0, 0.0], max_distance=1.5)(atoms)
print(near_origin["element"])  # ['C', 'C']
```

壳层选择——取距离在最小和最大之间的原子——是溶剂化分析中的常见操作。

```python
shell = DistanceSelector(
    center=[2.0, 0.0, 0.0],
    min_distance=1.0,
    max_distance=2.5,
)(atoms)
print(shell["element"])
```

## 用逻辑运算符组合选择器

选择器真正的灵活之处在于组合。`&` 是 AND（与），`|` 是 OR（或），`~` 是 NOT（非）。组合结果本身也是一个选择器，可以继续使用或再次组合。

```python
# (碳 或 氧) 且 (x > 0.5)
sel = (ElementSelector("C") | ElementSelector("O")) & CoordinateRangeSelector("x", min_value=0.5)
result = sel(atoms)
print(result["element"])   # ['C', 'O']

# 除氢以外的所有原子
no_h = ~ElementSelector("H")
print(no_h(atoms)["element"])   # ['C', 'C', 'O', 'N']
```

嵌套组合可以简洁地表达精确的科学查询。

```python
# 靠近特定点的重原子
heavy_near = (
    ~ElementSelector("H")
    & DistanceSelector(center=[2.0, 0.0, 0.0], max_distance=2.5)
)
print(heavy_near(atoms)["element"])
```

## 直接取掩码

有时候需要的是布尔掩码，而不是过滤后的 block——用来索引其他数组、做 NumPy 运算，或者跟外部逻辑对接。

```python
mask = ElementSelector("C").mask(atoms)
print(mask)                        # [ True  True False False False False]
print(np.where(mask)[0])           # [0, 1]
print(atoms["x"][mask])            # [0., 1.]
```

## 什么时候用选择器

只要需要在 `Block` 里划分原子——不管是为了分析、赋值，还是给计算准备子集——就优先用选择器。它们比手写循环快得多，代码也更容易读，而且可组合的特性让你可以用简单、经过测试的小部件拼出复杂的查询。

另请参阅：[Block 和 Frame](02_block_and_frame.md)、[盒子与周期性](03_box_and_periodicity.md)。
