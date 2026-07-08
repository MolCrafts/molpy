# 氢键网络

本文是一份氢键检测的入门指南，内容自成体系。氢键靠几何条件识别：供体-受体距离和供体-H···受体角度满足阈值，就认定成键。每帧生成的键列表可以用来统计配位数、映射网络，再配合[持久性](persistence.md)分析还能测量氢键寿命。典型应用包括水、醇类和质子型离子液体。

几何搜索在 Rust（`molrs`）中执行，MolPy 层只负责传入供体/受体选择并返回结果——和其他计算模块一个模式。

!!! note "本文使用的约定"
    - 距离以 Å 为单位，角度以度为单位。
    - **供体** 是一个 `(D, H)` 对（重原子及其键合的氢）；**受体** 是一个单独的重原子。
    - 默认判据采用 Luzar–Chandler 几何：供体-受体距离 ≤ 3.5 Å，且 D–H···A 角度 ≥ 150°。

---

## 1. 氢键是一个几何事件

没有哪个量子力学观测量会在形成氢键时发生跳变，所以模拟靠**几何判据**来判定：一帧之内，供体 `(D, H)` 和受体 `A` 满足以下条件，就认为形成了氢键

$$
r_{D\cdots A} \le r_c \qquad\text{and}\qquad \angle(D\text{–}H\cdots A) \ge \theta_c.
$$

距离可以量供体到受体，也可以量氢到受体，取决于判据设定。截断值不是普适常数——应从对应 $g(r)$ 的第一极小值和[距离-角度 CDF](distributions.md) 中读取：CDF 上的高密度区域*定义*了氢键的联合分布，那才是选参的依据。

---

## 2. 检测氢键

传入供体 `(D, H)` 对和受体索引；用可选的 `HBondCriterion` 调整几何参数：

```python
import numpy as np
from molpy.compute import HBonds, HBondCriterion

donors = np.array([[o1, h1], [o1, h2]], dtype=np.int64)   # (D, H) 对
acceptors = np.array([o2, o3, o4], dtype=np.int64)

hb = HBonds(donors, acceptors, HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0))
result = hb(frames)

result.counts      # 每帧的氢键数量
result.per_frame   # 每帧的 (D, H, A, distance, angle) 列表
```

每帧返回的元组可以拿来构建键网络（度分布、环统计），也可以把特定的供体-受体对喂给寿命分析。

---

## 3. 从键列表到寿命

单帧计数只能告诉你*有多少*氢键，说不清它们能*撑多久*。要获得寿命，把每个检测到的供体-受体对当作一次缔合事件，再对它做[对持久性](persistence.md)生存分析：**间歇**相关给出 Luzar–Chandler 的结构氢键寿命 $\tau_\text{HB}$，**连续**相关给出短得多的"首次断裂"时间。二者一并报告，连同所用几何判据，是表征氢键动力学的标准做法。

---

## 4. 常见陷阱清单

1. **判据敏感性**——计数和寿命严重依赖 $r_c$ 和 $\theta_c$，应从距离-角度 CDF 中选取并明确说明。
2. **供体列表必须把 D 和它的 H 配对**——每个供体条目是 `(heavy, hydrogen)`；含两个氢的重原子贡献两行供体。
3. **自对**——如果只关心分子间键，要排除分子内供体/受体组合。
4. **距离约定**——供体-受体和氢-受体截断值不能互换；选一种，在整个体系中保持一致。
5. **寿命 ≠ 计数**——瞬时计数高可以和寿命短共存；两者回答的是不同问题。

---

## 5. 参考文献

- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996) — 几何判据与氢键动力学。
- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — 连续与间歇键关联函数。
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — 参考实现。

## 参见

- [对持久性](persistence.md) — 将键列表转化为寿命。
- [分布函数](distributions.md) — 定义判据的距离-角度 CDF。
- [计算概述](index.md) — Compute → Result 模式。
- [API 参考：计算](../../api/compute.md)。
