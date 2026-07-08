# 扩散与离子输运

本页是一份自包含的教科书式介绍，讲述 MolPy 如何将平衡分子动力学（MD）轨迹转化为**输运系数**——扩散系数、Onsager 唯象系数和离子电导率。从随机行走出发，逐步推导到现代电解质分析中的集体相关函数。前提是读者具备本科统计力学基础和少量线性代数知识。

本文是[介电谱](dielectric.md)的姊妹篇：该页详细推导了*频域*响应和两条电导率路径；本页则聚焦于*扩散*与*位移*图像，涉及谱学工具时会回指到介电页面。

与所有 MolPy 分析一样，数值计算在 Rust（`molrs`）中完成；MolPy 层负责提取坐标、解包裹周期性镜像、构建集体量，并返回带类型的结果对象。

!!! note "全文使用的约定"
    - 原子 $i$ 在时间 $t$ 的位置：$\mathbf{r}_i(t)$；滞后 $\tau$ 上的位移：$\Delta\mathbf{r}_i(\tau) = \mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)$。
    - $\langle\cdots\rangle_t$ 表示对**时间原点** $t$ 的平均。
    - 单位（LAMMPS *real* 单位制）：长度 Å，时间 ps，电荷 $e$，体积 Å³，温度 K。扩散系数单位为 Å²·ps⁻¹；电导率单位为 S·m⁻¹。
    - $d = 3$ 个空间维度；Einstein 因子 $1/(2d) = 1/6$。

---

## 1. 随机行走与 Einstein 关系

液体中的粒子不断受到周围邻居的碰撞。短时间尺度上，粒子做*弹道*运动（速度方向尚未被扰乱），但在大量不相关碰撞之后，运动退化为**随机行走**：*方向*信息丢失，*弥散程度*持续增长。衡量这种弥散最直接的指标是**均方位移（MSD）**：

$$
\mathrm{MSD}(\tau) = \big\langle\,|\mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)|^2\,\big\rangle_{i,t}.
$$

对于扩散过程，MSD 随时间**线性**增长，其斜率定义了**自扩散系数** $D$（Einstein, 1905）：

$$
\boxed{\;D = \lim_{\tau\to\infty}\frac{1}{2d\,\tau}\,\mathrm{MSD}(\tau)
       = \lim_{\tau\to\infty}\frac{1}{6\tau}\big\langle|\Delta\mathbf{r}(\tau)|^2\big\rangle\;}
$$

因子 $1/6 = 1/(2d)$（$d=3$），对应随机行走粒子在三个独立方向上的弥散。

### 1.1 三个区间

真实的 MSD 曲线包含三个区间，只有中间一段才是真正的扩散过程：

- **弹道区**（短 $\tau$）：$\mathrm{MSD}\propto\tau^2$——粒子速度方向尚未充分随机化。
- **扩散区**（中等 $\tau$）：$\mathrm{MSD}\propto\tau$——线性区间；**在此处拟合斜率**。
- **噪声区**（长 $\tau$）：剩余的时间原点很少，统计涨落主导了估计值。

选择线性窗口是扩散计算的关键步骤。输运计算工具返回**完整的相关曲线**，以便在拟合之前查看；`IonicConductivity` 提供基于比例的拟合窗口（`fit_start_frac`/`fit_end_frac`），目的正是如此（见[§6](#6)）。

### 1.2 时间原点平均

在平衡状态下，动力学是平稳的，每一帧都可以作为时间原点 $t$。对所有原点取平均（"加窗"MSD）比仅从第 0 帧测量位移有效率得多：

$$
\mathrm{MSD}(\tau) = \frac{1}{N_\text{origins}}\sum_{t} |\mathbf{r}(t+\tau)-\mathbf{r}(t)|^2,
\qquad N_\text{origins} = N_\text{frames}-\tau.
$$

随着 $\tau$ 增大，可用的原点数量减少——这正是长 $\tau$ 尾部噪声大的原因。

### 1.3 最小镜像解包裹（常见陷阱）

MD 模拟使用周期性盒子：原子从一个面离开时，会从对面重新进入，因此存储的坐标会跳变一个盒子长度 $L$。直接用这种坐标计算 MSD，每次跨边界都会引入一个虚假的 $L$ 量级位移。解决方案是累积**最小镜像**步长：

$$
\Delta\mathbf{r}_k = \mathbf{r}(t_k)-\mathbf{r}(t_{k-1}) - L\,\mathrm{round}\!\Big(\tfrac{\mathbf{r}(t_k)-\mathbf{r}(t_{k-1})}{L}\Big),
\qquad
\mathbf{r}_\text{unwrap}(t_k) = \mathbf{r}_\text{unwrap}(t_{k-1}) + \Delta\mathbf{r}_k,
$$

该公式在粒子每帧移动距离小于 $L/2$ 时有效。MolPy 使用 `Box.delta(p1, p2, minimum_image=True)` 实现此功能；所有基于位移的输运计算均共享该方法。（偶极子场景下的相同讨论见[介电谱 §2.1](dielectric.md#21)。）

---

## 2. 自扩散与相异扩散：MDC

单个扩散系数掩盖了大量物理信息。多组分系统中，*不同*粒子的位移之间存在关联——阳离子拖曳反离子，溶剂围绕它们流动。**平均位移相关（MDC）** 将 MSD 泛化，揭示这类关联。[^gudla]

**自（标签 `"3"`）**——单个物种的普通 MSD，给出自扩散系数 $D^\mathrm{s}_\alpha$：

$$
D^\mathrm{s}_{\alpha} = \lim_{\tau\to\infty}\frac{1}{6\tau N}
   \sum_i\big\langle|\Delta\mathbf{r}_{i,\alpha}(\tau)|^2\big\rangle.
$$

**相异（标签 `"3,4"`）**——物种 $\alpha$ 和 $\beta$ 的位移之间的*交叉*关联，给出相异扩散系数 $D^\mathrm{d}_{\alpha\beta}$：

$$
D^\mathrm{d}_{\alpha\beta} = \lim_{\tau\to\infty}\frac{1}{6\tau N}
   \sum_i\sum_{j\ne i}\big\langle\Delta\mathbf{r}_{i,\alpha}(\tau)\cdot\Delta\mathbf{r}_{j,\beta}(\tau)\big\rangle.
$$

相异项是一个**集体**量：它主要取决于物种如何*一起*运动，而非任何单个粒子。

!!! note "归一化约定（MolPy 与 tame 的区别）"
    对于不同物种，MolPy 将相异项计算为集体交叉相关 $\big\langle(\sum_i\Delta\mathbf{r}_i)\cdot(\sum_j\Delta\mathbf{r}_j)\big\rangle$ ——这种物理上有意义的形式直接输入到[§3](#3-onsager)的 Onsager 系数中。原始的 [tame](https://github.com/Roy-Kid/tame) `mdc` 配方对参考物种 $i$ 取平均（而非求和），多了一个因子 $1/N_i$；MolPy 有意使用未归一化的集体形式。如需完全归一化的 Onsager 系数，请使用 [`Onsager`](#3-onsager)。

```python
from molpy.compute import MCDCompute

mdc = MCDCompute(tags=["3", "4", "3,4"], max_dt=20.0, dt=0.01)
result = mdc(trajectory)
result.correlations["3"]    # 物种 3 的自 MSD，随滞后时间变化
result.correlations["3,4"]  # 物种 3 和 4 的相异交叉相关
```

---

## 3. Onsager 唯象系数

描述电解质中耦合输运，最自然的方法是使用 Onsager 的**唯象系数** $\Omega_{\alpha\beta}$（也记为 $L_{\alpha\beta}$）。它们是集体位移关联的归一化形式：$\Omega_{\alpha\beta}$ 将物种 $\alpha$ 的通量与物种 $\beta$ 的热力学驱动力联系起来。

定义物种的**集体坐标**——其所有原子的求和（解包裹后）位置：

$$
\mathbf{P}_\alpha(t) = \sum_{i\in\alpha}\mathbf{r}_i(t),
\qquad
\Delta\mathbf{P}_\alpha(\tau) = \mathbf{P}_\alpha(t+\tau)-\mathbf{P}_\alpha(t).
$$

集体位移相关和 Onsager 系数为：

$$
\mathrm{corr}_{\alpha\beta}(\tau) = \big\langle\,\Delta\mathbf{P}_\alpha(\tau)\cdot\Delta\mathbf{P}_\beta(\tau)\,\big\rangle_t,
\qquad
\boxed{\;\Omega_{\alpha\beta} = \lim_{\tau\to\infty}\frac{\mathrm{corr}_{\alpha\beta}(\tau)}{6\,k_B T\,V\,N_A\,\tau}\;}
$$

- **对角项** $\Omega_{\alpha\alpha}$ 是物种 $\alpha$ 的集体 MSD——包含自项**加上**同种离子交叉项。
- **非对角项** $\Omega_{\alpha\beta}$（$\alpha\ne\beta$）捕捉阳离子–阴离子耦合。负值（反相关漂移）表明形成了离子对。

`Onsager` 返回相关曲线 $\mathrm{corr}_{\alpha\beta}(\tau)$；取长时斜率并乘以 $1/(6 k_B T V N_A)$ 前置因子即得到系数本身。

```python
from molpy.compute import Onsager

ons = Onsager(tags=["1,1", "1,2", "2,2"], max_dt=20.0, dt=0.01)
result = ons(trajectory)
result.correlations["1,2"]  # 阳离子-阴离子集体相关 L_12(tau)
```

### 3.1 从 Onsager 系数到电导率

离子电导率是 Onsager 系数按离子电荷 $z_\alpha$ 的加权和：

$$
\sigma = \frac{e^2}{V k_B T}\sum_{\alpha\beta} z_\alpha z_\beta\,\Omega_{\alpha\beta}.
$$

如果非对角（相异）项为零——即离子独立运动——则上式退化为仅由自扩散系数构建的**能斯特–爱因斯坦**估计值 $\sigma_\text{NE}$。比值 $\sigma/\sigma_\text{NE}$（*离子性*或 *Haven 比*）衡量离子关联对传导的抑制（或增强）程度——这个数值无法仅从单粒子扩散得到，这正是 Onsager 框架存在的意义。

---

## 4. 离子电导率：两条等价路径

电导率可以直接从**集体电荷输运**获得，无需经过单独的 $\Omega_{\alpha\beta}$。有两条等价路径（这是涨落耗散定理的一般推论；见[介电谱 §1.3](dielectric.md#13)）。

### 4.1 Einstein 路径——极化 MSD（PMSD）

构建离子的**集体电荷位移**（也称为平动偶极矩）：

$$
\mathbf{P}(t) = \sum_\text{cations}\mathbf{r}_i(t) - \sum_\text{anions}\mathbf{r}_j(t),
$$

并测量其 MSD。其长时斜率通过 Einstein 关系给出电导率：

$$
\mathrm{PMSD}(\tau) = \big\langle|\mathbf{P}(t+\tau)-\mathbf{P}(t)|^2\big\rangle_t,
\qquad
\sigma = \lim_{\tau\to\infty}\frac{1}{6\,V k_B T}\,\frac{d}{d\tau}\,\mathrm{PMSD}(\tau).
$$

```python
from molpy.compute import PMSDCompute, IonicConductivity

# PMSD 曲线本身：
pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01)(trajectory)

# 电导率（Einstein-Helfand 路径），拟合后转换为 S/m：
sigma = IonicConductivity(dt=0.01, temperature=298.15, max_correlation_time=1000)(ion_trajectory)
sigma.sigma  # S/m
```

`PMSDCompute` 返回曲线；`IonicConductivity` 完成拟合和单位转换。完整的推导、扩散窗口注意事项和 SI 前置因子见[介电谱 §7](dielectric.md#7-einsteinhelfand)。

### 4.2 Green–Kubo 路径——电流自相关（JACF）

等价地，对**电荷电流** $\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a(t)$ 的自相关进行积分：

$$
\boxed{\;\sigma = \frac{1}{3\,V k_B T}\int_0^\infty \big\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\big\rangle\,dt\;}
$$

被积函数 $C(\tau)=\langle\mathbf{J}(0)\cdot\mathbf{J}(\tau)\rangle$ 是电流自相关函数（JACF）；因子 $1/3$ 对应 Green–Kubo 路径中的 $1/6$（前者对 ACF 积分，后者对 MSD 微分）。单粒子层面的类似公式——通过速度自相关计算 $D$ 的 Green–Kubo 路径——在[速度自相关与 VDOS](vacf.md) 中推导。

```python
from molpy.compute import JACF

jacf = JACF(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01, temperature=298.15)
result = jacf(trajectory)        # 需要每个原子的速度 vx, vy, vz
result.jacf                      # <J(0).J(t)>
result.sigma                     # 直流电导率, S/m（GK 积分）
result.sigma_running             # 运行积分 sigma(tau)，用于检查收敛性
```

Einstein（PMSD）和 Green–Kubo（JACF）路径在数学上等价；实践中，Einstein 路径在粗采样时更稳健，JACF 则直接暴露电流记忆，便于观察积分的收敛过程。[介电谱 §8](dielectric.md#8-iigreenkubo) 推导了频域推广 $\sigma(\omega)$。

---

## 5. 读取结果

| 量 | 计算工具 | 物理含义 |
|---|---|---|
| $\mathrm{MSD}(\tau)$ / $D^\mathrm{s}$ | `MCDCompute`（单个标签） | 单粒子扩散 |
| $D^\mathrm{d}_{\alpha\beta}$ | `MCDCompute`（配对标签） | 相异（集体）扩散 |
| $\mathrm{corr}_{\alpha\beta}(\tau)$ / $\Omega_{\alpha\beta}$ | `Onsager` | 耦合输运；离子对形成（非对角项） |
| $\mathrm{PMSD}(\tau)$ | `PMSDCompute` | 集体电荷输运 |
| $\sigma$（Einstein） | `IonicConductivity` | 直流电导率，S/m |
| $C(\tau)$，$\sigma$（Green–Kubo） | `JACF` | 电流记忆 + 直流电导率 |

**交叉验证。** Einstein 路径（`IonicConductivity`/`PMSDCompute`）和 Green–Kubo 路径（`JACF`）给出的电导率应在统计误差内一致。当离子对形成显著时，`MCDCompute` 自项给出的能斯特–爱因斯坦估计值应大于 `Onsager`/`JACF` 的关联电导率——比值即为离子性。

---

## 6. 参数与超参数

### 6.1 参数及其含义

| 参数 | 计算工具 | 含义 |
|---|---|---|
| `tags` | `MCDCompute`, `Onsager` | 物种选择器：`"3"` = 类型 3 的自项/同类集体项；`"3,4"` = 类型 3 和 4 的相异交叉相关 |
| `max_dt` | `MCDCompute`, `Onsager`, `PMSDCompute`, `JACF` | 最长相关滞后，**ps**；返回的曲线有 `n_cache = int(max_dt / dt)` 个点 |
| `dt` | 所有输运计算 | 轨迹帧间隔，**ps**（*采集*间隔，非 MD 时间步长） |
| `center_of_mass` | `MCDCompute`, `Onsager` | 可选的 `{type: mass}` 映射；提供时，每帧在构建位移之前扣除系统质心（默认 `None`） |
| `cation_type` / `anion_type` | `PMSDCompute`, `JACF` | 在集体坐标 $\mathbf{P}(t)$ 或电流 $\mathbf{J}(t)$ 中赋予电荷 $+1$ / $-1$ 的原子类型索引 |
| `temperature` | `JACF`, `IonicConductivity` | $1/(V k_B T)$ 前置因子中的温度 $T$，**K**——$\sigma$ 与 $1/T$ 成正比 |
| `volume` | `JACF`, `IonicConductivity` | 系统体积，**Å³**；`None` → 使用轨迹上的平均盒子体积（`JACF`）或第一帧盒子体积（`IonicConductivity`，假定 NVT/NVE） |
| `max_correlation_time` | `IonicConductivity` | 最长 MSD 滞后**以帧数计**（限制为 `n_frames − 1`）；实际选择 $\le$ `n_frames / 5` |
| `fit_start_frac`，`fit_end_frac` | `IonicConductivity` | 扩散区间上的线性拟合窗口，表示为最大滞后的比例（默认值 `0.1`，`0.5`） |

`MCDCompute`、`Onsager` 和 `PMSDCompute` 返回**原始相关曲线**——$1/(2d\,\tau)$ 斜率、$1/(6 k_B T V N_A)$ Onsager 前置因子和单位转换都需自行处理，这样拟合窗口始终由自己掌控。只有 `IonicConductivity` 和 `JACF` 在内部完成拟合并转换单位。

### 6.2 超参数影响

- **拟合窗口位置。** 拟合**过早**会包含弹道/笼藏头部，此时局部指数 $d\ln\mathrm{MSD}/d\ln\tau \ne 1$——提取的斜率存在系统性偏差（采用 $D$ 之前，应检查对数–对数斜率在整个窗口内是否接近 1）。拟合**过晚**则是以偏差换取方差：独立时间原点的数量随 $N_\text{origins} = N_\text{frames} - \tau/\Delta t$ 减少，统计误差大致按 $\sqrt{\tau/T_\text{traj}}$ 增长。调整 `fit_start_frac`/`fit_end_frac`（或手动选定的窗口）并报告散布范围——少数载流子的集体量，散布通常不低于百分之几。
- **`max_dt` 与轨迹长度。** 接近轨迹长度的滞后几乎没有可用于平均的原点；建议保持 `max_dt` 为运行时长的一小部分（`IonicConductivity` 使用的 `≤ n_frames / 5` 规则是本页所有曲线的一个良好默认值）。加倍轨迹长度比加倍 `max_dt` 更有价值。
- **帧间隔 `dt`。** Einstein/MSD 路径在较粗的 `dt` 下仍然稳健——位移是累积量。Green–Kubo `JACF` 则不然：电流 ACF 在 ~0.1–1 ps 内衰减，较粗的帧间隔无法解析被积函数，$\sigma$ 会漂移。这里适用与 VACF 相同的采样率权衡——见 [vacf.md §6](vacf.md#6)。
- **维度。** 本页的前置因子假设 $d = 3$（$1/(2d) = 1/6$ Einstein，$1/3$ Green–Kubo）。对于准二维系统（受限薄膜、膜），需自行对原始曲线应用 $d = 2$ 因子——用三维因子处理二维运动会使 $D$ 低估 $2/3$。
- **质心漂移（`center_of_mass`）。** 集体量（Onsager、PMSD、JACF）每个物种只有*一个*实现——净质心漂移会引入相干 $\propto \tau^2$ 污染，且无法通过平均消除。报告非对角系数之前，应当扣除质心漂移（传入质量映射，或预先清理轨迹）。
- **`temperature` / `volume`。** $\sigma \propto 1/(V\,T)$：任一参数有 3% 的误差，电导率就线性缩放。对于 NPT 数据，应使用*生产运行*的平均值，而非目标恒温器值。
- **Green–Kubo 积分截止（`JACF.max_dt`）。** 在 `result.sigma_running` 的平台处报告 $\sigma$，绝不要取最末滞后——积分 ACF 尾部噪声会使估计值线性漂移（与 [vacf.md §2](vacf.md#2-greenkubo) 中运行积分 $D(\tau)$ 的平台规则相同）。

[§7](#7) 中散布的警告正是这些调节旋钮的失效模式。

---

## 7. 陷阱检查清单

1. **未解包裹** → 跨边界引入 $L$ 量级跳跃，所有 MSD 都被污染。（MolPy 通过 `Box.delta` 自动解包裹。）
2. **在扩散窗口外拟合** → 弹道头部或噪声尾部导致 $D$ 和 $\sigma$ 出现偏差。务必先查看曲线。
3. **载流子过少 / 轨迹过短** → 集体量（PMSD、Onsager 非对角项、JACF）本质上噪声很大，因为每个物种只有*一个*集体坐标。报告其范围而非单个数值。
4. **`JACF` 中速度单位错误** → 电流必须使用 $e\cdot$Å·ps⁻¹（速度单位为 Å/ps）；$\sigma$ 线性缩放，单位错误会直接缩放结果。
5. **忽略相异扩散** → 仅引用能斯特–爱因斯坦电导率忽略了离子关联，通常会高估 $\sigma$。

---

## 8. 参考文献

- A. Einstein, *Ann. Phys.* **322**, 549 (1905) — 扩散/MSD 关系。
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017) — MSD、时间原点平均、输运系数。
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. — Green–Kubo 关系与电流相关函数。
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002), §4.4 — 输运系数的 Einstein 关系。
- L. Onsager, *Phys. Rev.* **37**, 405 (1931); **38**, 2265 (1931) — 倒易关系与唯象系数。

[^gudla]: H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) — 结合持续函数的相异扩散，用于提取配对对输运的贡献。

## 参见

- [介电谱](dielectric.md) — 谱学工具（$\varepsilon^*(\omega)$、自相关、FFT）以及完整的电导率推导。
- [配对持续](persistence.md) — 驻留时间和生存函数，解析扩散中的*配对*贡献。
- [计算概览](index.md) — Compute → Result 模式。
- [API 参考：Compute](../../api/compute.md)。
