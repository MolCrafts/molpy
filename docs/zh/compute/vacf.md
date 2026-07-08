# 速度自相关与振动态密度

**速度自相关函数（VACF）** 是粒子运动的记忆函数。由它可以导出两样东西：扩散系数的 **Green–Kubo 途径**和**振动态密度（VDOS）**。本文是这一主题的教科书式入门。[扩散与离子输运](transport.md)从位移角度描述同一物理过程，是 VACF 的时域互补视角；[分子动力学振动谱](spectra.md)则把速度→谱的配方推广到偶极子和极化率，是 VACF 思路的自然延伸。读者具备本科统计力学基础即可。

ACF 累积由流式 NumPy 内核实现（[`compute_acf`](../../api/compute.md)，一个改编自 *tame* 库的滚动缓存直接相关器）；谱变换（`PowerSpectrum` 及整个 [spectra.md](spectra.md) 家族）在 Rust（`molrs`）中运行，集体分析器背后的 FFT 相关器也是如此。

!!! note "全文使用的约定"
    - 原子 $i$ 在时间 $t$ 的速度：$\mathbf{v}_i(t)$，分量 $(v_x, v_y, v_z)$。
    - $\langle\cdots\rangle_{i,t}$ 表示对**粒子** $i$ 和**时间原点** $t$ 的平均（平衡动力学是平稳的——每一帧都是一个原点）。
    - 帧间距 $\Delta t$（**捕获**间隔，不一定是 MD 步长）。谱以飞秒为单位（`dt_fs`）；扩散常数继承速度所携带的任何长度²/时间单位（LAMMPS *real* 单位：Å/fs 给出 Å²·fs⁻¹，Å/ps 给出 Å²·ps⁻¹）。
    - $d = 3$ 空间维度；Green–Kubo 因子为 $1/d = 1/3$。

---

## 1. VACF 测量什么

VACF 是最简单的动力学关联函数：

$$
C_{vv}(\tau) = \big\langle\, \mathbf{v}_i(t)\cdot\mathbf{v}_i(t+\tau)\,\big\rangle_{i,t},
\qquad
\hat{C}_{vv}(\tau) = \frac{C_{vv}(\tau)}{C_{vv}(0)} .
$$

它所度量的物理量是：*经过时间 $\tau$，粒子还记得多少自己当初的速度？* 零延迟由能量均分定理确定：

$$
C_{vv}(0) = \big\langle |\mathbf{v}|^2 \big\rangle = \frac{3 k_B T}{m},
$$

因此 $C_{vv}(0)$ 充当了一个天然的温度标尺——若它无法复现恒温器的温度，说明单位或速度输出已经出错了，不必继续往下走。

衰减的**形状**标定物相：

- **稀薄气体**——速度经不相关的二元碰撞失去记忆：无特征结构，接近指数衰减至零。
- **液体**——初始（弹道）衰减之后，粒子在周围粒子构成的"笼子"中反弹：$C_{vv}$ 穿越**负值**（背散射），而后恢复至零。这个负向凹陷是**笼效应**最原始的表现形式。
- **固体**——粒子在近似简谐的势阱中振动：$C_{vv}$ 以晶格/分子振动频率**振荡**，仅靠非谐性衰减，其累积积分（→ §2）趋近于零——无扩散。

---

## 2. 扩散系数的 Green–Kubo 途径

扩散是速度记忆的累积。将位移写成速度的积分，展开 MSD（见 [transport.md §1](transport.md#1-einstein)），便得到 **Green–Kubo 关系**[^green][^kubo]：

$$
\boxed{\;D = \frac{1}{3}\int_0^\infty \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, dt\;}
$$

Einstein（MSD 斜率）与 Green–Kubo（VACF 积分）在**数学上等价**——前者微分 MSD，后者积分 MSD 的二阶导数。实际使用中二者互补：

- **Einstein（MSD）** 在粗时间分辨率下更稳健——帧间距较大时首选（见 [transport.md](transport.md)）。
- **Green–Kubo（VACF）** 需要较密的速度采样（液体的 ACF 在 ~0.1–1 ps 内衰减完毕），但能揭示 $D$ 为什么是这个值：笼凹陷吞噬扩散率，长时间尾部则为其供能。

实用的估计量是**累积积分** $D(\tau) = \frac{1}{3}\int_0^\tau C_{vv}(t)\,dt$，它必须达到一个平台值；引用这个平台值，而不是最大滞后处的值。

---

## 3. VDOS：速度中的振动

VACF 的傅里叶变换是**振动态密度**（功率谱）[^dickey]：

$$
\boxed{\;g(\omega) \;\propto\; \int_{-\infty}^{\infty} \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, e^{-i\omega t}\, dt\;}
$$

原子实际执行的每一个振动模式都会在这里出现——不论红外活性还是拉曼活性，因为速度不带任何选择定则。这使得 VDOS 成为系统动力学的**参考图谱**：谐波模式在其频率处出现尖峰，扩散表现为 $\omega \to 0$ 处的强度（$g(0) \propto D$），非谐性则表现为峰的展宽。[spectra.md](spectra.md) 中的红外、拉曼、VCD 和 ROA 光谱采用相同的 ACF→谱构造，只是把速度换成了偶极子/极化率通量（它们**确实**携带选择定则）。

---

## 4. 在 MolPy 中计算 VACF

每个粒子的 ACF 内核是 [`compute_acf`](../../api/compute.md)：它接受一个 `(n_frames, n_particles, 3)` 速度数组，对每个粒子将其速度与自身的滞后速度做点积，再对粒子和时间原点取平均——正是 $\langle\mathbf{v}_i(0)\cdot\mathbf{v}_i(\tau)\rangle_{i,t}$，**未归一化**（索引 0 处是 $\langle v^2\rangle$）。

```python
import numpy as np
from molpy.compute import PowerSpectrum, compute_acf

# velocities: (n_frames, n_atoms, 3)，每 dt 采样一次
vacf = compute_acf(velocities, cache_size=4096)   # 原始 <v(0)·v(t)>，每个滞后一个值

D = np.trapezoid(vacf, dx=dt) / 3.0               # Green–Kubo D（先检查平台值！）
D_running = np.cumsum(vacf) * dt / 3.0            # 累积积分 D(tau)

vdos = PowerSpectrum()(vacf, dt_fs=dt_fs)         # -> {frequency (cm^-1), intensity}
```

`PowerSpectrum` 在 `molrs` 中应用单边 FFT 和谱前因子；[spectra.md](spectra.md) 家族中的所有谱计算都使用同一个对象。

---

## 5. 参数及其含义

| 参数 | 位置 | 含义 |
|---|---|---|
| `data` (速度) | `compute_acf` | `(n_frames, n_particles, 3)` 数组；单位决定了 $C_{vv}$ 和 $D$ 的单位 |
| `cache_size` | `compute_acf` | **以帧为单位的**曲线长度：返回的数组覆盖滞后 $0 \dots \text{cache\_size}-1$（最大滞后 = `cache_size − 1`） |
| `dropnan` | `compute_acf` | 对不规则/部分数据的 NaN 处理策略（默认为 `"partial"`） |
| `dt` / `dt_fs` | 你的记账 / `PowerSpectrum` | 帧间距；将滞后转换为时间并设定频率轴 |

---

## 6. 超参数影响

- **帧间距 $\Delta t$（采样率）。** 频谱的截止频率为奈奎斯特频率 $\tilde{\nu}_\text{max} = 1/(2c\,\Delta t) \approx 16678/(\Delta t/\text{fs})$ cm⁻¹。$\Delta t = 0.5$ fs 时可分辨到 ~33 000 cm⁻¹，覆盖全部分子振动；$\Delta t = 10$ fs 时，~1700 cm⁻¹ 以上的信号全部**混叠**——C–H 和 O–H 伸缩峰会折叠回虚假的低频区。$\Delta t$ 应根据关心的最硬模式来选，而非磁盘预算。
- **最大滞后（`cache_size`）。** 它同时设定了 Green–Kubo 的积分窗口和谱分辨率 $\Delta\tilde{\nu} \approx 33356/(\tau_\text{max}/\text{fs})$ cm⁻¹。窗口太短：$D(\tau)$ 达不到平台，VDOS 峰被人为展宽。窗口太长：尾部全是噪声——ACF 估计的统计误差随 $\sqrt{\tau/T_\text{traj}}$ 增长，因为独立原点数减少了——积分噪声会让 $D$ 值漂移。一个好的默认值是 $\hat{C}_{vv}$ 可见衰减时间的 5–10 倍。
- **轨迹长度 $T_\text{traj}$。** 对时间原点取平均，意味着固定滞后处的相对误差按 $1/\sqrt{T_\text{traj}}$ 标度；延长运行时间比加大滞后窗口更有效。
- **恒温器耦合。** 强随机恒温器（强 Langevin / 激进的速度重标度）会直接把摩擦和噪声注入 $\mathbf{v}(t)$，**重塑** VACF：抑制笼凹陷、移动 VDOS 峰。平衡后应在 **NVE**（或耦合极弱的恒温器）下采样速度。
- **平均速度扣除 / 漂移。** 质心漂移会给 $C_{vv}$ 增加一个永不衰减的常数偏移，使 Green–Kubo 积分线性发散。输出速度前务必去除 COM 运动。
- **计算开销。** `compute_acf` 是一个**直接**流式相关器：每来一个新帧，就与滚动缓存中保留的 `cache_size` 帧做点积。时间按 $O(n_\text{frames} \times \text{cache\_size} \times n_\text{atoms})$ 标度，内存为 $O(\text{cache\_size} \times n_\text{atoms})$，与轨迹总长无关。保持 `cache_size` 适中——它既是一个**时间**旋钮，也是一个物理旋钮。$O(N\log N)$ FFT 相关器（`molrs.signal.acf_fft`）供集体分析器（`ACFAnalyzer`、[spectra.md](spectra.md)）使用；它需要将整个序列一次性加载到内存。

---

## 7. 阅读结果

| 检查项 | 期望值 | 违反时的诊断 |
|---|---|---|
| $C_{vv}(0)$ | 每个粒子 $3k_BT/m$（能量均分） | 单位错误、质量错误，或速度并非你所认为的那样 |
| 负向凹陷（液体） | 在 ~0.1–0.5 ps 处出现浅极小值 | 液体中没有 → 采样太粗 |
| $\hat{C}_{vv}(\tau\to\infty)$ | 衰减至 0 | 常量偏移 → COM 漂移 |
| $D(\tau)$ 累积积分 | 在中间 $\tau$ 处达到平台 | 无平台 → 滞后窗口太短或存在漂移 |
| $D$（Green–Kubo）与 $D$（MSD 斜率） | 在统计误差范围内一致 | 不一致 → MSD 侧的拟合窗口或解卷问题 |
| $\omega=0$ 处的 VDOS | $\propto D$；对固体为零 | 虚假零频尖峰 → 漂移 |

---

## 8. 易错点检查清单

1. **速度采样过粗** → 谱混叠，VACF 完全错过笼凹陷。VACF 的衰减比 MSD 变线性快约 10 倍。
2. **引用最后一个滞后处的 $D$ 而非平台值** → 积分尾部噪声大；务必绘制 $D(\tau)$。
3. **恒温器污染** → Langevin 摩擦明显阻尼 $\hat{C}_{vv}$；生产级 VACF/VDOS 运行使用 NVE。
4. **未去除 COM 漂移** → ACF 存在永不衰减的偏移，Green–Kubo 积分发散，VDOS 出现 $\omega = 0$ 尖峰。
5. **在需要单粒子 ACF 的地方用了集体 ACF**——应当先对原子取平均**再**做相关。面向集体信号（如总偶极子）的分析器（如 `ACFAnalyzer`）度量的是 $\langle\bar{\mathbf{v}}(0)\cdot\bar{\mathbf{v}}(t)\rangle$，这是 COM 的记忆，不是 VACF。VACF/VDOS 应使用 `compute_acf`（每个粒子独立做点积）。
6. **$D$ 的单位错误**——Å/fs 速度给出的 $D$ 单位为 Å²·fs⁻¹（1 Å²·fs⁻¹ = 0.1 cm²·s⁻¹），Å/ps 给出 Å²·ps⁻¹（= 10⁻⁴ cm²·s⁻¹）；在每个数值旁注明所用单位。
7. **`cache_size` 超过可用原点数**——接近 `n_frames` 的滞后几乎没有统计平均；将滞后限制在轨迹的一小部分内。

---

## 9. 参考文献

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017), §2.7, §8.5 — 时间关联函数与输运系数。
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002), §4.4 — Green–Kubo 与 Einstein 估计量。
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21 — 谱密度作为 TCF 的傅里叶变换。
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., ch. 7 — 速度自相关函数与笼效应。

[^green]: M. S. Green, *J. Chem. Phys.* **22**, 398 (1954) — 来自时间关联函数的输运系数。
[^kubo]: R. Kubo, *J. Phys. Soc. Jpn.* **12**, 570 (1957) — 所有 Green–Kubo 关系背后的涨落-耗散形式体系。
[^dickey]: J. M. Dickey, A. Paskin, *Phys. Rev.* **188**, 1407 (1969) — 来自 MD 中速度自相关函数的声子谱。

## 参见

- [扩散与离子输运](transport.md) — Einstein/MSD 途径求 $D$ 及集体（电流）Green–Kubo 电导率。
- [分子动力学振动谱](spectra.md) — 红外、拉曼、VCD、ROA：相同的 ACF → 谱机制，但使用携带选择定则的通量。
- [介电谱](dielectric.md) — 频域响应及谱估计量的深入探讨。
- [计算概述](index.md) — Compute → Result 模式。
- [API 参考：Compute](../../api/compute.md)。
