# 介电谱

本文采用教材式推导风格，从头说明 MolPy 如何从平衡分子动力学（MD）轨迹计算**频率相关的介电常数** $\varepsilon^*(\omega)$ 和**离子电导率** $\sigma$。代码中每个预因子都有来源说明，每条公式都有推导或物理动机阐释，整个流程用一个贯穿始终的示例串联。读者无需线性响应理论前置知识，具备本科电磁学、少量统计力学和傅里叶变换基础即可。

所有谱分析物理量——自相关、加窗、FFT、预因子——都在 `molrs` 内部的 Rust 中运行。MolPy 层（`molpy.compute.DielectricSusceptibility`）只负责提取位置和电荷、解卷绕坐标、组装偶极序列，然后委托给 `molrs.dielectric`。

!!! note "全文使用的约定"
    - 复介电常数：$\varepsilon^*(\omega) = \varepsilon'(\omega) - i\,\varepsilon''(\omega)$（**正损耗**约定，因此 $\varepsilon'' \ge 0$）。
    - 傅里叶变换：$X(\omega) = \int_0^\infty f(t)\,e^{-i\omega t}\,dt$。
    - 单位（LAMMPS *real* 单位制）：长度 Å，电荷 $e$，时间 ps，体积 Å³，温度 K，角频率 rad·ps⁻¹。GROMACS 轨迹以原生 **nm** 读取，传入核函数前需将长度乘以 10。

---

## 运行示例

| 量 | 值 |
|---|---|
| 溶剂 | 852 个 SPC 水分子（每个 O、H、H 点电荷） |
| 离子 | 16 个 Na⁺ + 16 个 Cl⁻（32 个载流子，≈ 1 mol/L） |
| 原子数 | 852×3 + 32 = 2588 |
| 系综 | NVT（正则系综） |
| 温度 | 298.15 K |
| 盒子 | 立方，$L \approx 2.996$ nm，$V \approx 2.69\times10^4$ Å³ |
| 长度 / 步长 | 20 ns，1 fs 时间步长 |
| 输出 | 每 10 fs 输出位置**和**速度 |
| 电荷 | 水 O = −0.82 e，H = +0.41 e；Na = +1 e，Cl = −1 e |

该系统的两个特性决定了以下方法选择：

1. **同时存在取向极化（水转动）和平动电流（离子扩散）。** 两者必须**分开**处理——把全系统偶极直接代入静态介电常数公式会得到毫无意义的 $\varepsilon \approx 7000$（参见 [§5](#5)）。
2. **轨迹存有速度**，因此偶极自相关和电流自相关两条路线都可用，可以相互验证。

---

## 1. 物理图像：介电谱测量什么？

### 1.1 极化和介电常数

将介电体置于电场 $\mathbf{E}$ 中，其电荷会重新排列——分子偶极重新取向，电子云发生偏移——产生**极化** $\mathbf{P}$（单位体积的偶极矩）。线性区域内，

$$
\mathbf{P} = \varepsilon_0\,(\varepsilon - 1)\,\mathbf{E}.
$$

相对介电常数 $\varepsilon$ 越大，介质越容易被极化，对电场的屏蔽越强。真实水的 $\varepsilon \approx 78$；本例 SPC 模型给出 $\varepsilon \approx 54$——**这是 SPC 力场的已知特征，而非错误。**

### 1.2 为什么是*谱*——频率依赖

振荡场 $\mathbf{E}(t) = \mathbf{E}_0\cos(\omega t)$ 下，极化能否跟上取决于频率：

- **低 $\omega$**：分子有时间重新取向 → 完全极化 → $\varepsilon$ 较大。
- **高 $\omega$**：分子来不及响应 → 仅剩快速电子极化 → $\varepsilon \to \varepsilon_\infty$。
- **中间 $\omega$**：极化滞后于场，能量以热的形式耗散 → 出现**吸收峰**。

因此介电常数成为**复数、频率相关**的量：

$$
\boxed{\;\varepsilon^*(\omega) = \varepsilon'(\omega) - i\,\varepsilon''(\omega)\;}
$$

- $\varepsilon'(\omega)$（实部）——能量储存；低 $\omega$ 时高，衰减至 $\varepsilon_\infty$。
- $\varepsilon''(\omega)$（虚部）——**损耗 / 吸收**；对被动因果系统始终 $\ge 0$；在弛豫频率处出现峰值。（微波炉加热水正是利用其 $\varepsilon''$ 峰值附近的频率。）

### 1.3 涨落–耗散定理

这是概念上的关键。**涨落–耗散定理（FDT）**指出：

> 系统对外界微扰的*响应*完全由其*自发平衡涨落*的统计决定。

对于介电体，这意味着我们**不需要**施加振荡电场。观察系统**总偶极矩 $\mathbf{M}(t)$** 在平衡状态下**自身**的涨落就足够了：这些涨落的自相关编码了全部的介电响应。这是普通 NVT 轨迹足以完成计算的原因。

---

## 2. 基础积木：总偶极矩 $\mathbf{M}(t)$

一帧的瞬时总偶极矩是电荷加权的位置之和：

$$
\mathbf{M}(t) = \sum_{i=1}^{N_\text{atoms}} q_i\,\mathbf{r}_i(t),
\qquad [\mathbf{M}] = e\cdot\text{Å}.
$$

代码中，这对应简单的点积 `compute_dipole_moment`。

### 2.1 关键细节：最小镜像解卷绕

MD 在周期性盒子中运行。原子从一面移出盒子时，会从对面重新进入，坐标因此跳跃一个盒子长度 $L$。直接用这种跳跃坐标计算 $\mathbf{M}$，每次原子跨边界时 $\mathbf{M}$ 都会跳动 $q\cdot L$，使自相关失效。

**解决办法：累积最小镜像位移**

$$
\Delta\mathbf{r}_k = \mathbf{r}(t_k) - \mathbf{r}(t_{k-1}),
\qquad
\Delta\mathbf{r}_k \leftarrow \Delta\mathbf{r}_k - L\,\mathrm{round}\!\left(\frac{\Delta\mathbf{r}_k}{L}\right),
$$

$$
\mathbf{r}_\text{unwrap}(t_k) = \mathbf{r}_\text{unwrap}(t_{k-1}) + \Delta\mathbf{r}_k.
$$

`round` 项扣除整盒跳跃，保留真实位移（总是小于半盒长度）。MolPy 使用 `Box.delta(p1, p2, minimum_image=True)` 执行此操作。

!!! warning
    仅当每帧位移 $< L/2$ 时有效。每帧 10 fs 下，原子移动距离仅为 Å 的零头，远低于 $L/2 \approx 15$ Å——完全安全。这也是轨迹需要密集输出的原因之一。

---

## 3. 静态介电常数 $\varepsilon(0)$

零频率极限是计算中最简单的量，也确定了整个谱的低频基准。

### 3.1 Neumann 涨落公式

对于使用 Ewald / "tin-foil"（导体）边界条件的模拟，静态介电常数由**偶极矩涨落**给出（Neumann, 1983）：

$$
\boxed{\;\varepsilon(0) = \varepsilon_\infty + \frac{4\pi}{3}\,\frac{1}{V k_B T}\,
\big\langle\,|\mathbf{M} - \langle\mathbf{M}\rangle|^2\,\big\rangle\;}
$$

逐项解释：

- $\varepsilon_\infty$ —— 高频（电子）介电常数。对于 SPC 等非极化力场使用 **1**；对于可极化水模型使用 1.5–2.5。
- $\langle|\mathbf{M}-\langle\mathbf{M}\rangle|^2\rangle$ —— 总偶极的**方差**，单位为 $(e\cdot\text{Å})^2$。涨落越大 → 介电常数越大。这正是 FDT：大的涨落意味着强的响应。
- $V k_B T$ —— 体积 × 热能尺度；归一化到单位体积密度。
- $4\pi/3$ —— 导体边界条件下各向同性介质的几何因子。

### 3.2 无量纲化（实际单位）

LAMMPS *real* 单位制中，库仑常数不为 1，代码使用 $\kappa = 1/(4\pi\varepsilon_0) = 332.0637\ \text{kcal·Å·mol}^{-1}e^{-2}$：

$$
A_\text{stat} = \frac{4\pi}{3}\cdot\frac{\kappa}{V k_B T},
\qquad k_B = 1.987204\times10^{-3}\ \text{kcal·mol}^{-1}\text{K}^{-1}.
$$

有用恒等式（Green–Kubo 路线使用）：$1/\varepsilon_0 = 4\pi\kappa$。

### 3.3 数值注意事项：两趟方差

直接计算 $\langle M^2\rangle - \langle M\rangle^2$ 在 $|\mathbf{M}| \gg$ 其涨落时（$\sim10^6$ 帧场景下是实际风险）会遭受灾难性抵消。代码使用**两趟中心化方差**：先算均值 $\langle\mathbf{M}\rangle$，再算 $\tfrac1N\sum|\mathbf{M}-\langle\mathbf{M}\rangle|^2$。数学等价，数值稳定。

### 3.4 各向同性 vs 逐轴

- **各向同性** $\varepsilon(0)$：上述公式，预因子 $\propto 4\pi/3$。
- **逐轴** $\varepsilon_d$（$d=x,y,z$）：预因子**大 3 倍**（没有 $1/3$ 平均）：
  $\varepsilon_d = \varepsilon_\infty + \frac{4\pi\kappa}{V k_B T}(\langle M_d^2\rangle-\langle M_d\rangle^2)$。
  各向同性系统应有 $\varepsilon_x \approx \varepsilon_y \approx \varepsilon_z$，平均后回到标量值——这是内置的自检。

!!! example "运行示例"
    使用水偶极 $\mathbf{M}_D$（参见 [§5](#5)）得到 $\varepsilon(0) \approx 54$，与 GROMACS `gmx dipoles` 在溶剂组上得到的结果（54.06）吻合，偏差在 0.01–0.6 % 以内。

---

## 4. 谱路线 I——Einstein–Helfand（偶极自相关）

这是计算*整个* $\varepsilon^*(\omega)$ 曲线的主力方法。

### 4.1 核心线性响应关系

Caillol–Levesque–Weis（1986，其式 30）的久保关系：

$$
\varepsilon^*(\omega) - \varepsilon_\infty
= A\Big[\langle|\delta\mathbf{M}|^2\rangle - i\omega\,X(\omega)\Big],
\qquad A = \frac{4\pi\kappa}{3\,V k_B T},
$$

其中 $\delta\mathbf{M} = \mathbf{M} - \langle\mathbf{M}\rangle$，且

$$
C(t) = \langle\delta\mathbf{M}(0)\cdot\delta\mathbf{M}(t)\rangle\ \text{（偶极 ACF）},
\qquad X(\omega) = \int_0^\infty C(t)\,e^{-i\omega t}\,dt.
$$

简而言之：**介电谱是偶极涨落自相关函数的傅里叶变换。** $C(t)$ 衰减越慢（偶极"记忆"越长），低频响应就越强。

### 4.2 关键技巧：变换 ACF 的*导数*

直接构造 $i\omega X(\omega)$ 在数值上会发散：离散变换 $X(\omega)$ 携带一个与频率无关的本底 $\sim \Delta t\,C(0)$（来自 $t=0$ 的采样点），因此 $\omega X(\omega)$ 在奈奎斯特频率附近发散，导致损耗谱 $\varepsilon''$ 虚假上升（甚至变为负值）。

代码通过**分部积分**规避了这个问题。由于

$$
\int_0^\infty C'(t)\,e^{-i\omega t}\,dt = -C(0) + i\omega X(\omega)
\;\Rightarrow\;
i\omega X(\omega) = C(0) + \widehat{C'}(\omega),
$$

$A\,C(0) = A\langle|\delta\mathbf{M}|^2\rangle$ 项与显式的 $\langle|\delta\mathbf{M}|^2\rangle$ 相互抵消，得到**等价但稳定**的形式：

$$
\boxed{\;\varepsilon^*(\omega) - \varepsilon_\infty = -A\,\widehat{C'}(\omega),
\qquad \widehat{C'}(\omega) = \int_0^\infty C'(t)\,e^{-i\omega t}\,dt.\;}
$$

为什么稳定？$C'(t)$ 在**两端**均为零——$C'(0)=0$（ACF 是偶函数，其导数在原点为 0）且 $C'(\infty)=0$（相关性衰减）——因此其变换正确衰减，$\varepsilon''$ 保持有限。按实部和虚部分解（正损耗约定）：

$$
\varepsilon'(\omega) - \varepsilon_\infty = -A\,\mathrm{Re}\,\widehat{C'}(\omega),
\qquad
\varepsilon''(\omega) = A\,\mathrm{Im}\,\widehat{C'}(\omega).
$$

### 4.3 锚定 DC 频段

离散导数变换仅能以 $O(\Delta t)$ 的精度重现 $\omega=0$，代码因此使用**精确的 Neumann 值**覆盖 DC 频段：

$$
\varepsilon'(0) = \varepsilon_\infty + A\,\langle|\delta\mathbf{M}|^2\rangle,
\qquad \varepsilon''(0) = 0.
$$

这样谱的低频端点与 [§3](#3-varepsilon0) 完全一致（由单元测试强制执行）。

### 4.4 五步算法

这正是 `einstein_helfand_spectrum` 所做的：

1. **中心化方差** $\langle|\delta\mathbf{M}|^2\rangle$ 用于精确的 DC 频段。
2. **自相关**：对减均值后的偶极在 $x,y,z$ 方向上求和，通过 FFT（Wiener–Khinchin，[§6.1](#61-fft-wienerkhinchin)）并使用无偏估计 $C(k)=r[k]/(N-k)$（[§6.2](#62-vs)）。
3. **加窗**：使用单侧余弦平方锥度 $w[k]=\cos^2(\tfrac{\pi k}{2L})$——在 $C(0)$ 处为 1，在 $C(L)$ 处为 0（[§6.3](#63)）。
4. **求导 + FFT**：中心差分 $C'(t)$，零填充至 $n_\text{pad}=\big(2(L{+}1)\big)$ 向上取整到 2 的幂，前向 FFT，乘以 $\Delta t$（[§6.4](#64-fft)）。
5. **组装** $\varepsilon'(\omega), \varepsilon''(\omega)$，用精确静态值覆盖 DC 频段。

偶极 ACF 严格是单侧的，因此该路线始终使用余弦平方锥度，无论 `window_type` 参数为何。

---

## 5. 电解质步骤：分解偶极

这是含离子体系的关键物理要点。将**全系统**偶极 $\mathbf{M}_\text{tot}$ 代入 Neumann 公式得到 $\varepsilon \approx 7000$（这里实测 ≈ 7257）。原因是：

> 离子是**自由载流子**。它们的持续扩散使 $\mathbf{M}_\text{tot}$ 进行无界随机游走，方差永不收敛。这不是*极化*，而是**直流电导率**，不应出现在静态介电常数公式中。

解决办法是按物理来源分解总偶极（`decompose_current`，或简单地通过原子切片）：

$$
\mathbf{M}_D(t) = \sum_{i\in\text{水}} q_i\mathbf{r}_i(t)\ \text{（转动）},
\qquad
\mathbf{M}_J(t) = \sum_{i\in\text{离子}} q_i\mathbf{r}_i(t)\ \text{（平动）}.
$$

- **$\mathbf{M}_D$**（水取向偶极）——**有界涨落** → 送入介电路线（[§3](#3-varepsilon0)、[§4](#4-ieinsteinhelfand)）计算 $\varepsilon(0)$ 和 $\varepsilon(\omega)$。
- **$\mathbf{M}_J$**（离子平动偶极）——**扩散性增长** → 送入 Einstein 关系（[§7](#7-einsteinhelfand)）计算电导率 $\sigma$。

$\mathbf{M}_D + \mathbf{M}_J$ 之和在浮点精度内等于系统电流，分解是无损的——它仅仅分离了两个物理上不同的过程（取向 vs 传导）。

---

## 6. 数值方法（信号处理部分）

### 6.1 通过 FFT 计算自相关（Wiener–Khinchin）

直接按定义 $r[k]=\sum_\tau x[\tau]x[\tau+k]$ 是 $O(N^2)$——对于 $10^6$ 帧不可行。**Wiener–Khinchin 定理**指出自相关是功率谱的逆变换：

$$
r = \mathrm{IFFT}\big(|\mathrm{FFT}(x)|^2\big),
$$

计算成本降至 $O(N\log N)$。信号**必须零填充至 $\ge 2N$**（代码使用 $(2N)$ 向上取整到 2 的幂）；否则 FFT 返回的是*循环*自相关，尾部会绕回并污染小延迟处的值。

### 6.2 有偏估计 vs 无偏估计

FFT 产生的是**线性的、未归一化的** ACF
$r[k]=\sum_{\tau=0}^{N-1-k}x[\tau]x[\tau+k]$。更大延迟处配对的样本更少（$N-k$ 个），因此**无偏系综估计**除以 $N-k$：

$$
C(k\,\Delta t) = \frac{r[k]}{N-k} \approx \langle x(0)\,x(k\Delta t)\rangle.
$$

代价：大延迟处的噪声更大——这就是 `max_correlation_time` 不宜过大的原因（参见 [§6.5](#65)）。

### 6.3 加窗：选什么窗，为什么

ACF 在 `max_lag` 处截断；硬截断会导致谱振铃（Gibbs 现象）。乘以一个使尾部平滑衰减的**窗口**可以抑制这种现象。但介电 ACF 是**严格单侧的**（$t \ge 0$），其 $t=0$ 处的值 $C(0)=\langle|\delta\mathbf{M}|^2\rangle$*就是*静态信号——不能被锥化掉。

- ❌ **Hann / Blackman**（对称窗口）在*两端*都归零，抹杀了 $C(0)$。仅对双侧数据或电流 ACF 有效。
- ✅ **单侧余弦平方锥度** $w[k]=\cos^2(\tfrac{\pi k}{2L})$——在 $k=0$ 处为 1（保留 $C(0)$），在 $k=L$ 处为 0（平滑截断）。EH 路线始终使用它。

### 6.4 离散 FFT 到连续变换

物理需要**连续**变换 $X(\omega)=\int_0^\infty C(t)e^{-i\omega t}dt$；FFT 给出的是**离散** DFT。矩形法则（Numerical Recipes §13.9）建立了桥梁：

$$
\int_0^T f(t)e^{-i\omega t}dt \approx \Delta t \cdot \mathrm{DFT}[f](\omega_k).
$$

因此代码将 FFT 输出乘以 $\Delta t$（而非 FFT 内部的 $1/n_\text{pad}$）。这保证了最终 $\varepsilon$ 的量纲正确。

### 6.5 频率网格、分辨率和奈奎斯特

令 $n_\text{pad} = \big(2(\text{max\_lag}+1)\big)$ 向上取整到 2 的幂：

$$
\text{频段数} = \frac{n_\text{pad}}{2}+1,
\qquad
\Delta\omega = \frac{2\pi}{n_\text{pad}\,\Delta t}\ \text{（分辨率）},
\qquad
\omega_\text{max} = \frac{\pi}{\Delta t}\ \text{（奈奎斯特）}.
$$

权衡：

- **`max_lag` 越大** → $\Delta\omega$ 越精细（更好的低频分辨率），但尾部噪声越大。经验法则：`max_lag` $\le$ 帧数的四分之一。
- **$\Delta t$ 越小**（输出越密集）→ 奈奎斯特频率越高 → 可访问更高频的特征（例如数十 rad/ps 处的平动共振）。这就是示例每 **10 fs** 写入一帧的原因。对于静态 $\varepsilon$ 和慢弛豫，可以欠采样（例如每 200 帧 = 2 ps）以节省内存。

---

## 7. 离子电导率（Einstein–Helfand）

Green–Kubo 积分在粗采样下收敛性差。对于**离子电导率**，更稳健的等价路线是 **Einstein–Helfand** 关系——平动偶极 $\mathbf{M}_J$ 均方位移（MSD）的长时间斜率：

$$
\boxed{\;\sigma = \lim_{t\to\infty}\frac{1}{6\,V k_B T}\,
\frac{d}{dt}\big\langle|\mathbf{M}_J(t)-\mathbf{M}_J(0)|^2\big\rangle\;}
$$

这正是 `gmx current` 报告的 "Einstein–Helfand" 量。

### 7.1 步骤

1. **集体 MSD**：
   $\text{MSD}(k)=\langle|\mathbf{M}_J(t{+}k)-\mathbf{M}_J(t)|^2\rangle$，
   对所有时间原点 $t$ 取平均（`collective_msd`）。
2. **对斜率做线性拟合**：短时间 MSD 是*弹道*的，中间是*扩散*的，长时间是*噪声*的。仅在扩散窗口 $\lbrack\text{fit\_start\_frac},\text{fit\_end\_frac}\rbrack\cdot \text{max\_lag}$ 内拟合直线，通常取 $[0.1, 0.5]$。
3. **转换为 S/m**（SI 常数将 Å、ps 换算为 m、s）：

$$
\sigma\,[\text{S/m}] = \text{斜率}\,[(e\text{Å})^2/\text{ps}]\cdot
\frac{e^2\cdot 10^{-8}}{6\,V[\text{Å}^3]\cdot 10^{-30}\,k_B T},
$$

其中 $e=1.602\times10^{-19}$ C，$k_B=1.381\times10^{-23}$ J/K；$10^{-8}$ 因子包含了 Å²→m² 和 ps→s 的换算。

### 7.2 诚实警示：载流子少 → $\sigma$ 不确定

示例在 **20 ns 内仅有 32 个离子**。离子偶极 MSD 略显超扩散，$\sigma$ 对拟合窗口敏感：当窗口从 `[50,200]` 移动到 `[1000,3000]` ps 时，$\sigma$ 从 ≈ 5.8 漂移到 ≈ 10 S/m。**报告一个范围，而不是单个数字。** 默认的 `[100,400]` ps 窗口给出 $\sigma \approx 6.1$ S/m（与 `gmx current` 的 6.12 S/m 吻合，偏差 1.2 %）；更长的窗口接近 ≈ 8.5 S/m（1 M NaCl 的实验值）。更紧的收敛需要更多载流子和更长轨迹。

---

## 8. 谱路线 II——Green–Kubo（电流自相关）

第二条路线从**电流**出发，是导电体系的等价路径，也是路线 I 的自然交叉验证。

### 8.1 电流密度

电流密度是偶极对时间的导数除以体积，通过有限差分离散化：

$$
\mathbf{J}(t) = \frac{\dot{\mathbf{M}}}{V}
\approx \frac{\mathbf{M}(t)-\mathbf{M}(t-\Delta t)}{V\,\Delta t},
\qquad [\mathbf{J}] = e\cdot\text{Å}^{-2}\text{ps}^{-1}.
$$

`compute_current_density` 执行此操作。**第 0 行是 `NaN`**（无前一帧），所有使用者必须跳过它（Green–Kubo 核函数内部会自动处理）。

### 8.2 电导率谱 → 介电常数

$$
\sigma(\omega) = \frac{V}{3 k_B T}\int_0^\infty \langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle e^{-i\omega t}dt
= \sigma'(\omega) + i\sigma''(\omega).
$$

（预因子为 $V/(3k_BT)$，因为输入是电流密度 $\mathbf{J}=\dot{\mathbf{M}}/V$；对于总 $\dot{\mathbf{M}}$ 则为 $1/(3Vk_BT)$，相差 $V^2$，因为 $\langle\dot M\dot M\rangle=V^2\langle JJ\rangle$。）麦克斯韦关系将电导率与介电常数联系起来：

$$
\boxed{\;\varepsilon^*(\omega) - \varepsilon_\infty = -\frac{i\,\sigma(\omega)}{\omega\,\varepsilon_0}\;}
\;\Rightarrow\;
\varepsilon'(\omega)-\varepsilon_\infty = \frac{\sigma''(\omega)}{\omega\varepsilon_0},
\quad
\varepsilon''(\omega) = \frac{\sigma'(\omega)}{\omega\varepsilon_0},
$$

其中 $1/\varepsilon_0 = 4\pi\kappa$。DC 频段（$\omega=0$）是不定式 $\sigma/\omega = 0/0$，被正则化为 $(\varepsilon_\infty, 0)$；真正的静态值来自 [§3](#3-varepsilon0) 或低 $\omega$ 外推。Debye 极限下，该路线与路线 I 一致（由单元测试验证）。

---

## 9. 使用 MolPy 的端到端流程

高层 `DielectricSusceptibility` 计算将提取 → 解卷绕 → 偶极组装 → 两条路线 → 静态 $\varepsilon$ 打包为一次调用；物理计算仍在 `molrs` 中运行。

```python
from molpy.compute import DielectricSusceptibility

dc = DielectricSusceptibility(
    dt=0.01,                  # 保留帧之间的时间间隔（ps），这里为 10 fs
    temperature=298.15,       # K
    max_correlation_time=2000,# 帧数（设定分辨率；保持 <= n_frames/4）
    epsilon_inf=1.0,          # 非极化 SPC 水
    window_type="cosine_sq",
    routes=["einstein-helfand", "green-kubo"],
)
result = dc(trajectory)       # 一个解卷绕后的 molpy Trajectory

eh = result.results["EH-full"]
# eh.frequency      -> rad/ps
# eh.epsilon_real   -> epsilon'(omega)
# eh.epsilon_imag   -> epsilon''(omega)
# eh.epsilon_static -> epsilon(0)（Neumann），附加到每条路线

# Debye 弛豫时间，使用 NumPy 拟合（无需 SciPy）——参见第 10.1 节
fit = eh.fit_debye()        # fit.tau (ps), fit.delta_eps, fit.omega_peak
```

**离子电导率**有自己的计算类 `IonicConductivity`，封装了 Einstein-Helfand 核函数（[§7](#7-einsteinhelfand)）。传入仅含离子的轨迹（通过*选择*进行分解，[§5](#5)）：

```python
from molpy.compute import IonicConductivity

sigma = IonicConductivity(
    dt=0.01, temperature=298.15, max_correlation_time=1000,
    fit_start_frac=0.1, fit_end_frac=0.5,
)(ion_trajectory)
# sigma.sigma (S/m), sigma.slope, sigma.msd, sigma.time (滞后时间 ps)
```

每帧的 `atoms` 块必须包含 `x, y, z`（Å）和 `charge`（e），以及一个非自由的 `Box`。对于电解质，构建两个轨迹（或两个偶极序列）：一个仅含水原子，另一个仅含离子原子（如 [§5](#5)），对水偶极运行介电路线，对离子偶极运行电导率。

如需完全手动控制，底层的核函数可以直接调用：

| 步骤 | `molrs.dielectric` 函数 |
|------|---------------------------|
| 总/子系统的偶极 $\mathbf{M}=\sum q_i\mathbf{r}_i$ | `compute_dipole_moment` |
| 电流密度 $\mathbf{J}=\dot{\mathbf{M}}/V$ | `compute_current_density` |
| 拆分水/离子电流 | `decompose_current` |
| 静态 $\varepsilon(0)$ | `static_dielectric_constant` |
| 谱（偶极路线） | `einstein_helfand_spectrum` |
| 谱（电流路线） | `green_kubo_spectrum` |
| 离子电导率 $\sigma$ | `einstein_helfand_conductivity` |

---

## 10. 谱的拟合

曲线拟合与应用场景密切相关，有意**不**包含在计算核函数中——它作为 `scipy` 配方存在于你的分析脚本中。常用的物理模型如下。

### 10.1 Debye 弛豫（单弛豫时间）

最简单的极性液体模型：单指数 ACF $C(t)=C(0)e^{-t/\tau}$，给出

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \frac{\varepsilon(0)-\varepsilon_\infty}{1+i\omega\tau},
$$

$$
\varepsilon'(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{1+(\omega\tau)^2},
\qquad
\varepsilon''(\omega) = \frac{\Delta\varepsilon\,\omega\tau}{1+(\omega\tau)^2},
\qquad
\Delta\varepsilon = \varepsilon(0)-\varepsilon_\infty.
$$

- $\varepsilon'$ 从 $\varepsilon(0)$ 阶梯下降到 $\varepsilon_\infty$，在 $\omega\tau=1$ 处拐折。
- $\varepsilon''$ 在 $\omega_\text{peak}=1/\tau$ 处有一个峰值（在对数轴上对称），峰高 $\Delta\varepsilon/2$。
- **最快估计**：读取损耗峰位置 → $\tau = 1/\omega_\text{peak}$。
- **在 MolPy 中**：`DielectricResult.fit_debye()` 仅使用 NumPy 返回 $\tau$、$\Delta\varepsilon$ 和 $\omega_\text{peak}$——$\tau$ 是精确恒等式 $\varepsilon''/(\varepsilon'-\varepsilon_\infty)=\omega\tau$ 在上升支上的最小二乘斜率（比单个频段更稳健），并附有损耗峰回退方案。`DebyeFit.epsilon(omega)` 评估拟合模型。仅展宽/歪斜的拟合（如下所述）才需要 SciPy。

!!! example "运行示例"
    水（$\mathbf{M}_D$）的损耗峰给出 $\tau \approx 6.5$ ps（使用干净的 10 fs 数据；粗糙的 2 ps 采样使其低估至 ≈ 4.7 ps），与已知的 SPC 弛豫时间一致。

### 10.2 非 Debye：Cole–Cole / Cole–Davidson / Havriliak–Negami

真实液体具有弛豫时间的*分布*，从而展宽或歪斜了峰值。通用的 **Havriliak–Negami（HN）** 模型：

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{\big[1+(i\omega\tau)^\alpha\big]^\beta},
\qquad 0<\alpha\le1,\ 0<\beta\le1.
$$

- $\alpha<1,\beta=1$ → **Cole–Cole**（对称展宽）。
- $\alpha=1,\beta<1$ → **Cole–Davidson**（高频歪斜）。
- $\alpha=\beta=1$ → Debye。

联合拟合 $\varepsilon'$ 和 $\varepsilon''$ 以得到参数 $(\Delta\varepsilon,\tau,\alpha,\beta,\varepsilon_\infty)$，使用 `scipy.optimize.curve_fit`。

### 10.3 多个过程

水存在多个过程（主弛豫、快弛豫和高频平动共振）。叠加 Debye/HN 项：

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \sum_j \frac{\Delta\varepsilon_j}{[1+(i\omega\tau_j)^{\alpha_j}]^{\beta_j}}
+ \text{（平动共振）}.
$$

**平动（受阻转动）共振**出现在数十 rad·ps⁻¹ 处，是真正的*共振吸收*（阻尼谐振子），而非单调弛豫；使用洛伦兹 / DHO 线型进行拟合。解析它需要密集的 10 fs 采样（高奈奎斯特频率）。

### 10.4 电导率贡献（电解质）

即使经过分解，残留的直流电导率也会提升低频 $\varepsilon''$：

$$
\varepsilon''_\text{cond}(\omega) = \frac{\sigma_\text{DC}}{\omega\,\varepsilon_0}.
$$

这是一个 **$1/\omega$ 发散**（对数–对数图上斜率为 −1）。要么将其作为拟合项加入，要么先将其减去。通过观察 $\varepsilon''$ 低频端是否存在 −1 斜率来判断。

### 10.5 电导率"拟合" = MSD 斜率

`IonicConductivity` 计算端到端完成此工作：它在扩散窗口 $[0.1,0.5]\cdot\text{max\_lag}$ 内拟合 $\mathbf{M}_J$ 的 MSD，并应用第 7.1 节的预因子，返回以 S/m 为单位的 $\sigma$。务必验证（a）窗口位于线性扩散区域内，以及（b）窗口敏感性（第 7.2 节）。

---

## 11. 结果解读

| 量 | 来源 | 物理含义 | 示例值 |
|---|---|---|---|
| $\varepsilon(0)$ | $\mathbf{M}_D$ 涨落，Neumann | 静态介电常数 / 屏蔽能力 | ≈ 54（SPC；偏低属正常） |
| $\varepsilon'(\omega)$ | EH 谱，实部 | 能量储存；$\varepsilon(0)$→$\varepsilon_\infty$ | 54 → 1 |
| $\varepsilon''(\omega)$ | EH 谱，虚部 | 损耗 / 吸收；弛豫峰 | 峰值在 $\omega\approx1/\tau$ 处 |
| $\tau$ | 损耗峰 $1/\omega_\text{peak}$ 或 HN 拟合 | 偶极弛豫时间 | ≈ 6.5 ps |
| 平动峰 | 高频谱（密集采样） | 受阻转动共振 | 数十 rad/ps |
| $\sigma$ | $\mathbf{M}_J$ MSD 斜率，Einstein–Helfand | 直流离子电导率 | ≈ 6 S/m（范围 6–8.5） |

**交叉验证。** 路线 I（偶极）和路线 II（电流）在 Debye 极限下必须一致；静态 $\varepsilon(0)$ 必须同时被 Neumann 公式和 EH 谱的 DC 频段重现（由测试强制执行）。不一致几乎总是簿记问题——解卷绕、单位（nm vs Å，298.15 vs 300 K）、体积或偶极分组。历史上每次差异都能追溯到上述某项，匹配设定后的结果一致到 ≈ 1 %。

---

## 12. 陷阱检查清单

1. **未做解卷绕** → 偶极跳跃 $q\cdot L$；谱完全无效。
2. **将全系统 $\mathbf{M}_\text{tot}$ 放入静态公式** → $\varepsilon$ 发散到数千。电解质*必须*分解（[§5](#5)）。
3. **在介电 ACF 上使用 Hann/Blackman** → 抹杀 $C(0)$。使用余弦平方（[§6.3](#63)）。
4. **忘记 $\Delta t$ 因子或 nm→Å 换算** → 量级偏差高达数量级。
5. **`max_lag` 过大** → 噪声主导谱尾部。保持在帧数的 1/4 以内。
6. **电流第 0 行为 NaN** → 必须跳过（GK 核函数已处理）。
7. **将 $\sigma$ 报告为一个精确数字** → 载流子少时对窗口敏感；报告一个范围（[§7.2](#72-sigma)）。
8. **$T$ 或 $V$ 错误**（例如 GROMACS 内部使用 300 K，$V=26.952$ nm³） → $\varepsilon$ 和 $\sigma$ 出现系统性偏差。

---

## 13. 参考文献

- M. Neumann, *Mol. Phys.* **50**, 841 (1983) — 偶极涨落公式与边界条件（静态 $\varepsilon$）。
- J.-M. Caillol, D. Levesque, J.-J. Weis, *J. Chem. Phys.* **85**, 6645 (1986) — 久保关系；EH 式 (30) 与 GK 式 (36)–(39)。
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 式 7.7.20 — $\sigma(\omega)\leftrightarrow\varepsilon(\omega)$。
- N. Wiener (1930) / A. Khinchin (1934) — 自相关 ↔ 功率谱。
- W. H. Press et al., *Numerical Recipes* §13.2, §13.9 — FFT 自相关与离散→连续变换。
- S. Havriliak, S. Negami, *Polymer* **8**, 161 (1967) — 非 Debye 弛豫模型。

## 参见

- [Compute 概述](index.md) — Compute → Result 模式及其他分析方法。
- [API 参考：Compute](../../api/compute.md) — 计算类的自动文档。
- [概念：Box 与周期性](../tutorials/03_box_and_periodicity.md) — 最小镜像约定。
- [概念：Trajectory](../tutorials/05_trajectory.md) — 帧序列。
