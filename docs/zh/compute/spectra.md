# 分子动力学振动光谱：IR、Raman、VDOS、VCD 与 ROA

分子动力学**振动光谱**预测走的是"时间相关函数"路线——这也是从头算分子动力学参考实现率先采用的路径。核心思路很简单：对某个涨落量做自相关函数（ACF），再傅里叶变换到频域，乘上对应的前置因子就得到光谱。速度 ACF 给出振动态密度，偶极导数 ACF 给出红外光谱，极化率 ACF 给出拉曼光谱，互相关则给出手性光谱（VCD、ROA）。

谱变换在 Rust 中运行（`molrs`）。与结构类算子不同，这些变换**不**直接接收帧，而是接收*预先计算好的 ACF*和采样间隔，返回光谱。

!!! note "本文使用的约定"
    - 采样间隔 `dt_fs` 的单位是**飞秒**；输出频率的单位是波数（cm⁻¹）。
    - 速度、偶极通量、极化率等原始 ACF 由对应的时间序列构建。MolPy 的
      [`compute_acf`/`ACFAnalyzer`](transport.md) 负责从某一列组装 ACF。

---

## 1. 每种振动光谱都是 ACF 的傅里叶变换

根据线性响应理论（维纳-辛钦定理），光谱密度是动力学变量 $A(t)$ 的时间相关函数的傅里叶变换：

$$
I(\omega) \;\propto\; Q(\omega)\int_{-\infty}^{\infty}\!\big\langle A(0)\,A(t)\big\rangle\,e^{-i\omega t}\,\mathrm{d}t,
$$

其中 $Q(\omega)$ 是方法相关的校正因子（谐波量子校正；拉曼多一个频率/温度因子）。选择不同的 $A$ 就得到不同的光谱：

| 光谱 | 动力学变量 $A$ | 算子 |
|---|---|---|
| VDOS（功率谱） | 原子速度 | `PowerSpectrum` |
| 红外 | 总偶极导数（通量） | `IRSpectrum` |
| 拉曼 | 极化率（各向同性 + 各向异性） | `RamanSpectrum` |
| VCD | 电偶极 ⊗ 磁偶极 | `VcdSpectrum` |
| ROA | ROA 不变量（各向同性 + 各向异性） | `RoaSpectrum` |
| 共振拉曼 | 共振极化率 | `ResonanceRamanSpectrum` |

---

## 2. 由速度计算振动态密度

振动态密度（VDOS）是速度 ACF 的**功率谱**，也是最简单的一种光谱。它不依赖任何电子结构信息，只需要速度数据就能定位每个振动模式——无论该模式是否有红外活性。完整理论（笼效应、Green–Kubo 扩散）见[速度自相关与 VDOS](vacf.md)；下面 `dt_fs` 和 `cache_size` 的参数选择参见 [vacf.md §6 — 超参数影响](vacf.md#6)。

```python
from molpy.compute import PowerSpectrum, compute_acf

# velocities: (n_frames, n_atoms, 3); cache_size = 最大滞后帧数
vacf = compute_acf(velocities, cache_size=4096)  # 原始速度自相关
vdos = PowerSpectrum()(vacf, dt_fs=0.5)          # -> {频率 (cm^-1), 强度}
```

---

## 3. 红外光谱与拉曼光谱

**红外**光谱是体系总偶极导数（$\dot{\mathbf M}$，偶极"通量"）自相关函数的傅里叶变换。红外强度需要分子偶极矩，从头算分子动力学的偶极矩来自电子密度的[Voronoi 积分](voronoi.md)。**拉曼**光谱使用分子极化率，分为各向同性和各向异性两部分：

```python
from molpy.compute import IRSpectrum, RamanSpectrum

ir = IRSpectrum()(dipole_flux_acf, dt_fs=0.5)

raman = RamanSpectrum(incident_frequency_cm1=20000.0, temperature_k=300.0)(
    acf_iso, acf_aniso, dt_fs=0.5
)
```

`incident_frequency_cm1` 和 `temperature_k` 控制拉曼散射前置因子；设为零则返回裸光谱密度。

---

## 4. 手性光谱：VCD、ROA 与共振拉曼

手性光谱来自*互*相关。**振动圆二色性**（VCD）是电偶极 ⊗ 磁偶极互 ACF 的傅里叶变换；**拉曼光学活性**（ROA）和**共振拉曼**则用到相关响应张量的各向同性和各向异性不变量：

```python
from molpy.compute import VcdSpectrum, RoaSpectrum, ResonanceRamanSpectrum

vcd = VcdSpectrum()(electric_magnetic_acf, dt_fs=0.5)
roa = RoaSpectrum(averaged=True)(acf_iso, acf_aniso, dt_fs=0.5)
rr  = ResonanceRamanSpectrum(incident_frequency_cm1=20000.0)(acf_iso, acf_aniso, dt_fs=0.5)
```

这些函数复现了参考实现的体相手性光谱预测——分子动力学领域首次实现。

---

## 5. 常见陷阱

1. **采样间隔过粗** → 奈奎斯特极限 $\tilde\nu_\text{max} = 1/(2c\,\Delta t)$ 必须覆盖最高频率的模式；C–H 伸缩振动（~3000 cm⁻¹）需要亚飞秒量级的 `dt_fs`。
2. **ACF 太短** → 光谱分辨率由 ACF 长度决定；ACF 过短会导致峰展宽、无法分辨。对 ACF 加窗可抑制截断振荡。
3. **缺少量子校正** → 经典强度需要谐波量子校正因子才能与实验定量比较。
4. **用错了光谱变量** → 红外需要偶极*通量*（导数），而不是偶极本身；VDOS 需要速度，而不是位置。
5. **分子偶极矩/极化率不收敛** → 垃圾进垃圾出：验证 [Voronoi](voronoi.md) 电荷收敛之前，不要轻信红外强度。

---

## 6. 参考文献

- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976) — 时间相关函数与光谱密度。
- M. Thomas, M. Brehm, R. Fligg, P. Vöhringer, B. Kirchner, *Phys. Chem. Chem. Phys.* **15**, 6608 (2013) — 基于 TCF 的 AIMD 红外与拉曼光谱。
- M. Brehm, M. Thomas, *J. Phys. Chem. Lett.* **8**, 3409 (2017) — 基于 MD 的 VCD、ROA 与共振拉曼。
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105 (2020) — 参考实现。

## 另请参阅

- [Radical Voronoi](voronoi.md) — 提供红外强度所需的分子偶极矩。
- [Van Hove 与取向动力学](van-hove.md) — 线形背后的动力学关联。
- [介电谱](dielectric.md) — 通过偶极涨落计算介电响应。
- [计算模块 API 参考](../../api/compute.md)。
