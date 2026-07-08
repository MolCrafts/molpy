# Van Hove 相关函数与转向动力学

MolPy 提供了两种时间分辨相关函数：**Van Hove 相关函数** $G(r, t)$ 和 **Legendre 转向相关函数** $C_1(t), C_2(t)$。Van Hove 函数是径向分布函数在时间维度上的推广；转向函数则量化分子矢量方向随时间的衰减。两者共同连接了[结构指南](structure.md)中的静态结构与[输运指南](transport.md)中的输运系数——核心都是*结构随时间退相关*。

相关计算核心在 Rust（`molrs`）中运行；MolPy 层负责展开轨迹并返回带类型的结果。

!!! note "全文约定"
    - 距离单位为 Å；时间延迟以**帧**给出（乘以帧间隔以转换为皮秒）。
    - $\langle\cdots\rangle$ 表示对粒子和时间原点取平均。

---

## 1. Van Hove 函数是 g(r) 的时间分辨形式

Van Hove 函数统计经过延迟 $t$ 后相距 $r$ 的粒子数，可分为**自**部分和**异**部分：

$$
G(r,t) = \underbrace{\frac{1}{N}\Big\langle\sum_i \delta\big(r - |\mathbf r_i(t) - \mathbf r_i(0)|\big)\Big\rangle}_{G_s(r,t)\ \text{(同粒子)}}
       + \underbrace{\frac{1}{N}\Big\langle\sum_{i\ne j}\delta\big(r - |\mathbf r_i(t) - \mathbf r_j(0)|\big)\Big\rangle}_{G_d(r,t)\ \text{(不同粒子)}}.
$$

- **$G_s(r,t)$** 为自部分：单粒子位移的分布。短时间内在 $r=0$ 处有尖锐峰；长时间下展宽为高斯分布，对应 Fick 扩散，其二阶矩即为[均方位移](transport.md)。出现非高斯形状则标志着跳跃或笼效应。
- **$G_d(r,t)$** 为异部分：$t=0$ 时退化为 $\rho\,g(r)$；随时间推移，邻居扩散使粒子周围的壳层结构逐渐抹平。

---

## 2. 计算 Van Hove 函数

```python
from molpy.compute import VanHove

vh = VanHove(n_rbins=200, r_max=15.0, lags=[1, 5, 10, 50, 100])
result = vh(frames)

result.r_centers    # 径向网格，Å
result.lags         # 时间延迟（帧）
result.g_self       # G_s(r, t)：行对应延迟，列对应径向 bin
result.g_distinct   # G_d(r, t)（当 result.has_distinct 时存在）
```

`lags` 的选取应覆盖感兴趣的动力学范围：少量短延迟用来分辨弹道运动和笼效应区间，较长延迟用来达到扩散平台。

---

## 3. 转向：矢量多快失去初始方向

固定在分子上的单位向量 $\mathbf u(t)$（如化学键、偶极矩、对称轴），其方向随时间变化。**Legendre 转向相关函数**定义为

$$
C_\ell(t) = \big\langle P_\ell\big(\mathbf u(0)\cdot\mathbf u(t)\big)\big\rangle,
\qquad P_1(x)=x,\quad P_2(x)=\tfrac{1}{2}(3x^2-1).
$$

两者均从 1（完全相关）衰减到 0（完全随机化），衰减快慢由转向动力学决定。$\ell$ 的选择取决于实验手段：**不同实验探测不同的 $\ell$**。介电弛豫测量 $C_1$，核磁自旋弛豫、荧光退偏振和红外/拉曼线形则测量 $C_2$。拟合 $C_\ell(t)$ 长时段的指数尾部 $C_\ell(t)\approx e^{-t/\tau_\ell}$ 可得转向相关时间 $\tau_\ell$；旋转扩散极限下 $\tau_1/\tau_2 = 3$。

---

## 4. 计算转向相关函数

```python
import numpy as np
from molpy.compute import LegendreReorientation

pairs = np.array([[o, h1], [o, h2]], dtype=np.int64)   # O–H 键矢
reor = LegendreReorientation(max_lag=500)
result = reor(frames, pairs)

result.lags   # 延迟（帧）
result.c1     # C_1(t)
result.c2     # C_2(t)
```

---

## 5. 常见陷阱检查清单

1. **`r_max` 超过盒子一半** → 异部分会被周期性镜像污染；保持 `r_max ≤ L/2`。
2. **延迟跨度超过轨迹长度** → 可用的时间原点太少，长延迟尾部噪声过大。应保证每个延迟都有足够的时间原点做平均。
3. **从非指数头部读取 $\tau$** → 应在 $C_\ell$ 的长时间指数尾部拟合，而非亚皮秒的 librational 振荡衰减段。
4. **矢量未归一化** → 提供真实的键端点即可，计算核心会自动构造成单位矢量。注意退化配对（两个端点相同）未定义。
5. **$G_s$ 用包裹坐标计算** → 单粒子位移需要非包裹轨迹，否则自部分会饱和在盒子尺寸上。

---

## 6. 参考文献

- L. Van Hove, *Phys. Rev.* **95**, 249 (1954) — 相关函数 G(r, t)。
- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976) — 转向相关函数及 $C_1$/$C_2$ 的区分。
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105 (2020) — 参考实现。

## 参见

- [扩散与离子输运](transport.md) — MSD 是 $G_s$ 的二阶矩。
- [结构分析](structure.md) — $G_d(r, 0) = \rho\,g(r)$。
- [介电谱](dielectric.md) — $C_1$ 是介电响应的基础。
- [API 参考：计算模块](../../api/compute.md)。
