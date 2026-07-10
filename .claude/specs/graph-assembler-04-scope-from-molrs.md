---
title: reach 由模式集导出 —— 语法事实下沉,体系判断留在 molpy
status: draft
created: 2026-07-10
depends_on: "graph-assembler-01-reach; molrs: pattern-syntax-01 (NOT YET IMPLEMENTED)"
blocked_on: "molrs pattern-syntax-01"
---

# 局域性是模式集的性质,不是用户旋钮

> chain graph-assembler 4/4。**BLOCKED on molrs。**
>
> 01–03 已经把 `reach` 变成一个正确的、由 ac-003 实测钉死的数,但它仍然是**人写在 docstring
> 里的常数**。本 spec 让它从模式集**导出**,并让无界模式集在 `SmartsTypifier` 构造时 `TypeError`。

## Domain basis:谁做这个判断(铁律 2)

前一版的本 spec 让 molrs 直接返回 `scope -> Option<(depth, ring_size)>` —— 即由 **Rust 决定**
"这个模式集有没有有限感受野"、"`reach` 该取多少"。那是把**力场语义的判断**下沉进引擎,换取
"编译期"这个名义。按 `.claude/notes/architecture.md` §设计铁律 2,这是错的。

正确的切分:

| | molrs(引擎) | molpy(体系) |
|---|---|---|
| 回答什么 | **语法/结构事实** | **力场语义判断** |
| 本 spec | 一条编译好的 SMARTS 的最大键深;它用到了哪些环谓词(带尺寸的与不带尺寸的);图的最大环系尺寸 | 这些事实合起来是否蕴含有限感受野;`reach` 取多少;无界就 `TypeError` |

molrs **不认识** `TypeScope`,不知道 `⌊N/2⌋ + 1` 这条公式,也不判定"无界"。它只报告它作为
SMARTS 编译器天然知道的东西。`TypeScope.from_patterns()` 在 molpy 里做合成 —— 那里才知道
"环成员性是全局性质"这件化学事实。

> **与 `opls-typifier-downsink` 的张力。** 那条 spec 要把整个 OPLS 分型器下沉进 molrs,
> 与铁律 2 直接冲突(分型是力场语义)。它当前 BLOCKED。**本 spec 不依赖 `molrs.OplsTypifier`**,
> 只依赖模式集的语法事实。`opls-typifier-downsink` 重启前须按铁律 2 重新裁决,不在此处预判。

## molrs 侧需求(slug `pattern-syntax-01`)

纯语法查询,无策略:

```
SmartsPattern.max_bond_depth  -> int
    该模式从任一查询原子出发的最大键路径长度。

SmartsPattern.ring_primitives -> list[RingPrimitive]
    该模式用到的环谓词清单。每项报告 kind 与(若有)size:
      Sized(n)      —  [r3] … [r8]
      Membership    —  [R]  / [!R]  / [R0]
      RingCount(n)  —  [R1] / [R2]
      RingBondCount —  [x2]
    只是"用了什么",不判断"因此有界与否"。

Atomistic.max_ring_system_size -> int
    最大**环系**(稠环整体)的原子数。萘 = 10,不是 6 —— 芳香性是环系性质。
```

`ring_primitives` 返回**枚举**而不是布尔 `is_bounded`:枚举是语法,布尔是判断。
molpy 拿到枚举后决定哪些 kind 使 `reach` 不存在。

## molpy 侧

```python
# molpy/typifier/scope.py
@dataclass(frozen=True)
class TypeScope:
    TERM_REACH: ClassVar[int] = 2
    reach: int

    @classmethod
    def from_patterns(cls, patterns: Iterable[molrs.SmartsPattern]) -> Self:
        """由模式集的语法事实合成感受野。无界 → UnboundedPatternSet。

        体系判断在这里,不在 molrs:环成员性 / 环数 / 环键数需要全图 SSSR,
        故它们使感受野不存在;带尺寸的 [rN] 是局域的。
        """
```

```python
class SmartsTypifier(LocalTypifier):
    def __init__(self, patterns, ...):
        try:
            self._scope = TypeScope.from_patterns(patterns)
        except UnboundedPatternSet as exc:
            raise TypeError(
                f"pattern set contains {exc.primitive}; ring membership is a global "
                f"property, so this typifier has no finite receptive field and cannot "
                f"be used for region typing. Replace [R] with a sized [r6]."
            ) from exc
```

`ForceFieldTypifier` / `MMFFTypifier` 的 `reach` 从 docstring 常数变成上式。
01 的 ac-003 扫描测试**保留** —— 它现在验证"导出的 reach == 实测最小 reach",
从一个声明变成一个**可证伪的预测**。

`AmberToolsTypifier` **不进入这条路径**:antechamber 是黑盒,没有可编译的模式集。
它的 `scope` 仍由构造器必填(01 已定),docstring 标注这是外部工具的边界。

`UnboundedPatternSet` 是一个携带 `primitive` 的异常类(有数据,不是命名空间壳)。

**铁律 5:无界模式集不降级。** 不存在"检测到 `[R]` 就退回全图分型"的分支,也不存在
"警告一下然后按 `reach=4` 试试"。无界 ⟹ 构造 `TypeError` ⟹ 这个分型器不能用于区域分型,
调用方必须换模式集或换分型器。同理,`max_bond_depth` 查询失败(molrs 版本不匹配)是
`AttributeError`,不是"那就用默认值"。

## Tasks

- [ ] T1 (molrs) `pattern-syntax-01`:`SmartsPattern.max_bond_depth` / `.ring_primitives`
      (枚举,不判断)/ `Atomistic.max_ring_system_size`
- [ ] T2 (molrs) 发版;molpy `pyproject.toml` 精确版本 pin
- [ ] T3 `TypeScope.from_patterns()` + `UnboundedPatternSet`;合成公式与无界判据全在 molpy
- [ ] T4 `SmartsTypifier(LocalTypifier)`:构造时合成 scope;无界 → `TypeError` 点名谓词
- [ ] T5 删 typifier docstring 里的 reach 常数;01 的 ac-003 扫描测试改为验证导出值

## Testing

- 含 `[R]` / `[!R]` / `[R2]`(环数)/ `[x2]`(环键数)的模式集 → `SmartsTypifier` 构造 `TypeError`,
  错误信息**点名该谓词**
- 把 `[R]` 换成 `[r6]` → 构造成功,`scope.reach` 有界
- 导出的 `reach` == 01 ac-003 扫描出的最小通过值(OPLS-AA 与 MMFF94 各一遍)
- 稠环体系(萘)的 `max_ring_system_size == 10`;返回 6 ⟹ molrs bug,在 molrs 修
- **铁律 2 的可执行检查**:`grep -rn 'TypeScope\|reach' <molrs 源码>` → 零命中
  (molrs 不认识感受野这个概念)

## Out of scope

- `AmberToolsTypifier` 的 `scope` 推导(黑盒,永远由构造器声明)
- `opls-typifier-downsink` 的去留裁决(与铁律 2 冲突,单独处理)
