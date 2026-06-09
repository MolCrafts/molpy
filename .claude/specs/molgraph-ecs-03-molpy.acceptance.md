---
slug: molgraph-ecs-03-molpy
spec: molgraph-ecs-03-molpy.md
created: 2026-06-08
verdict:
  ac-001: verified   # last_checked 2026-06-09 — IS-A leaves, interning identity, zero-copy column
  ac-002: verified   # last_checked 2026-06-09 — Struct/3 Mixins/_ordered_*/_MOLRS_KIND deleted; _GraphViews sole base; primitives unshadowed; remove-middle stable
  ac-003: verified   # last_checked 2026-06-09 — adopt zero-copy (from_molrs_graph removed); convention keys; no "X"; missing→None/raise
  ac-004: verified   # last_checked 2026-06-09 — pytest tests/ -m "not external" 1940 passed; ruff+ty clean; class S(Atomistic) ok
note: |
  base-order deviation — spec prescribes Atomistic(_GraphViews, molrs.Atomistic),
  but a pyo3 `extends` class must be the FIRST base, so the impl uses
  Atomistic(molrs.Atomistic, _GraphViews). MRO still resolves the leaf's own
  methods first and issubclass/IS-A holds; spec intent preserved.
---

# Acceptance — molgraph-ecs-03-molpy

molpy 收成 ECS world 句柄视图的验收（chain molgraph-ecs 3/3，依赖 molrs 01+02）。全 `type: code`。

## Criteria

```yaml
- id: ac-001
  type: code
  summary: 容器即 molrs world（IS-A，无桥）；Entity/Link = 按句柄 intern 的列视图，身份保持
  evaluator_hint: "pytest tests/test_core -m 'not external'"
  pass_when: |
    molpy.Atomistic 继承 molrs.Atomistic、CoarseGrain 继承 molrs.CoarseGrain（issubclass 成立，无
    __new__ 垫片）。s=Atomistic(); a=s.def_atom(element="C"); b=s.def_atom(element="O");
    bd=s.def_bond(a,b)：bd.itom is a、a in s.atoms、list(s.atoms)[0] is a、hash(a) 跨重复迭代稳定
    （WeakValueDictionary intern；持引用期间身份成立）。属性经句柄走 component 列；s.column(fields.X)
    为 numpy view（热路径零拷贝）。

- id: ac-002
  type: code
  summary: 扁平化——删 Struct+3 Mixin + 镜像 + _MOLRS_KIND；删中间原子句柄稳定无重排
  evaluator_hint: "pytest tests/test_core -m 'not external' && grep"
  pass_when: |
    类层级 = Atomistic(_GraphViews, molrs.Atomistic) / CoarseGrain(_GraphViews, molrs.CoarseGrain)；
    grep 确认 Struct/SpatialMixin/MembershipMixin/ConnectivityMixin、_ordered_nodes/_ordered_links、
    _index 平移循环、_MOLRS_KIND 均删除，唯一共享基为 _GraphViews。方法身份测试
    Atomistic.<molrs 原语> is molrs.Atomistic.<原语>（未遮蔽）。s 删中间原子后其它原子/键仍有效、计数正确。

- id: ac-003
  type: code
  summary: adopt 零拷贝取代 from_molrs_graph；构造走约定键、无 "X"、缺/类型错报错
  evaluator_hint: "pytest tests/test_core tests/test_parser -m 'not external'"
  pass_when: |
    g 为 molrs 产出图（parse_smiles("CCO").to_atomistic() 或 Conformer 结果）；m=Atomistic.adopt(g)
    零拷贝接管（源空、属性完整、不逐节点拷贝；from_molrs_graph 已删）。def_atom 经 world.spawn +
    world.set(h, fields.ELEMENT, …) 建；无 element 的节点 get(h, fields.ELEMENT) 为 null（无 "X"）；
    缺字段/类型错抛异常（无兜底）。字段经 core/fields 约定访问，无散落字面量。

- id: ac-004
  type: code
  summary: 构造模板 / blast radius / 全量套件 + 质量门不回归
  evaluator_hint: "pytest tests/ -m 'not external' && ruff check src tests && ty check src/molpy/"
  pass_when: |
    .entities/.links 故事落地（句柄迭代兼容视图，或 ≥7 模块同 PR 改写）；带 __post_init__ 的构造模板
    Water()/CH3() 仍实例化并建图；class S(Atomistic): pass; S() 可实例化。
    `pytest tests/ -m "not external"` 全绿（test_core/test_io/test_parser/embed）：CAT 单体 Generate3D
    n=32、带电 [N+]/[N-] 氢数正确。ruff format --check / ruff check / ty check 全过。
```
