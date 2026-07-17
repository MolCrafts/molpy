---
slug: molrs-core-cutover
created: 2026-07-14
criteria:
  - id: ac-001
    summary: molrs 是唯一内核，molpy.core 只有 re-export 与语法糖
    type: code
    pass_when: |
      除 logger.py/script.py/config.py 外，molpy.core 的全部公开符号都在已提交的
      manifest 中被标成 identity re-export 或 Python sugar。AST gate 拒绝 manifest 外的
      class/function。molpy.core 中不存在产生新科学/数据结果的实现：无图算法、
      PBC/坐标数学、元素/单位真值表、FF 公式或参数表。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      `.claude/specs/molrs-core-cutover.manifest.json` classifies every public top-level
      definition; `test_core_cutover_architecture.py` verifies manifest coverage and
      rejects conversion bridges/NumPy linalg kernels. 4 architecture tests pass.

  - id: ac-002
    summary: 纯 re-export 保持对象身份
    type: code
    pass_when: |
      entity.py 导出的 NodeRef/RelationRef/Entity/Link/Atom/Bond/Angle/Dihedral/
      Improper/Bead/CGBond 逐个满足
      `molpy.core.<module>.<Name> is <canonical molrs symbol>`。molpy 顶层导出也指向
      同一对象，没有同名复制类。Frame/Block/Element 则只从 molrs 导入，molpy 不重导出。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      Architecture identity test covers all entity handles, canonical FF classes and
      native scaleLJ exports with object `is` checks; negative gates cover Frame/Block/Element.

  - id: ac-003
    summary: Python 语法糖是真继承，不是转发门面
    type: code
    pass_when: |
      Atomistic/CoarseGrain/Box/Trajectory/UnitSystem 等只在确有 Python 增补时存在子类，
      且逐个 `issubclass(MolpyType, MolrsType)` 为真。对象不含 `_inner`，代码树不含
      `to_molrs`/`from_molrs` bridge。没有“每个方法只转发一次”的门面类。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      Architecture subclass test covers Atomistic/CoarseGrain/Box/Trajectory/UnitSystem
      and Region classes; all are native subclasses and define no `_inner` facade.

  - id: ac-004
    summary: molrs 对 molpy 零依赖，发布顺序正确
    type: code
    pass_when: |
      molrs workspace 中 `import molpy`/`from molpy` 零命中，Cargo/pyproject 也不依赖 molpy。
      molrs 先以包含所有前置 API 的版本发布；molpy 再精确 pin 该版本。
      molpy 不指向未发布的本地偶然状态。
    status: pending
    last_checked:
    evidence:

  - id: ac-005
    summary: Atomistic/CoarseGrain 只有一个 world 与一套图算法
    type: runtime
    pass_when: |
      def_*、删除、拓扑生成、邻居/BFS、copy/merge、子图抽取、Frame 投影、
      translate/rotate/scale/align/replicate 的计算与存储全在 molrs。molpy 只做 handle/
      list/dict/callback/chaining 整形。Atom/Link 句柄身份稳定，写句柄视图立即写回
      Rust world；不存在第二坐标或关系列镜像。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      Graph storage/topology/copy/merge/extract/frame projection and transforms route to
      molrs. Native align_direction and replicate bindings have Rust/Python regression;
      molpy only shapes handles/callbacks and adopts native results. Core suite passes.

  - id: ac-006
    summary: Box 数学只在 molrs，molpy.Box 只剩 Python 表面
    type: scientific
    pass_when: |
      free/orthogonal/triclinic 和 partial-PBC 下，volume/style/lengths/angles/tilts/bounds、
      fractional<->cartesian、wrap/unwrap/images、minimum-image vector/distance、isin 都由
      molrs 计算，并对当前 molpy oracle 在既有容差内一致。molpy/core/box.py
      只剩公开构造器/属性、array/repr/hash/plot/to_dict，无 PBC 数值实现或私有状态别名。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      SimBox/PyBox now own from_bounds/bounds, matrix conversions, face distances,
      wrap/images/unwrap, matched and pairwise MIC displacement/distance, and transform.
      molpy Box contains public shaping only; `_matrix`/`_origin`/`_pbc` are absent.
      Native and molpy Box regressions pass.

  - id: ac-007
    summary: Region 计算委托 native，且 molrs.Region 不被 shadow
    type: runtime
    pass_when: |
      SphereRegion/BoxRegion 的 contains/bounds 与 And/Or/Not 全部使用 molrs native region。
      molpy 只增 coord_field 与 mask(Block)，各类是对应 molrs native 类的真继承，
      不持有 `_inner`/`_region` 转发。`molrs.Region` 仍是原生组合区域类，
      现有 `isinstance(native_composition, molrs.Region)` 测试不回归；
      `molpy.core.region.Region` 的 selection protocol 不通过 top-level shadow 伪装成同一类。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      molpy Region wrappers directly subclass subclassable molrs.Cuboid/Sphere/Region;
      contains/bounds/And/Or/Not are native and no `_inner`/`_region` exists. Region and
      no-forwarding tests pass.

  - id: ac-008
    summary: Element 真值只有 Rust 一份
    type: scientific
    pass_when: |
      crate-root `molrs.Element` 是唯一 Python 导出，覆盖 1..118；name/symbol/Z 大小写
      无关往返；暴露 mass/vdw/covalent。0、X、越界和未知输入 fail-fast。
      `ElementData` 与 molpy/core/element.py 均不存在；无 Python 周期表或半径回退。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      molrs-python Element tests cover 1..118 plus invalid 0/X; molpy negative gates prove
      `molpy.Element`, `molpy.core.Element`, `molpy.core.element`, and `ElementData` are absent.

  - id: ac-009
    summary: UnitSystem 使用 molrs 单位引擎，molpy 不再依赖 Pint
    type: runtime
    pass_when: |
      molrs-python 暴露 UnitRegistry/Unit/Quantity 的 parse/define/quantity/conversion/维度检查/
      arithmetic。molpy.UnitSystem 只在 native registry 上增 LAMMPS preset 名与 LJ 尺度构造。
      7 个 preset 的所有 required dimensions 与 LJ length/energy/time/temperature 转换有回归。
      molpy pyproject 无 pint 运行时依赖。无法同义支持的 Pint context API 明示报错/
      记录 breaking，不静默返回不同语义。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      Rust units 79 tests pass；molrs-python 已暴露 UnitRegistry/Unit/Quantity/UnitsError，
      LJ 派生公式在 UnitRegistry::define_lj_units。molpy UnitSystem 为 native 真子类，
      7 presets 与 LJ 回归通过，pyproject.toml 已删除 pint。

  - id: ac-010
    summary: ForceField/fields/utils 没有第二份数据模型
    type: code
    pass_when: |
      ForceField/Style/Type/Parameters 来自 molrs 唯一状态；molpy 只留固定 style-name 的
      无状态子类与 AtomisticForcefield 兼容别名。规范 FieldSpec 全部来自
      molrs.fields，molpy 只留 SITE 上层 schema 和 formatter。无生产消费者的
      TypeBucket/utils.py 已删除；CSV 入口统一使用 Block.from_csv。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      FF hierarchy and canonical fields are molrs re-exports. Consumer audit confirmed
      TypeBucket/get_nearest_type/dict converters/read_csv had no production use, so the
      entire utils.py module and its top-level exports were removed; CSV is Block.from_csv.

  - id: ac-011
    summary: scaleLJ 公式、参数和 FF 改写全部在 molrs
    type: scientific
    pass_when: |
      FragmentScaling、compute_k_ij、fragment COM 与 pair epsilon/sigma 改写是 molrs FF 能力。
      CL&Pol fragment 参数是 molrs/src/ff/params/ 下的编译期 Rust 表，运行时不解析
      clpol_fragments.ff，也不反向访问 molpy.data。molpy scale_lj 只将 legacy Python
      fragments 整形为 native 输入。回归覆盖：只缩 epsilon、sigma 开关、charge 不变、
      源 FF 不变、缺 fragment/非正 alpha fail-fast。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      FragmentScaling/compute_k_ij/COM/ForceField clone+pair rewrite 位于
      molrs/src/ff/scale_lj.rs；编译期表位于 molrs/src/ff/params/clpol.rs。
      molpy 的参数文件已删除，scale_lj.py 只整形 Atom views。Rust feature-ff 测试、
      molrs-python native 测试及 molpy 原 10 个 scaleLJ 回归通过。

  - id: ac-012
    summary: Python 数值重复被真正删除
    type: code
    pass_when: |
      molpy/src/molpy/core/ops/geometry.py 不存在；core/atomistic.py 和 cg.py 不导入
      _vec_* 帮助函数；box.py 无 wrap/distance/image/matrix-conversion 数值内核；
      element.py 不存在；unit.py 无 Pint 引擎；ops/scale_lj.py 无公式、COM 和 FF 循环。
      对应行为测试已迁到 molrs/molrs-python，不是连测试一起删掉。
    status: pass
    last_checked: 2026-07-15
    evidence: |
      geometry.py, Python element/unit/scaleLJ kernels and runtime CL&Pol table are removed.
      AST gate rejects NumPy linalg and conversion bridges; corresponding native Rust and
      molrs-python behavior suites pass.

  - id: ac-013
    summary: 全量质量门和用户导入不回归
    type: runtime
    pass_when: |
      molrs: cargo fmt --all --check；cargo clippy --all-targets --all-features -- -D warnings；
      cargo check --all-features；cargo test --all-features。molrs-python 全测试与 stub 导出检查通过。
      molpy 全测试、ruff 和 ty 通过。明确保留的 molpy facade 导入仍可用；已删除的
      Frame/Block/Element 旧路径必须失败。文档直接从 molrs 导入这些内核类型。
    status: pending
    last_checked: 2026-07-15
    evidence: |
      Passing: cargo fmt/check/clippy, native 2008 tests, molrs-python 579 tests,
      molpy core 375 tests, ty and full-tree ruff. Full molpy before the final dead-utils
      removal: 1930 passed, 5 skipped,
      but 3 pre-existing dirty example scripts lack main() and fail doc-example tests;
      unrelated user edits were intentionally not overwritten. Release/pin also remains.
---

# Acceptance criteria

本 spec 的核心不是“测试还绿”，而是证明 **第二份内核真的消失了**。

- `ac-001` / `ac-012` 是反向架构门：断言 Python 实现不存在，不只是 molrs API 存在。
- `ac-002` / `ac-003` 把 re-export 与真语法糖分开：没增补就必须 `is`，有增补就必须真继承。
- `ac-006` / `ac-008` / `ac-009` / `ac-011` 要求真值迁移，不允许把旧 Python 算法藏在“兼容层”里。
- `ac-013` 同时守住用户边界：molrs 是实现真值，但用户的世界仍然只需要 molpy。
