---
slug: builder-reacter-02-template-io
criteria:
  - id: ac-001
    summary: write_bond_react_map 单元测试通过（四节格式 + 1-based ID + 不一致报错）
    type: runtime
    pass_when: |
      pytest tests/test_io/test_data/test_lammps_bond_react.py -v -k "map" exits 0,
      covering header counts, InitiatorIDs/EdgeIDs/DeleteIDs/Equivalences sections,
      1-based indices, and ValueError on pre/post atom-set mismatch.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: write_lammps_bond_react_system 输出与重构前 golden 夹具字节级一致
    type: runtime
    pass_when: |
      pytest tests/test_io/test_data/test_lammps_bond_react.py -v -k "golden" exits 0;
      the test compares every produced file (.data, .ff, *_pre.mol, *_post.mol, *.map)
      byte-for-byte against fixtures in tests/test_io/test_data/golden/bond_react/
      that were generated with the pre-refactor implementation.
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: import molpy.reacter 不加载 molpy.io
    type: runtime
    pass_when: |
      pytest tests/test_reacter/test_import_hygiene.py -v exits 0; it runs
      python -c "import sys, molpy.reacter; assert 'molpy.io' not in sys.modules"
      in a subprocess and asserts success.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: reacter 包内无 molpy 顶层导入（层违规清零）
    type: code
    pass_when: |
      rg -n "^import molpy|import molpy as|^from molpy import" src/molpy/reacter/
      returns no matches (sub-package imports like "from molpy.core..." allowed;
      "from molpy.typifier..." only under TYPE_CHECKING in base.py).
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: BondReactTemplate 不再含文件 I/O 方法与死代码助手
    type: code
    pass_when: |
      rg -n "def write\b|def write_map|_unify_type_mappings|_ensure_typified"
      src/molpy/reacter/bond_react.py returns no matches, and
      rg -n "def assign_atom_ids" src/molpy/reacter/bond_react.py returns exactly one match.
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: io 写出器不再访问模板私有成员
    type: code
    pass_when: |
      rg -n "_assign_atom_ids|tpl\.write_map" src/molpy/io/writers.py returns no
      matches, and rg -n "write_bond_react_map" src/molpy/io/writers.py and
      src/molpy/io/__init__.py each return at least one match (delegation + export).
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: 迁移后的 reacter 模板写出测试通过
    type: runtime
    pass_when: |
      pytest tests/test_reacter/test_bond_react.py -v exits 0 with no remaining
      reference to template.write( in the file
      (rg -n "template\.write\(" tests/test_reacter/test_bond_react.py empty).
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: 新 io 模块 docstring 完整且 changelog 含迁移说明
    type: docs
    pass_when: |
      Every public function in src/molpy/io/data/lammps_bond_react.py has a
      Google-style docstring referencing https://docs.lammps.org/fix_bond_react.html,
      and rg -n "write_lammps_bond_react_system|write_bond_react_map" docs/changelog.md
      returns at least one match in a migration note for the removed
      BondReactTemplate.write/write_map.
    status: pending
  - id: ac-009
    summary: 全量检查与测试套件通过
    type: runtime
    pass_when: |
      ruff format --check src tests && ruff check src tests &&
      ty check src/molpy/ && pytest tests/ -m "not external" -v all exit 0.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- **ac-001 / ac-002** 锁定行为保持：`.map` 格式契约与系统写出路径的字节级兼容是本规格"行为不变"的可裁决定义。
- **ac-003 / ac-004** 锁定层修复：运行时（sys.modules）与静态（grep）双重断言 reacter 不依赖 io。
- **ac-005 / ac-006** 锁定职责迁移：reacter 无文件 I/O，io 只经公有 API 访问模板。
- **ac-007** 保证既有测试意图在新路径下存活；**ac-008** 覆盖文档义务；**ac-009** 为项目统一收尾闸门。

验证说明（供 /mol:impl 参考）：golden 夹具必须在重构提交之前由当前 HEAD 实现生成并单独提交，否则 ac-002 失去约束力。已核实 `_ensure_typified` 唯一调用方为被删除的 `write()` 路径（src/molpy/reacter/bond_react.py:87-88），删除安全；`docs/user-guide/04_crosslinking.ipynb` 已使用 `write_lammps_bond_react_system`（第 312 行附近），无需修改。
