# 测试

MolPy 用 pytest 做测试。测试文件放在 `tests/` 下，目录结构跟包一一对应：`tests/test_core/`、`tests/test_io/`、`tests/test_parser/` 等等。

## 运行测试

```bash
pytest tests/ -v -m "not external"                     # 标准本地运行
pytest tests/test_core/test_frame.py -v                # 单个文件
pytest tests/test_core/test_frame.py::test_creation -v # 单个测试
pytest tests/ -k "lammps" -v                           # 关键字过滤
pytest --cov=src/molpy tests/ -v --cov-report=term     # 带覆盖率
```

`-m "not external"` 排除需要外部可执行文件（LAMMPS、Packmol、AmberTools）的测试。本地开发和 CI 默认都走这个配置。

## 测试内容

每加一个新行为，就要补一个对应的测试。覆盖四类场景：

1. **正常路径**——正常输入下能否返回正确结果？
2. **边界情况**——空输入、单元素、边界值
3. **错误处理**——非法输入时抛出正确的异常
4. **回归测试**——修完 bug，加一个测试确认它不会再冒出来

MolPy 特有代码还有两类常见测试模式：

**不可变性检查**——验证操作返回新对象，不改输入。

```python
def test_typify_does_not_mutate():
    original = build_test_mol()
    result = typifier.typify(original)
    assert result is not original
    assert len(original.atoms) == original_count
```

**往返测试**——对 I/O 格式，写出去再读回来，确认数据不变。

```python
def test_pdb_round_trip(tmp_path):
    write_pdb(tmp_path / "out.pdb", frame)
    restored = read_pdb(tmp_path / "out.pdb")
    assert restored["atoms"].nrows == frame["atoms"].nrows
```

## 标记（Markers）

需要外部可执行文件的测试，用 `@pytest.mark.external` 标记：

```python
import pytest

@pytest.mark.external
def test_lammps_integration():
    # 需要 PATH 里有 lmp_serial
    ...
```

这样没装对应工具的机器也能跑默认测试套件，不会报错。

## 编写良好的测试

断言行为，不是断言实现细节。测试只应在可观察结果变化时失败，重构内部实现不影响它。Fixture 保持小巧——为了测一个键操作就搭 100 个原子，那测试范围太大了。文件 I/O 测试用 `tmp_path` fixture，别污染工作目录。
