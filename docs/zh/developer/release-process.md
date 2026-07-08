# 发布流程

本文档是 MolPy 版本发布的操作检查清单。


## 版本号权威来源

版本号定义在 `src/molpy/version.py` 中。发布前更新以下两个字段：

```python
version = "X.Y.Z"
release_date = "YYYY-MM-DD"
```


## 发布前检查

创建发布分支前，先在本地做完这三项验证：

```bash
pytest tests/ -v -m "not external"    # 测试通过
zensical build                         # 文档构建
python -m build && twine check dist/*  # 包有效
```


## 发布工作流

**master 分支受保护**，发布工作流也拒收从 `master` 不可达的标签。因此发布提交必须**先**合入 `master`，**再**推送标签——否则会产生孤立标签，发布任务直接失败。顺序不能乱：

```bash
# 1. 在 dev 分支上更新 version.py 和 CHANGELOG，提交。
# 2. 通过 PR 将发布提交合并到 master（直接推送被拒绝）：
gh pr create --base master --head dev --title "Release vX.Y.Z"
gh pr merge --merge            # 检查通过后执行

# 3. 只有在 master 包含发布提交后，才打标签并推送：
git fetch main master
git tag -a vX.Y.Z -m "Release vX.Y.Z"    # 在合并后的 master 提交上打标签
git push main vX.Y.Z
```

**不要**用 `git push <remote> master --tags`：推送 master 时受保护分支规则可能拦下提交，但标签还是会以孤立形式流出去，发布流程不认。

推送 `v*` 标签会触发 GitHub Actions 执行 `.github/workflows/release.yml`。工作流先核对标签与 `molpy.version.version` 一致，再跑测试、构建产物、发布到 PyPI。


## 夜间构建

夜间构建独立于标签发布流程运行，发布到另一个 PyPI 项目 `molcrafts-molpy-nightly`，不动稳定版 `molcrafts-molpy`。

- **触发方式：** 推送到 `nightly` 分支，或通过 `workflow_dispatch` 手动运行 *Nightly* 工作流（`.github/workflows/nightly.yml`）。
- **版本号：** 工作流自动读取 `molpy.version.version`，追加 UTC 时间戳，生成 `X.Y.Z.dev<YYYYMMDDHHMM>`（PEP 440 开发版）。不需要手动改版本号或打标签；**不要**为夜间构建编辑 `version.py`。
- **发行版重命名：** 构建时自动将 PyPI 包名改为 `molcrafts-molpy-nightly`，`nightly` 分支上的提交不受影响。
- **发布：** 通过 PyPI Trusted Publishing（OIDC）发布到 `pypi-nightly` GitHub Environment。不需要 API 令牌，不需要人工审批——夜间构建不会卡在等待审核上。

发布一个夜间构建，把 `nightly` 分支快进到目标提交然后推送：

```bash
git push origin master:nightly      # 或将你需要的分支推送到 nightly
```

用 `pip install --pre molcrafts-molpy-nightly` 安装。导入时包名也是 `molpy`，跟稳定版冲突，必须在专用虚拟环境中测试。


## 发布说明

GitHub Releases 页面按以下模板填写：

```markdown
## MolPy vX.Y.Z

### Added（新增）
- ...

### Changed（变更）
- ...

### Fixed（修复）
- ...

### Breaking Changes（破坏性变更）
- ...（若无则填"无"）
```


## 热修复

针对已发布版本的紧急修复：

```bash
git checkout -b hotfix/vX.Y.Z vA.B.C
# 修复、测试、更新 version.py
git commit -am "fix: ..."
git tag -a vX.Y.Z -m "Hotfix vX.Y.Z"
git push origin --tags
```
