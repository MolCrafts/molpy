# 演示模型

`aspirin.glb` 用 **molvis 自带的 glTF 导出器**生成。引擎渲染用的球棍几何体直接序列化成自包含的 glTF 文件，不依赖 molvis 运行时就能查看。**不是**手写几何体。

- 源结构：`aspirin.sdf`（PubChem CID 2244，3D 构象）。
- 导出器：`@molcrafts/molvis-core` 中的 `exportFrameToGLB` / `MolvisApp.exportGLTF`。

重新生成（molvis 在同级目录 `../molvis`）：

```sh
cd ../molvis
npx tsx core/scripts/gen-molecule-glb.mts \
  ../molpy/docs/assets/models/aspirin.sdf \
  ../molpy/docs/assets/models/aspirin.glb
```

颜色（CPK）、半径、分色键和键级都来自 molvis 的渲染缓冲区，所以导出的模型和 molvis 绘制的结果一致。材质用哑光 PBR（金属度 0，粗糙度 0.9）。
