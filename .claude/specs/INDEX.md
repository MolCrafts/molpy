# Specs

- [molgraph-ecs-03-molpy](molgraph-ecs-03-molpy.md) — molpy 收成 molrs ECS world 上的句柄列视图：Atomistic(_GraphViews, molrs.Atomistic) IS-A 无桥；删 Struct+3 Mixin + _ordered_* 镜像 + from_molrs_graph 拷贝 + _MOLRS_KIND；属性走零拷贝 component 列；system 即自由函数；零硬编码字段=引用 fields 约定+错即报错（chain molgraph-ecs 3/3，依赖 molrs 01+02；supersedes atomistic-cg-on-molrs-molgraph + molgraph-views-02） [draft]
- [cg-atomistic-mapping-redesign](cg-atomistic-mapping-redesign.md) — 简化 CoarseGrain 至与 Atomistic 对称 [draft]
- [molrs-backend](molrs-backend.md) — molrs 作为 molpy 必选后端：Box 继承 + NeighborList/RDF + 替换 RDKit + 暴露 MSD/Cluster 等 [code-complete]
- [molrs-analyses-expose-02](molrs-analyses-expose-02.md) — 暴露剩余 molrs.compute 分析（Steinhardt/Hexatic/Nematic/SolidLiquid、LocalDensity/GaussianDensity、StaticStructureFactorDebye、BondOrder、PMFTXY、ClusterProperties）为 molpy 薄壳算子 [code-complete]
