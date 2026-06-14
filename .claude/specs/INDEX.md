# Specs

- [molgraph-ecs-03-molpy](molgraph-ecs-03-molpy.md) — molpy 收成 molrs ECS world 上的句柄列视图：Atomistic(_GraphViews, molrs.Atomistic) IS-A 无桥；删 Struct+3 Mixin + _ordered_* 镜像 + from_molrs_graph 拷贝 + _MOLRS_KIND；属性走零拷贝 component 列；system 即自由函数；零硬编码字段=引用 fields 约定+错即报错（chain molgraph-ecs 3/3，依赖 molrs 01+02；supersedes atomistic-cg-on-molrs-molgraph + molgraph-views-02） [draft]
- [cg-atomistic-mapping-redesign](cg-atomistic-mapping-redesign.md) — 简化 CoarseGrain 至与 Atomistic 对称 [draft]
- [molrs-backend](molrs-backend.md) — molrs 作为 molpy 必选后端：Box 继承 + NeighborList/RDF + 替换 RDKit + 暴露 MSD/Cluster 等 [code-complete]
- [molrs-analyses-expose-02](molrs-analyses-expose-02.md) — 暴露剩余 molrs.compute 分析（Steinhardt/Hexatic/Nematic/SolidLiquid、LocalDensity/GaussianDensity、StaticStructureFactorDebye、BondOrder、PMFTXY、ClusterProperties）为 molpy 薄壳算子 [code-complete]
- [clp-typifier-02-forcefield](clp-typifier-02-forcefield.md) — CL&P 离子液体力场：ClpTypifier(OplsTypifier) 继承不合并 + 新建 clp.xml（咪唑阳离子 + BF4/PF6/NTf2/dca 四阴离子），复用 OPLSAAForceFieldReader，按 il.ff 回归（chain clp-typifier 2/2，依赖 01；CL&Pol out of scope） [code-complete]
- [clpol-01-virtualsite-drude](clpol-01-virtualsite-drude.md) — CL&Pol 1/3：core 虚位点数据模型 VirtualSite/DrudeParticle/MasslessSite + 通用增补基类 VirtualSiteBuilder + DrudeBuilder（极化器）+ Tip4pBuilder（证明基类通用）+ alpha.ff 极化率数据;α=q_D²/k_D、电荷守恒、不改入参（molpy 内,不导出） [approved]
- [clpol-02-damping-potentials](clpol-02-damping-potentials.md) — CL&Pol 2/3：potential 层 Thole + Tang-Toennies 阻尼求值器（calc_energy/calc_forces,闭式+解析力 FD 校验+长短程极限），依赖 01 [approved]
- [clpol-03-scalelj](clpol-03-scalelj.md) — CL&Pol 3/3：scaleLJ SAPT 因子缩放 fragment 对 LJ ε（k_ij=1/[1+C0·r²q²/α+C1·μ²/α],μ² 无 r² 前因子）,写回 PairType 不改 σ/电荷,依赖 01 [approved]
