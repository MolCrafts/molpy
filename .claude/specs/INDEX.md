# Specs

- [opls-typifier-downsink](opls-typifier-downsink.md) — molpy 消费侧：OPLS typifier+估计器整体下沉 molrs 后，molpy OplsTypifier 退成 molrs.OplsTypifier 薄壳委托、删 Python 分型实现(_OplsAtomTypifier/ForceField*Typifier/SMARTS 引擎)、pin 新 molrs；对应 molrs 链 opls-typifier-01/02/03 + ff-parameter-estimator；删除前 molrs parity 门须绿。**BLOCKED (2026-06-19)**: molrs 尚未 PyO3 暴露 `OplsTypifier`（`dir(molrs)` 仅有 `MMFFTypifier`）——需 molrs 先 bind `PyOplsTypifier` + parity 绿 + 发布，molpy 才能委托并删 Python 分型 [approved]
