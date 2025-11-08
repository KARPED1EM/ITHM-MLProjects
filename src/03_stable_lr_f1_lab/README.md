# 03 · Stable LR F1 Lab

这个项目固定使用 `overall_ranking.csv` 中的获胜方案：少量特征交互、log1p、标准化，One Hot

1. 运行一个轻量级的超参数网格，涵盖少量正则化设置
2. 不同的上采样方法（虽然结果是没有上采样是最好的）

| 子目录        | 内容                                                   |
| ---------- | ---------------------------------------------------- |
| `data/`    | `sampler_summary.csv`、完整网格结果、验证集阈值扫描、预测结果、精简 JSON 报告 |
| `figures/` | 各采样器 F1 柱状图、阈值 vs F1 曲线、（最佳阈值下的）混淆矩阵、PR 曲线           |
| `models/`  | `stable_lr_best_<sampler>.pkl` —— 可直接用于 notebook 或部署 |
