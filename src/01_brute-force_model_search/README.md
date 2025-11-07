# AttriPredict – Brute-Force Model Search

基于模型家族执行一次系统性的暴力搜索，涵盖 11 种模型类型、每个家族 4 套特征处理流程，以及两个固定的随机种子（42 和 2025）。线性模型采用激进的特征工程策略，树模型尽量保持接近原始信号，而神经网络则更关注特征缩放及其与 SMOTE 的结合

> 共尝试 332 种组合（KFold），在 i9-13900HX / 64GB RAM 下运行 70 分钟左右

## 产物结构

```
src/01_brute-force_model_search/
├── artifacts/
│   └── fair_bruteforce/
│       ├── summaries/                # CSV 导出文件和清单
│       ├── global_figures/           # 跨模型家族的对比图
│       ├── linear/
│       │   ├── data/                 # 线性模型结果表格
│       │   ├── figures/              # 线性模型相关图表
│       │   └── models/               # 前十线性模型（joblib 格式）
│       ├── tree/
│       │   └── ...
│       └── neural/
│           └── ...
├── brute-force_model_search.ipynb
└── README.md
```
