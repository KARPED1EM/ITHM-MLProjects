# 02 · Brute-Force LR Search

从穷举搜索中筛选出四个逻辑回归（LR）模型组合，每个都服务于不同的生产防护目标：

- lr_score_chaser：看个乐子模型，最大化测试集的 AUC。
- lr_prod_auc_guardian：生产优先模型，针对最高交叉验证（CV）AUC 进行调优，并保持低折叠方差。
- lr_prod_precision_guardian：生产精度守护模型，倾向于减少误报（假阳性）。
- lr_prod_recall_guardian：生产召回守护模型，为敏感告警设置最低召回率。

本来应该是这样的，改着改着不再穷举了，也发现花里胡哨的东西都是白费功夫
