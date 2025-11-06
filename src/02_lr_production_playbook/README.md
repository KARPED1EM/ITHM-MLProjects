# AttriPredict - LR Production Playbook

使用逻辑回归的生产流程，产出分两条线路：

- `lr_score_chaser`：以最高测试集 AUC 为基准，看个乐子
- `lr_validation_guardian`：以最稳定验证集 AUC 为基准，面向生产环境
