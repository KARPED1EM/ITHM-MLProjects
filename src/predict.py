import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from commonUtil import data_preprocessing, pre_data_preprocessing
from logUtil import Logger
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def model_predict(data_x, data_y, logger):
    logger.info("=========开始测试训练===================")
    x_test, y_test = data_x, data_y
    # # 1.2 加载训练时保存的特征顺序
    # joblib.load('../model/training_features.pkl')
    # 1.3 标准化
    scaler = joblib.load('../model/standard_scaler.pkl')
    x_test = scaler.transform(x_test)
    # 2.模型调用
    logger.info('开始调用最优模型...')
    model = joblib.load('../model/best_model.pkl')
    # 3.模型预测
    y_pre = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pre)
    print(accuracy_score(y_test, y_pre))
    report = classification_report(y_test, y_pre)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(auc)
    # print(report)
    logger.info(f"模型准确率: {accuracy:.4f}")
    logger.info(f"详细分类报告:{report}")
    logger.info(f'AUC值:{auc}')
    logger.info(f"\n{classification_report(y_test, y_pre)}")

    # ============ 新增：绘制预测结果图表 ============
    plot_prediction_results(y_test, y_pre, y_pred_proba, model, logger)
    return accuracy, report


def plot_prediction_results(y_test, y_pred, y_pred_proba, model, logger):
    """绘制预测结果图表"""

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型预测结果分析', fontsize=16, fontweight='bold')

    # 1. 混淆矩阵
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('混淆矩阵')

    # 2. ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    axes[0, 1].set_xlabel('假正率 (False Positive Rate)')
    axes[0, 1].set_ylabel('真正率 (True Positive Rate)')
    axes[0, 1].set_title('ROC曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测概率分布
    axes[0, 2].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
                    bins=20, alpha=0.7, label=['真实类别0', '真实类别1'],
                    color=['red', 'blue'])
    axes[0, 2].set_xlabel('预测概率')
    axes[0, 2].set_ylabel('样本数量')
    axes[0, 2].set_title('预测概率分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 分类报告热力图
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)

    # 只保留数值型数据
    numeric_cols = ['precision', 'recall', 'f1-score', 'support']
    report_numeric = report_df[numeric_cols].iloc[:-3]  # 去掉最后三行（accuracy等）

    sns.heatmap(report_numeric, annot=True, cmap='YlOrRd', fmt='.3f',
                ax=axes[1, 0], cbar_kws={'label': '分数'})
    axes[1, 0].set_title('分类报告热力图')

    # 5. 特征重要性（如果模型有feature_importances_属性）
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            # 只显示前20个最重要的特征
            top_features = feature_importance.tail(20)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('重要性')
            axes[1, 1].set_title('Top 20 特征重要性')
        else:
            # 对于逻辑回归，使用系数绝对值作为重要性
            if hasattr(model, 'coef_'):
                coef_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(model.coef_[0]))],
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=True)

                top_features = coef_importance.tail(20)
                axes[1, 1].barh(range(len(top_features)), top_features['importance'])
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features['feature'])
                axes[1, 1].set_xlabel('系数绝对值')
                axes[1, 1].set_title('Top 20 特征重要性(系数)')
            else:
                axes[1, 1].text(0.5, 0.5, '无特征重要性数据',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('特征重要性')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'特征重要性计算错误: {str(e)}',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('特征重要性')

    # 6. 预测结果对比
    results_comparison = pd.DataFrame({
        '真实标签': y_test,
        '预测标签': y_pred,
        '预测概率': y_pred_proba
    })

    # 随机选择50个样本显示
    sample_size = min(50, len(results_comparison))
    sample_results = results_comparison.sample(sample_size, random_state=42).sort_index()

    x_pos = range(len(sample_results))
    axes[1, 2].scatter(x_pos, sample_results['真实标签'],
                       color='blue', label='真实标签', alpha=0.6, s=50)
    axes[1, 2].scatter(x_pos, sample_results['预测标签'],
                       color='red', marker='x', label='预测标签', alpha=0.6, s=50)
    axes[1, 2].set_xlabel('样本索引')
    axes[1, 2].set_ylabel('标签')
    axes[1, 2].set_title('预测结果对比 (50个随机样本)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('../model/prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("预测结果图表已保存至 '../model/prediction_results.png'")



class PowerLoadPredict(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "predict_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_x,self.data_y = pre_data_preprocessing(filename)


if __name__ == '__main__':
    pred_obj = PowerLoadPredict('../data/test.csv')

    trained_xgboost, accuracy = model_predict(
        pred_obj.data_x,
        pred_obj.data_y,
        pred_obj.logfile
    )