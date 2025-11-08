import datetime
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from commonUtil import data_preprocessing
from logUtil import Logger
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def model_train(data_x, data_y, logger):

    # logger.info("=========开始模型训练===================")
    # 1.数据集切分
    # x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=20)
    # 2.模型训练
    # logger.info('开始训练xgboost模型...')
    # model = xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=50, learning_rate=0.5)
    # # 模型训练
    # model.fit(x_train, y_train)
    # # 评估准确率
    # y_pre = model.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pre)
    # logger.info(f"模型准确率: {accuracy:.4f}")
    # joblib.dump(model, '../model/xgb.pkl')
    # auc = roc_auc_score(y_test, y_pre)
    # print(auc)
    # print(accuracy_score(y_test, y_pre))
    # return model, accuracy
    logger.info("=========算法比较===================")

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42)
    # # 2. 保存训练集特征顺序（重要！）
    # joblib.dump(x_train.columns.tolist(), '../model/training_features.pkl')
    #  标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    joblib.dump(scaler, '../model/standard_scaler.pkl')
    x_test = scaler.transform(x_test)
    # 定义多个算法
    classifiers = {
        'XGBoost': xgb.XGBClassifier(
            random_state=42,
            objective='binary:logistic',
            learning_rate=0.3,
            n_estimators=100,
            max_depth=1
        ),
        # 随机森林
        'RandomForest': RandomForestClassifier(
            n_estimators=50,
            random_state=22,
            class_weight='balanced',
            max_depth=5
        ),

        # 逻辑回归
        'LogisticRegression': LogisticRegression(
            random_state=22,
            class_weight='balanced',
            max_iter=1500,
            C=0.5,
            penalty="l2",
            solver='sag'
        )
    }

    results = {}
    trained_models = {} # 新增：存储训练好的模型对象
    y_pred_proba_dict = {}
    for name, clf in classifiers.items():
        logger.info(f"训练 {name}...")

        # 训练模型
        clf.fit(x_train, y_train)
        trained_models[name] = clf
        # 预测
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]
        y_pred_proba_dict[name] = y_pred_proba # 保存概率预测
        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        logger.info(f"{name} - 准确率: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")

    # 结果比较
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('auc', ascending=False)

    logger.info("算法性能排名:")
    logger.info(f"\n{results_df}")

    # ============ 新增：绘制图表 ============
    plot_model_results(results_df, y_test, y_pred_proba_dict, logger)

    # 6. 选出最佳模型 (假设以 AUC 为主要指标)
    # 处理可能存在的 NaN AUC 值
    valid_results_df = results_df.dropna(subset=['auc'])
    if not valid_results_df.empty:
        best_model_name = valid_results_df.index[0]  # AUC 最高的模型名称
        best_model = trained_models[best_model_name]  # 对应的模型对象

        # 保存最佳模型
        model_save_path = '../model/best_model.pkl'
        joblib.dump(best_model, model_save_path)
        logger.info(f"性能最佳的模型 ({best_model_name}) 已保存至 {model_save_path}")
    return results_df, classifiers

def dem01_use_mode(data_x, data_y, logger):
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=20)
    logger.info("=========网格搜索和交叉验证===================")
    # model = joblib.load('../model/xgb.pkl')
    #
    # param_dict = {
    #     'max_depth': [1, 3, 5, 10, 15, 20],
    #     'n_estimators': [40, 50, 70, 100,120,150,170,200],
    #     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1]
    # }
    model = LogisticRegression()
    param_dict = {
    'max_iter' : [300,500,700,1000,1200,1500,1700],
    'C' : [0.1,0.3,0.5,0.7,0.9]
    }
    # 交叉验证： 特别适合类别不平衡的数据集,并确保每个训练集和测试集中各类样本的比例与原始数据集相同
    # 创建分层采样: cv = StratifiedKFold
    # n_splits  折数       shuffle 是否打乱数据(作用：防止原始数据顺序对模型训练产生影响)
    cv = StratifiedKFold(n_splits=4, shuffle=True)
    # 交叉 + 网格： GridSearchCV
    cv_model = GridSearchCV(estimator=model, param_grid=param_dict, cv=cv)
    cv_model.fit(x_train, y_train)
    y_pre2 = cv_model.best_estimator_.predict(x_test)
    y_pred_proba = cv_model.best_estimator_.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test,y_pred_proba)
    # 最优评分和模型
    print(cv_model.best_estimator_)
    print(accuracy_score(y_test, y_pre2))
    print(auc)


def plot_model_results(results_df, y_test, y_pred_proba_dict, logger):
    """绘制模型比较图表"""

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能比较', fontsize=16, fontweight='bold')

    # 1. 模型指标对比柱状图
    metrics_to_plot = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    results_df[metrics_to_plot].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('模型性能指标对比')
    axes[0, 0].set_ylabel('分数')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. AUC排名图
    sorted_auc = results_df['auc'].sort_values(ascending=True)
    axes[0, 1].barh(range(len(sorted_auc)), sorted_auc.values)
    axes[0, 1].set_yticks(range(len(sorted_auc)))
    axes[0, 1].set_yticklabels(sorted_auc.index)
    axes[0, 1].set_title('模型AUC排名')
    axes[0, 1].set_xlabel('AUC Score')

    # 在柱子上添加数值
    for i, v in enumerate(sorted_auc.values):
        axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

    # 3. ROC曲线
    from sklearn.metrics import roc_curve
    for name, y_pred_proba in y_pred_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='随机分类器')
    axes[1, 0].set_xlabel('假正率 (False Positive Rate)')
    axes[1, 0].set_ylabel('真正率 (True Positive Rate)')
    axes[1, 0].set_title('ROC曲线比较')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 热力图 - 相关性矩阵
    metrics_corr = results_df[metrics_to_plot].corr()
    sns.heatmap(metrics_corr, annot=True, cmap='coolwarm', center=0,
                ax=axes[1, 1], square=True)
    axes[1, 1].set_title('指标相关性热力图')

    plt.tight_layout()

    # 保存图片
    plt.savefig('../model/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("模型比较图表已保存至 '../model/model_comparison.png'")



class PowerLoadModel(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        self.data_x,self.data_y = data_preprocessing(filename)

if __name__ == '__main__':
    #1.加载处理过的数据集
    power_load_model = PowerLoadModel('../data/train.csv')  # 你的自定义类实例

    # 训练并获取机器学习模型
    trained_xgboost, accuracy = model_train(
        power_load_model.data_x,
        power_load_model.data_y,
        power_load_model.logfile
    )
    # dem01_use_mode(
    #     power_load_model.data_x,
    #     power_load_model.data_y,
    #     power_load_model.logfile)


