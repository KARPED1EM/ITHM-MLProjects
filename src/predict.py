import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from commonUtil import data_preprocessing, pre_data_preprocessing
from logUtil import Logger

import pandas as pd
import joblib

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
    # print(auc)
    # print(report)
    logger.info(f"模型准确率: {accuracy:.4f}")
    logger.info(f"详细分类报告:{report}")
    logger.info(f'AUC值:{auc}')
    logger.info(f"\n{classification_report(y_test, y_pre)}")
    return accuracy, report

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