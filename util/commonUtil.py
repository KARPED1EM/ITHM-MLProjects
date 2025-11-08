import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def data_preprocessing(path):
    """
    1.获取数据源
    2.去重
    3.剔除出现空值的样本
    4.特征热编码
    5.特征筛选排序
    :param path:
    :return:
    """
    # 1.获取数据源
    data = pd.read_csv(path)
    x = data.iloc[:,1:]
    y = data.iloc[:,0]
    # 2.去重
    x.drop_duplicates(inplace=True)
    # 3.剔除出现空值的样本
    y = y.loc[x.index]
    x.dropna(axis=0,inplace=True)
    x.drop('Over18',axis=1,inplace=True)
    # 4.特征热编码
    df_encoded = pd.get_dummies(x,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'])
    # 使用PCA降维
    X_final = df_encoded
    pca = PCA(n_components=min(2,df_encoded.shape[1]))  # 这里降到2维是为了方便可视化
    X_pca = pca.fit_transform(X_final)
    # 使用特征选择
    model = RandomForestClassifier(random_state=20)
    model.fit(X_final, y)
    importances = pd.Series(model.feature_importances_, index=X_final.columns).sort_values(ascending=False)
    # print("\n特征重要性排名:")
    # print(importances)
    feature_columns = ['MonthlyIncome','Age','TotalWorkingYears','DistanceFromHome',
                       'EmployeeNumber','YearsAtCompany','PercentSalaryHike','OverTime_Yes',
                       'StockOptionLevel','NumCompaniesWorked','JobSatisfaction',
                       'EnvironmentSatisfaction','YearsWithCurrManager','YearsInCurrentRole',
                       'JobInvolvement','TrainingTimesLastYear','YearsSinceLastPromotion',
                       'RelationshipSatisfaction','WorkLifeBalance','JobLevel','Education','MaritalStatus_Single',
                       'MaritalStatus_Divorced','MaritalStatus_Married']
    result_x = df_encoded[feature_columns]
    result_y = y

    return result_x, result_y

def pre_data_preprocessing(path):
    """
    1.获取数据源
    2.去重
    3.剔除出现空值的样本
    4.特征筛选
    :param path:
    :return:
    """
    # 1.获取数据源
    data = pd.read_csv(path)
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    # 2.去重
    x.drop_duplicates(inplace=True)
    # 3.剔除出现空值的样本
    y = y.loc[x.index]
    x.dropna(axis=0,inplace=True)

    # x.drop('Over18',axis=1,inplace=True)
    # 4.特征热编码
    df_encoded = pd.get_dummies(x, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                                            'MaritalStatus', 'OverTime'])
    # 筛选出特征列
    feature_columns = ['MonthlyIncome','Age','TotalWorkingYears','DistanceFromHome',
                       'EmployeeNumber','YearsAtCompany','PercentSalaryHike','OverTime_Yes',
                       'StockOptionLevel','NumCompaniesWorked','JobSatisfaction',
                       'EnvironmentSatisfaction','YearsWithCurrManager','YearsInCurrentRole',
                       'JobInvolvement','TrainingTimesLastYear','YearsSinceLastPromotion',
                       'RelationshipSatisfaction','WorkLifeBalance','JobLevel','Education','MaritalStatus_Single',
                       'MaritalStatus_Divorced','MaritalStatus_Married']
    result_x = df_encoded[feature_columns]
    result_y = y

    return result_x, result_y

if __name__ == '__main__':
# #     # input_file = pd.read_csv('../data/train.csv')
# #     result = data_preprocessing('../data/train.csv')
    result = pre_data_preprocessing('../data/test.csv')
    print(result)