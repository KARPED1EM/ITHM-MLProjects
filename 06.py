import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LinearRegression,SGDRegressor,LogisticRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,root_mean_squared_error,confusion_matrix
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import xgboost as xgb



df=pd.read_csv(r'D:\Files\VerySyncFiles\ITHM\今日资料\05_数据挖掘\04_人才流失项目实战\01_人才流失实战\人才流失预测\train.csv')
# print(df)
# 假设 df 是你的DataFrame
df=pd.get_dummies(df)
df.dropna(inplace=True)
correlation_matrix = df.corr()
dict=dict(correlation_matrix.iloc[0])
feature=[]
for i in dict.keys():
    if (dict[i]>=0.1 or dict[i]<=-0.1) and i !='Attrition':
        feature.append(i)
# print(feature)
x=df[feature]
y=df['Attrition']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
model_1=LinearRegression()
es1=model_1.fit(x_train,y_train)
pre_1=es1.predict(x_test)
# print(mean_squared_error(y_test,pre_1))
# print(root_mean_squared_error(y_test,pre_1))
# print(mean_absolute_error(y_test,pre_1))
model_2=SGDRegressor(eta0=0.01,learning_rate="invscaling")
es2=model_2.fit(x_train,y_train)
pre_2=model_2.predict(x_test)
# print(mean_squared_error(y_test,pre_2))
# print(root_mean_squared_error(y_test,pre_2))
# print(mean_absolute_error(y_test,pre_2))
model_3 = LogisticRegression()
es3 = model_3.fit(x_train, y_train)
pre_3=es3.predict(x_test)
# print(accuracy_score(y_test,pre_3))
# print(classification_report(y_test,pre_3))
model_4= DecisionTreeClassifier(max_depth=3)
es4=model_4.fit(x_train,y_train)
pre_4=es4.predict(x_test)
# print(accuracy_score(y_test,pre_4))
# print(classification_report(y_test,pre_4))
model_5 = RandomForestClassifier(n_estimators=50,max_depth=3)
model_5.fit(x_train,y_train)
pre_5=model_5.predict(x_test)
# print(accuracy_score(y_test,pre_5))
model_6=RandomForestClassifier()
params={'n_estimators':[10,30,50],'max_depth':[1,3,5]}
cv=GridSearchCV(estimator=model_6,param_grid=params,cv=4)
cv.fit(x_train,y_train)
pre_6=cv.best_estimator_.predict(x_test)
# print(cv.best_estimator_,accuracy_score(y_test,pre_6))
model_7= DecisionTreeClassifier(max_depth=3)
model_8=AdaBoostClassifier(estimator=model_7,n_estimators=50,algorithm="SAMME")
model_8.fit(x_train,y_train)
pre_8=model_8.predict(x_test)
# print(accuracy_score(y_test,pre_8))
model_9=xgb.XGBClassifier(max_depth=5, n_estimators=50)
model_9.fit(x_train,y_train)
pre_9=model_9.predict(x_test)
# print(accuracy_score(y_test,pre_9))
params={'max_depth':[2],'n_estimators':[17,18,19,20],'learning_rate':[0.4]}
cv = StratifiedKFold(n_splits=4, shuffle=True)
model_10=GridSearchCV(estimator=model_9,param_grid=params,cv=cv)
model_10.fit(x_train,y_train)
pre_10=model_10.best_estimator_.predict(x_test)
print(model_10.best_estimator_,accuracy_score(y_test,pre_10))
print(roc_auc_score(y_test,pre_2))






