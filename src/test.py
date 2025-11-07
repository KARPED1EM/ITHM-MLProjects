import torch
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from train import Model, DataSet

def load_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    business_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df_train['BusinessTravel'] = df_train['BusinessTravel'].map(business_map)
    df_test['BusinessTravel'] = df_test['BusinessTravel'].map(business_map)
    df_train.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    df_test.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    categorical = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    df_train = pd.get_dummies(df_train, columns=categorical, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical, drop_first=True)
    common_cols = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]
    feature = df_train.drop(['Attrition'], axis=1).values
    label = df_train['Attrition'].values
    test_feature = df_test.drop(['Attrition'], axis=1).values
    test_label = df_test['Attrition'].values
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(x_train)
    test_feature = scaler.transform(test_feature)
    test_dataset = DataSet(test_feature, test_label)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=128 , shuffle=False)
    return test_data_loader

def model_test(model, test_data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../model/best_model.pth'))
    model = model.to(device)

    all_predicts = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for test_x, test_y in test_data_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            model.eval()

            output = model(test_x)
            pre_label = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1)[:, 1]
            all_predicts.extend(pre_label.cpu().numpy())
            all_labels.extend(test_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_predicts)
    prec = precision_score(all_labels, all_predicts)
    rec = recall_score(all_labels, all_predicts)
    f1 = f1_score(all_labels, all_predicts)
    auc = roc_auc_score(all_labels, all_probs)

    print('\n' + '=' * 50)
    print('Final Test Results:')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-score : {f1:.4f}')
    print(f'Auc      : {auc:.4f}')
    print('\nDetailed Classification Report:')
    print(classification_report(all_labels, all_predicts, target_names=['Not Attrition', 'Attrition']))
    print(confusion_matrix(all_labels, all_predicts))
    print('=' * 50)

if __name__ == '__main__':
    test= load_data()
    model = Model()
    model_test(model, test)