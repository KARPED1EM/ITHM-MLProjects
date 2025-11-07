import copy
import torch
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Relu(self.fc1(x))
        x = f.dropout(x, 0.5, training=self.training)
        x = self.Relu(self.fc2(x))
        x = f.dropout(x, 0.3, training=self.training)
        x = self.fc3(x)
        return x

class DataSet(Dataset):
    def __init__(self, x, y):
        self.feature = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]

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
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    test_feature = scaler.transform(test_feature)
    train_dataset = DataSet(x_train, y_train)
    val_dataset = DataSet(x_test, y_test)
    test_dataset = DataSet(test_feature, test_label)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=128 , shuffle=False)
    return train_data_loader, val_data_loader, test_data_loader

def model_train(model, train_data_loader, val_data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_f1 = 0.0
    train_acc_list = []
    val_acc_list = []
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(100):
        train_corrects = 0
        train_total = 0
        for b_x, b_y in train_data_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            model.train()
            output = model(b_x)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_corrects += (torch.argmax(output, dim=1) == b_y).sum().item()
            train_total += b_y.size(0)
            train_acc = train_corrects / train_total
        print(f'Epoch {epoch + 1}, Train Acc: {train_acc:.4f}')

        with torch.no_grad():
            val_corrects = 0
            val_f1 = 0
            val_total = 0
            all_val_predicts = []
            all_val_labels = []
            for b_x, b_y in val_data_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                model.eval()
                output = model(b_x)
                pre_label = torch.argmax(output, dim=1)
                val_corrects += (pre_label == b_y).sum().item()
                val_total += b_y.size(0)
                all_val_predicts.extend(pre_label.cpu().numpy())
                all_val_labels.extend(b_y.cpu().numpy())
            val_acc = val_corrects / val_total
            val_f1 = f1_score(all_val_labels, all_val_predicts)
            print(f'Epoch {epoch + 1}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            print('*' * 20)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())
            print(f"🎉 New best F1: {best_acc:.4f}, saving model...")

    torch.save(best_model, '../model/best_model2.pth')
    print(f"Training finished. Best Val F1: {best_f1:.4f}")
    print('*' * 20)

if __name__ == '__main__':
    train, val, test= load_data()
    model = Model()
    model_train(model, train, val)