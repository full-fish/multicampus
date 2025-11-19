import torch  # 파이토치 메인 패키지
import torch.nn as nn  # 신경망(레이어/손실/활성화 등) 모듈
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)  # 텐서를 Dataset/Loader로 감싸는 유틸

import numpy as np  # 수치 연산(정확도 계산 등에 보조)
from sklearn.datasets import load_breast_cancer  # 유방암 이진분류 예제 데이터
from sklearn.model_selection import train_test_split  # 학습/테스트 분할
from sklearn.preprocessing import StandardScaler  # 표준화 스케일러(평균0, 표준편차1)
from sklearn.linear_model import LogisticRegression  # 선형 분류 베이스라인
from sklearn.metrics import accuracy_score  # 정확도(accuracy) 계산 함수

data = load_breast_cancer()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.4, stratify=y_train
)

print("Train :", x_tr.shape, "Valid :", x_va.shape, "Test :", x_test.shape)

scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_va_scaled = scaler.transform(x_va)
x_test_scaled = scaler.transform(x_test)

x_tr_t = torch.tensor(x_tr_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

x_va_t = torch.tensor(x_va_scaled, dtype=torch.float32)
y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

x_te_t = torch.tensor(x_test_scaled, dtype=torch.float32)
y_te_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_tr_t, y_tr_t)
val_dataset = TensorDataset(x_va_t, y_va_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


in_dim = x_tr_t.shape[1]
model = MLPClassifier(in_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam()
