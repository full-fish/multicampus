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

# =====================================================================
# 1. 데이터 불러오기 (breast_cancer) + train/test 분할 + 스케일링
# =====================================================================
data = load_breast_cancer()  # 사이킷런 내장 유방암 데이터셋 로드
X = data.data  # (n_samples, n_features)  입력 특성 행렬
y = data.target  # (n_samples,)             타깃 벡터(0/1 이진라벨)

print("특성 개수:", X.shape[1])  # 특성(컬럼) 개수 출력
print("클래스 라벨:", data.target_names)  # 라벨 이름 출력(['malignant','benign'] 등)

# train/test 분할 (stratify=y로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # 80% 학습/20% 테스트, 클래스 비율 동일하게 분리

# 스케일링 (입력 특성만)
scaler = StandardScaler()  # 표준화 객체 생성(학습셋 통계로 맞춤)
X_train_scaled = scaler.fit_transform(X_train)  # 학습셋 평균/표준편차 계산 후 변환
X_test_scaled = scaler.transform(
    X_test
)  # 테스트셋은 학습셋 통계로만 변환(데이터 누수 방지)

# =====================================================================
# 2. 기준 모델: Logistic Regression (선형 분류모델)
# =====================================================================
log_clf = LogisticRegression(
    max_iter=1000
)  # 로지스틱 회귀 분류기(최대 반복수 1000으로 설정)
log_clf.fit(X_train_scaled, y_train)  # 스케일된 학습셋으로 파라미터 학습

y_pred_log = log_clf.predict(X_test_scaled)  # 스케일된 테스트셋 예측(0/1)
acc_log = accuracy_score(y_test, y_pred_log)  # 정확도 계산
print(f"[Logistic Regression] Test Accuracy: {acc_log:.4f}")  # 베이스라인 성능 출력

# =====================================================================
# 3. PyTorch용 Tensor / Dataset / DataLoader 준비
# =====================================================================
# numpy -> torch tensor
X_train_t = torch.tensor(
    X_train_scaled, dtype=torch.float32
)  # 학습 입력을 float32 텐서로 변환
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
    1
)  # 타깃을 (N,) -> (N,1)로 변환(BCE 호환)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)  # 테스트 입력 텐서
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 테스트 타깃 (N,1)

# Dataset, DataLoader
train_dataset = TensorDataset(
    X_train_t, y_train_t
)  # (입력, 타깃) 쌍으로 인덱싱 가능한 Dataset
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)  # 미니배치 학습용 로더(셔플로 일반화 ↑)


# =====================================================================
# 4. 딥러닝 MLP 분류 모델 정의 (이진분류, 출력 1개 + BCEWithLogitsLoss)
# =====================================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),  # 입력차원 -> 32차원 은닉층
            nn.ReLU(),  # ReLU 활성화(기울기 소실 방지/학습 안정화)
            nn.Linear(32, 16),  # 32 -> 16 은닉층
            nn.ReLU(),  # ReLU 활성화
            nn.Linear(16, 1),  # 최종 출력 1개(로짓 값). 시그모이드는 손실에서 내부 처리
        )

    def forward(self, x):
        return self.net(x)  # 순전파: (배치, in_dim) -> (배치, 1) 로짓(logit) 반환


in_dim = X_train_t.shape[1]  # 입력 특성 차원 수
model = MLPClassifier(in_dim)  # 모델 인스턴스 생성

criterion = (
    nn.BCEWithLogitsLoss()
)  # 이진분류용 손실(시그모이드+BCELoss를 합친 안정적 버전)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)  # Adam 옵티마이저, 학습률 0.001

# =====================================================================
# 5. 학습 루프
# =====================================================================
num_epochs = 50  # 전체 에폭 수

for epoch in range(num_epochs):  # 에폭 반복
    model.train()  # 학습 모드(드롭아웃/BN 등 학습 동작, 여기선 ReLU만)
    running_loss = 0.0  # 에폭 손실 누적 변수 초기화

    for xb, yb in train_loader:  # 미니배치 반복
        optimizer.zero_grad()  # 직전 스텝의 gradient 누적분 0으로 초기화

        logits = model(xb)  # 순전파: 예측 점수(로짓) 계산, shape=(batch,1)
        loss = criterion(logits, yb)  # 손실 계산: 내부에서 sigmoid(logits) 후 BCE 계산

        loss.backward()  # 역전파: 각 파라미터의 gradient 계산
        optimizer.step()  # 경사하강: gradient를 사용해 파라미터 업데이트

        running_loss += loss.item() * xb.size(
            0
        )  # 배치 평균 손실 * 배치 크기 -> 샘플수 가중합으로 누적

    epoch_loss = running_loss / len(train_dataset)  # 에폭 단위 평균 손실(정확한 평균)

    if (epoch + 1) % 10 == 0:  # 10에폭마다 로그 출력
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

# =====================================================================
# 6. 테스트셋에서 MLP 성능 평가 (정확도)
# =====================================================================
model.eval()  # 평가 모드(추론 시 설정)
with torch.no_grad():  # 추론 단계: 그래프/gradient 계산 비활성화로 효율↑
    logits_test = model(X_test_t)  # 테스트셋 로짓 예측(확률 아님, 임계 전 점수)
    # print(logits_test)                            # 필요 시 로짓 값 확인용

    probs_test = torch.sigmoid(logits_test)  # 로짓 -> 시그모이드 확률(0~1)
    # print(probs_test)                             # 필요 시 확률 값 확인용

    y_pred_mlp = (
        (probs_test >= 0.5).int().squeeze(1).numpy()
    )  # 0.5 임계로 0/1 이진 예측 생성

acc_mlp = accuracy_score(y_test, y_pred_mlp)  # 정확도 계산
print(f"\n[MLP (PyTorch)] Test Accuracy: {acc_mlp:.4f}")  # 최종 테스트 정확도 출력
