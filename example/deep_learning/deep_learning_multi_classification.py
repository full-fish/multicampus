import torch  # 파이토치 메인 패키지
import torch.nn as nn  # 신경망 모듈(레이어/손실/활성화 등)
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)  # 텐서를 Dataset/Loader로 감싸는 유틸

import numpy as np  # 수치 연산(클래스 개수 계산 등)
from sklearn.datasets import load_iris  # 다중분류 예제: 아이리스 데이터셋
from sklearn.model_selection import train_test_split  # 학습/테스트 분할
from sklearn.preprocessing import StandardScaler  # 특성 표준화(평균0, 표준편차1)
from sklearn.linear_model import LogisticRegression  # 베이스라인 분류기(로지스틱 회귀)
from sklearn.metrics import accuracy_score  # 정확도 계산 함수

# =====================================================================
# 1. 데이터 불러오기 (iris) + train/test 분할 + 스케일링
# =====================================================================
data = load_iris()  # 아이리스 데이터 로드
X = data.data  # (n_samples, n_features)  입력 특성 행렬
y = data.target  # (n_samples,)             정수 라벨 0,1,2

print("특성 개수:", X.shape[1])  # 특성(컬럼) 수 출력
print(
    "클래스 라벨:", data.target_names
)  # 라벨 이름 출력: ['setosa' 'versicolor' 'virginica']

# train/test 분할 (stratify=y로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # 80/20 분할 + 시드 고정 + 계층화 분할
)

# 스케일링 (입력 특성만)
scaler = StandardScaler()  # 표준화 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # 학습셋 통계로 학습+변환
X_test_scaled = scaler.transform(
    X_test
)  # 테스트셋은 학습셋 통계로만 변환(데이터 누수 방지)

# =====================================================================
# 2. 기준 모델: Logistic Regression (다중분류 소프트맥스)
# =====================================================================
# multi_class="multinomial" + 적절한 solver(lbfgs)로 다중분류 소프트맥스 수행
log_clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
log_clf.fit(X_train_scaled, y_train)  # 스케일된 학습셋으로 파라미터 학습

y_pred_log = log_clf.predict(X_test_scaled)  # 테스트셋 예측(정수 라벨)
acc_log = accuracy_score(y_test, y_pred_log)  # 정확도 계산
print(f"[Logistic Regression] Test Accuracy: {acc_log:.4f}")  # 베이스라인 성능 출력

# =====================================================================
# 3. PyTorch용 Tensor / Dataset / DataLoader 준비
# =====================================================================
# numpy -> torch tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)  # 입력을 float32 텐서로
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)  # 테스트 입력 텐서

# CrossEntropyLoss는 라벨을 long 타입 + shape (N,) 로 기대함(원-핫 아님)
y_train_t = torch.tensor(y_train, dtype=torch.long)  # 학습 라벨(정수, (N,))
y_test_t = torch.tensor(y_test, dtype=torch.long)  # 테스트 라벨(정수, (N,))

# Dataset, DataLoader
train_dataset = TensorDataset(
    X_train_t, y_train_t
)  # (입력, 라벨) 튜플로 인덱싱 가능한 Dataset
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)  # 미니배치 학습 + 셔플


# =====================================================================
# 4. 딥러닝 MLP 분류 모델 정의 (다중분류, 출력 = 클래스 개수)
# =====================================================================
class MLPClassifierMulti(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),  # 입력차원 -> 32 은닉 노드
            nn.ReLU(),  # ReLU 활성화(비선형성, 기울기 소실 완화)
            nn.Linear(32, 16),  # 32 -> 16 은닉층
            nn.ReLU(),  # ReLU 활성화
            nn.Linear(16, num_classes),  # 최종 출력: 클래스 개수만큼 로짓(logits)
        )

    def forward(self, x):
        return self.net(x)  # 소프트맥스 전 단계의 로짓 반환


in_dim = X_train_t.shape[1]  # 입력 특성 수
num_classes = len(np.unique(y_train))  # 클래스 개수(여기선 3)

model = MLPClassifierMulti(in_dim, num_classes)  # 모델 인스턴스 생성

# 다중분류용 손실: CrossEntropyLoss(내부에서 LogSoftmax + NLLLoss를 처리)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)  # Adam 옵티마이저, 학습률 0.001

# =====================================================================
# 5. 학습 루프
# =====================================================================
num_epochs = 50  # 전체 에폭 수

for epoch in range(num_epochs):  # 에폭 반복
    model.train()  # 학습 모드로 전환
    running_loss = 0.0  # 에폭 손실 누적 변수

    for xb, yb in train_loader:  # 미니배치 반복
        optimizer.zero_grad()  # 직전 스텝의 gradient 초기화

        logits = model(xb)  # 순전파: (batch, num_classes) 로짓 출력
        # ★ CrossEntropyLoss는 (logits, 정수 라벨) 입력을 그대로 받음(소프트맥스 불필요)
        loss = criterion(logits, yb)  # 배치 평균 손실 계산

        loss.backward()  # 역전파: 각 파라미터의 gradient 계산
        optimizer.step()  # 경사하강 스텝: 파라미터 업데이트

        running_loss += loss.item() * xb.size(
            0
        )  # 배치 손실 * 배치 크기 → 샘플수 가중합

    epoch_loss = running_loss / len(train_dataset)  # 에폭 평균 손실

    if (epoch + 1) % 10 == 0:  # 10에폭마다 로깅
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

# =====================================================================
# 6. 테스트셋에서 MLP 성능 평가 (정확도)
# =====================================================================
model.eval()  # 평가 모드(드롭아웃/BN 고정)
with torch.no_grad():  # 추론 시 그래프 저장/기울기 계산 비활성화
    logits_test = model(X_test_t)  # 테스트셋 로짓 예측 (확률 아님)
    y_pred_mlp = logits_test.argmax(
        dim=1
    ).numpy()  # 각 샘플에서 최고 점수의 클래스 인덱스

acc_mlp = accuracy_score(y_test, y_pred_mlp)  # 정확도 계산
print(f"\n[MLP (PyTorch, Multi-class)] Test Accuracy: {acc_mlp:.4f}")  # 최종 성능 출력
