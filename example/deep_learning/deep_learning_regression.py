import torch  # 파이토치 핵심 패키지
import torch.nn as nn  # 신경망 모듈(레이어, 손실 등)
from torch.utils.data import TensorDataset, DataLoader  # 텐서 → 데이터셋/로더 유틸
import seaborn as sns  # 예제 데이터셋(mpg) 로드용
import pandas as pd  # 판다스(여기서는 크게 사용 X)
import numpy as np  # 수치 연산 및 RMSE 계산용
from sklearn.model_selection import train_test_split  # 학습/테스트 분할
from sklearn.preprocessing import StandardScaler  # 입력 스케일링
from sklearn.linear_model import LinearRegression  # 베이스라인 선형회귀
from sklearn.metrics import mean_squared_error  # RMSE 계산용 MSE

torch.manual_seed(0)  # 파이토치 난수 고정(재현성)
np.random.seed(0)  # 넘파이 난수 고정(재현성)

df = sns.load_dataset("mpg").dropna().copy()  # mpg 데이터 로드, 결측 제거

X = df[["horsepower", "weight"]].values  # 특징 2개(마력, 무게)만 추출 → ndarray
y = df["mpg"].values  # 타깃 mpg 추출 → ndarray

x_train, x_test, y_train, y_test = train_test_split(  # 학습/테스트 분할
    X, y, test_size=0.2, random_state=42  # 테스트 20%, 시드 고정
)

scaler = StandardScaler()  # 표준화 스케일러(평균 0, 표준편차 1)
x_train_scaled = scaler.fit_transform(x_train)  # 학습셋으로 평균/표준편차 학습+변환
x_test_scaled = scaler.transform(x_test)  # 테스트셋은 같은 파라미터로 변환

lin_reg = LinearRegression()  # 스킷런 선형회귀 베이스라인
lin_reg.fit(x_train_scaled, y_train)  # 스케일된 입력으로 학습
y_pred_lr = lin_reg.predict(x_test_scaled)  # 테스트셋 예측
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # RMSE 계산
print(f"[Linear Regression] Test RMSE : {rmse_lr:.4f}")  # 베이스라인 성능 출력

x_train_t = torch.tensor(
    x_train_scaled, dtype=torch.float32
)  # 학습 입력을 torch 텐서로
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
    1
)  # 학습 타깃을 (N,1)로
x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)  # 테스트 입력 텐서
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 테스트 타깃 (N,1)

train_dataset = TensorDataset(
    x_train_t, y_train_t
)  # 텐서 쌍을 인덱싱 가능한 데이터셋으로
test_dataset = TensorDataset(x_test_t, y_test_t)  # 테스트 데이터셋도 동일하게

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 미니배치+셔플
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 평가는 셔플 X


class MLPRegressor(nn.Module):  # 다층퍼셉트론 회귀 모델 정의
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(  # 순차적으로 레이어 쌓기
            nn.Linear(2, 16),
            nn.ReLU(),  # 입력 2차원 → 은닉 16, ReLU 활성화
            nn.Linear(16, 16),
            nn.ReLU(),  # 은닉 → 은닉
            nn.Linear(16, 1),  # 은닉 → 출력 1차원(회귀라 활성화 없음)
        )

    def forward(self, x):  # 순전파 정의: model(x) 시 호출됨
        return self.net(x)


model = MLPRegressor()  # 모델 인스턴스 생성
criterion = nn.MSELoss()  # 회귀용 손실함수: 평균제곱오차
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 최적화, lr=0.01

num_epochs = 200  # 학습을 200 에폭 수행

for epoch in range(num_epochs):  # 에폭 루프 시작
    model.train()  # 학습 모드로 전환(드롭아웃/BN 등 학습 동작)
    se_sum = 0.0  # 훈련 RMSE 계산용 제곱오차 누적 변수
    n_train = 0  # 훈련 샘플 수 누적

    for xb, yb in train_loader:  # 미니배치 반복
        optimizer.zero_grad()  # 이전 스텝의 기울기 초기화
        pred = model(xb)  # 순전파로 예측값 계산
        loss = criterion(pred, yb)  # 배치 평균 MSE 계산
        loss.backward()  # 역전파로 기울기 계산
        optimizer.step()  # 가중치 업데이트

        se_sum += ((pred - yb) ** 2).sum().item()  # 배치 제곱오차 합 누적
        n_train += xb.size(0)  # 샘플 개수 누적

    train_rmse = np.sqrt(se_sum / n_train)  # 에폭 단위 훈련 RMSE 계산

    model.eval()  # 평가 모드 전환
    with torch.no_grad():  # 추론 시 그래프/기울기 비활성화
        se_sum_te = 0.0  # 테스트 RMSE 계산용 제곱오차 누적
        n_test = 0  # 테스트 샘플 수 누적
        for xb, yb in test_loader:  # 테스트 배치 반복
            pred = model(xb)  # 예측
            se_sum_te += ((pred - yb) ** 2).sum().item()  # 제곱오차 누적
            n_test += xb.size(0)  # 샘플 수 누적
        test_rmse = np.sqrt(se_sum_te / n_test)  # 테스트 RMSE

    if (epoch + 1) % 20 == 0:  # 20에폭마다 로그 출력
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
        )

print(f"\n[Sklearn Linear Regression] Test RMSE: {rmse_lr:.4f}")  # 선형회귀 RMSE 요약
print(f"[PyTorch MLP]             Test RMSE: {test_rmse:.4f}")  # 최종 MLP RMSE 요약
