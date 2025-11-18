"""
실습 목표
1) seaborn 'mpg' 회귀 데이터로 모델링 (target='mpg')
2) train/valid/test 분할
3) LinearRegression 베이스라인 → 테스트 MSE, RMSE
4) MLP 회귀(입력-은닉-출력)
5) 학습 루프 200에폭: 에폭별 train/valid MSE 기록, Early Stopping
6) 최종 테스트 MSE, RMSE
7) 시각화로 과적합 분석 + 모델 비교
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ===== 한글 폰트 설정(플랫폼별) =====
from matplotlib import font_manager
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")  # 윈도우: 맑은고딕
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")  # 맥: 애플고딕
else:
    plt.rc("font", family="NanumGothic")  # 리눅스: 나눔고딕 (설치 필요할 수 있음)
plt.rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지

# ===== 1) 데이터 로드 =====
df = sns.load_dataset("mpg").dropna().copy()  # 결측 제거 후 복사
print(df)  # 데이터 확인(선택)

# 사용 특성: horsepower, weight (2개) / 타깃: mpg
X = df[["horsepower", "weight"]].values.astype(np.float32)
y = df["mpg"].values.astype(np.float32)

# ===== 2) train/valid/test 분할 =====
# 1단계: test 20% 분리 → 남은 80%는 train+valid
x_trainval, x_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2단계: 남은 80%에서 valid 25% 분리 → 최종 비율: train 60%, valid 20%, test 20%
valid_ratio_within_train = 0.25
x_train, x_valid, y_train, y_valid = train_test_split(
    x_trainval, y_trainval, test_size=valid_ratio_within_train, random_state=42
)

# 스케일링(입력만): 학습셋 통계로 맞추고(valid/test는 transform만)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# ===== 3) 선형회귀 베이스라인(LinearRegression) =====
lin_reg = LinearRegression()
lin_reg.fit(x_train_scaled, y_train)  # 학습
y_pred_lr = lin_reg.predict(x_test_scaled)  # 테스트 예측
mse_lr = mean_squared_error(y_test, y_pred_lr)  # 테스트 MSE
rmse_lr = float(np.sqrt(mse_lr))  # 테스트 RMSE
print(f"[Linear Regression] Test MSE: {mse_lr:.4f} | RMSE: {rmse_lr:.4f}")


# ===== 4) 딥러닝 회귀모델(MLP) 정의 =====
class MLPRegressor(nn.Module):
    """
    입력 2 → 은닉 16(ReLU) → 은닉 16(ReLU) → 출력 1
    간단하지만 비선형성을 학습할 수 있어 선형모델보다 유연함
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MLPRegressor()
criterion = nn.MSELoss()  # 회귀 손실: MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 옵티마이저

# ===== 텐서/데이터로더 준비 =====
# 넘파이 → 토치 텐서 (타깃은 (N,1)로 변환해 MSE에서 브로드캐스팅 혼선을 방지)
x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_valid_t = torch.tensor(x_valid_scaled, dtype=torch.float32)
y_valid_t = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_train_t, y_train_t)
valid_dataset = TensorDataset(x_valid_t, y_valid_t)
test_dataset = TensorDataset(x_test_t, y_test_t)
"""
train_loader: 가중치를 학습시키는 용도(셔플, 역전파)
valid_loader: 에폭마다 일반화 성능을 점검하고 하이퍼파라미터/early stopping을 결정하는 용도
test_loader: 모든 결정이 끝난 뒤 최종 성능을 한 번만 재는 용도

학습(train) batch_size는 작게
이유: 배치가 작을수록 기울기(gradient)에 적당한 노이즈가 생겨 일반화에 도움이 되고, 메모리 여유도 생김.
또, 역전파가 있으니 메모리·연산량이 커서 너무 큰 배치는 오히려 느리거나 OOM이 나기 쉬움.

검증/테스트(valid/test) batch_size는 크게
이유: 역전파가 없어서 메모리 부담이 훨씬 적고, 큰 배치로 “적은 스텝 수”로 빨리 끝낼 수 있음.
model.eval()이면 Dropout 꺼지고 BatchNorm은 러닝 통계를 쓰니까 배치 크기가 결과에 영향 거의 없음.
그래서 가능한 한 “메모리에 맞는 최대치”로 두면 속도가 좋아짐. 128, 256, 심지어 전체 한 번에도 가능
""" ""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ===== 5) 학습 루프(200에폭) + Early Stopping =====
num_epochs = 200
patience = 20  # 개선 없이 연속 20에폭이면 중단
train_losses, valid_losses = [], []

# Early Stopping 상태값
best_state = None  # best 가중치 스냅샷
best_epoch = -1  # best 시점(사람 기준 1부터 표시)
best_valid = np.inf  # 지금까지 본 검증 MSE 최소값
no_improve = 0  # 연속 미개선 카운터

for epoch in range(num_epochs):
    # ----- train 단계 -----
    model.train()
    se_sum, n_train = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)  # 순전파
        loss = criterion(pred, yb)  # 배치 평균 MSE
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트

        # 에폭 평균을 정확히 계산: 배치 제곱오차 총합을 누적
        se_sum += ((pred - yb) ** 2).sum().item()
        n_train += xb.size(0)
    train_mse = se_sum / n_train  # 학습셋 MSE
    train_losses.append(train_mse)

    # ----- valid 단계 -----
    model.eval()
    with torch.no_grad():
        se_sum_v, n_valid = 0.0, 0
        for xb, yb in valid_loader:
            pred = model(xb)
            se_sum_v += ((pred - yb) ** 2).sum().item()
            n_valid += xb.size(0)
        valid_mse = se_sum_v / n_valid  # 검증셋 MSE
        valid_losses.append(valid_mse)

    # 진행 로그(20에폭마다 + 첫 에폭)
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train MSE {train_mse:.4f} (RMSE {np.sqrt(train_mse):.4f}) | "
            f"Valid MSE {valid_mse:.4f} (RMSE {np.sqrt(valid_mse):.4f})"
        )

    # ----- Early Stopping 판정(검증 MSE 기준) -----
    # 개선이면: 최저 기록 갱신 + 가중치 스냅샷 저장 + 카운터 초기화
    if valid_mse < best_valid:
        best_valid = valid_mse
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())  # 깊은 복사로 안전 저장
        no_improve = 0
    else:
        # 미개선이면: 연속 미개선 카운터 증가 → patience 도달 시 중단
        no_improve += 1
        if no_improve >= patience:
            print(
                f"Early Stopping 발동: epoch {epoch+1}에서 중지 (best epoch={best_epoch})"
            )
            break

# best 에폭의 가중치로 복원(마지막 에폭이 아닌, 최적 시점 성능으로 평가하기 위함)
if best_state is not None:
    model.load_state_dict(best_state)

# ===== 6) 테스트 평가(MSE, RMSE) =====
model.eval()
with torch.no_grad():
    preds_list, ys_list = [], []
    for xb, yb in test_loader:
        preds_list.append(model(xb).squeeze(1).cpu().numpy())
        ys_list.append(yb.squeeze(1).cpu().numpy())
y_pred_mlp = np.concatenate(preds_list)  # MLP 테스트 예측
y_true = np.concatenate(ys_list)  # 테스트 실제값

mse_mlp = mean_squared_error(y_true, y_pred_mlp)
rmse_mlp = float(np.sqrt(mse_mlp))
print(f"[MLP Regression]  Test MSE: {mse_mlp:.4f} | RMSE: {rmse_mlp:.4f}")
print(
    f"(Best epoch={best_epoch}, Best Valid MSE={best_valid:.4f}, RMSE={np.sqrt(best_valid):.4f})"
)

# ===== 7) 시각화 =====
# 7-1) 학습/검증 손실 곡선: 과적합 신호와 Early Stopping 지점 확인
plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Train MSE")
plt.plot(valid_losses, label="Valid MSE")
if best_epoch > 0:
    plt.axvline(best_epoch - 1, linestyle="--", label=f"Best epoch = {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("학습/검증 손실 곡선 (과적합 분석 및 Early Stopping 지점)")
plt.legend()
plt.tight_layout()
plt.show()
# 해석 가이드:
# - Train은 내려가는데 Valid가 어느 지점부터 되돌아 올라가기 시작하면 과적합 신호.
# - 점선(Best epoch)은 검증 MSE 최저 시점 → 그때의 가중치로 복원해 테스트를 평가함.

# 7-2) 실제 vs 예측 산점도(Linear / MLP) + 7-3) 모델 오차 막대그래프(MSE, RMSE)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 대각 기준선 범위를 두 모델/실제 전체 값에서 공통으로 산정
lo = min(y_test.min(), y_pred_lr.min(), y_true.min(), y_pred_mlp.min())
hi = max(y_test.max(), y_pred_lr.max(), y_true.max(), y_pred_mlp.max())

# (좌) Linear 산점도
axes[0].scatter(y_test, y_pred_lr, alpha=0.7)
axes[0].plot([lo, hi], [lo, hi], lw=2)
axes[0].set_title(f"Linear Regression\nMSE {mse_lr:.3f}, RMSE {rmse_lr:.3f}")
axes[0].set_xlabel("True MPG")
axes[0].set_ylabel("Predicted MPG")

# (중) MLP 산점도
axes[1].scatter(y_true, y_pred_mlp, alpha=0.7)
axes[1].plot([lo, hi], [lo, hi], lw=2)
axes[1].set_title(f"MLP Regression\nMSE {mse_mlp:.3f}, RMSE {rmse_mlp:.3f}")
axes[1].set_xlabel("True MPG")
axes[1].set_ylabel("Predicted MPG")

# (우) 모델 비교 막대그래프(MSE, RMSE)
labels = ["Linear", "MLP"]
mse_values = [mse_lr, mse_mlp]
rmse_values = [rmse_lr, rmse_mlp]
x = np.arange(len(labels))
barw = 0.35
axes[2].bar(x - barw / 2, mse_values, width=barw, label="MSE")
axes[2].bar(x + barw / 2, rmse_values, width=barw, label="RMSE")
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)
axes[2].set_ylabel("Error")
axes[2].set_title("Linear Regressor VS Deep Learning (MSE, RMSE)")
axes[2].legend()

fig.suptitle("테스트셋: 실제 vs 예측 비교 + 모델 오차 비교", y=0.97)
fig.tight_layout()
plt.show()

# ===== 분석 메모(요약) =====
# • 모델 비교: 테스트 MSE/RMSE에서 MLP가 Linear보다 낮음 → 비선형 표현력이 이득.
# • 산점도: 두 모델 모두 대각선 부근 분포, MLP가 약간 더 촘촘(큰 오차 감소).
# • 손실 곡선: Best epoch 이후 Valid MSE 개선 정체/상승 → Early Stopping으로 과적합 구간 진입 전 종료.
# • 추가 개선 팁: 특성 추가(예: cylinders, displacement, acceleration, model_year),
#                 정규화(weight_decay), 드롭아웃, 은닉 차원/학습률/배치 크기 튜닝,
#                 또는 선형모델에 다항/상호작용 특성 추가로 베이스라인 상향.
