# =========================================================
# mpg 회귀: (1) 선형성 검증(잔차 vs 각 피처)
#          (2) 다항식은 horsepower, weight만 2차/교차항 추가
# =========================================================
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1) 데이터 로드 및 피처 선택
df = sns.load_dataset("mpg")

# 사용할 컬럼 정의
poly_cols = ["horsepower", "weight"]  # 다항식(2차) 적용 대상
num_all = [
    "horsepower",
    "weight",
    "acceleration",
    "displacement",
    "cylinders",
    "model_year",
]
cat_cols = ["origin"]  # 범주형(원-핫)
use_cols = ["mpg"] + num_all + cat_cols

# 결측 제거
df = df[use_cols].dropna()

X = df[num_all + cat_cols]
y = df["mpg"]

# 학습/평가 분리
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# A) 베이스라인: 순수 선형(모든 피처 선형), 스케일은 수치형만
# ---------------------------------------------------------
num_rest = [c for c in num_all]  # 전 수치형을 선형으로 사용
num_linear_transformer = Pipeline([("scaler", StandardScaler())])

cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", num_linear_transformer, num_rest),  # 수치형만 표준화
        ("cat", cat_transformer, cat_cols),  # 범주형 원-핫
    ],
    remainder="drop",
)

pipe_linear = Pipeline([("prep", preprocess_linear), ("model", LinearRegression())])

pipe_linear.fit(X_tr, y_tr)

# (선형성 검증) 훈련셋 잔차 계산
y_tr_pred_lin = pipe_linear.predict(X_tr)
resid_lin = y_tr - y_tr_pred_lin

# (성능) 테스트셋
y_te_pred_lin = pipe_linear.predict(X_te)
r2_lin = r2_score(y_te, y_te_pred_lin)
mae_lin = mean_absolute_error(y_te, y_te_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_te, y_te_pred_lin))

print("=== [Linear] Test 성능 ===")
print(f"R²   : {r2_lin:.4f}")
print(f"MAE  : {mae_lin:.4f}")
print(f"RMSE : {rmse_lin:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col in zip(axes, ["horsepower", "weight"]):
    ax.scatter(X_tr[col], resid_lin, alpha=0.6)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel(col)
    ax.set_title(f"Residuals vs {col} (Linear baseline)")
axes[0].set_ylabel("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_tr_pred_lin, resid_lin, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predict (Linear baseline)")
plt.xlabel("Predict")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# B) 다항 모델: horsepower, weight만 2차 + 교차항, 나머지는 선형
#    - poly_cols -> PolynomialFeatures(deg=2)
#    - 다른 수치형/범주형은 베이스와 동일
# ---------------------------------------------------------
poly_transformer = Pipeline(
    [
        (
            "poly",
            PolynomialFeatures(degree=2, include_bias=False),
        ),  # [hp, wt, hp^2, hp*wt, wt^2]
        ("scaler", StandardScaler()),
    ]
)

num_rest_no_poly = [c for c in num_all if c not in poly_cols]  # 다항 미적용 수치형
num_rest_transformer = Pipeline([("scaler", StandardScaler())])

preprocess_poly = ColumnTransformer(
    transformers=[
        ("polyNum", poly_transformer, poly_cols),  # hp, wt만 다항+스케일
        ("numRest", num_rest_transformer, num_rest_no_poly),  # 나머지 수치형 스케일
        ("cat", cat_transformer, cat_cols),  # 범주형 원-핫
    ],
    remainder="drop",
)

pipe_poly = Pipeline([("prep", preprocess_poly), ("model", LinearRegression())])

pipe_poly.fit(X_tr, y_tr)

# (선형성 검증) 훈련셋 잔차 계산
y_tr_pred_poly = pipe_poly.predict(X_tr)
resid_poly = y_tr - y_tr_pred_poly
# print("resid_poly", resid_poly)

# (성능) 테스트셋
y_te_pred_poly = pipe_poly.predict(X_te)
r2_poly = r2_score(y_te, y_te_pred_poly)
mae_poly = mean_absolute_error(y_te, y_te_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_te, y_te_pred_poly))

print("\n=== [Polynomial on hp, wt only] Test 성능 ===")
print(f"R²   : {r2_poly:.4f}")
print(f"MAE  : {mae_poly:.4f}")
print(f"RMSE : {rmse_poly:.4f}")

# ---------------------------------------------------------
# C) 성능 비교표
# ---------------------------------------------------------
comp = pd.DataFrame(
    {
        "Model": [
            "Linear (all features linear)",
            "Poly(deg=2) on [horsepower, weight]",
        ],
        "R2": [r2_lin, r2_poly],
        "MAE": [mae_lin, mae_poly],
        "RMSE": [rmse_lin, rmse_poly],
    }
)
print("\n=== 성능 비교 (Test) ===")
print(comp)

# ---------------------------------------------------------
# D) 선형성 검증: 잔차 vs 각 피처(훈련셋, 베이스라인 잔차 기준)
#    - 잔차가 0 주변에 무작위면 선형성 OK
#    - 곡률(U/∩) 보이면 해당 피처에 다항/변환 고려
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col in zip(axes, ["horsepower", "weight"]):
    ax.scatter(X_tr[col], resid_poly, alpha=0.6)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel(col)
    ax.set_title(f"Residuals vs {col} (Linear poly)")
axes[0].set_ylabel("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_tr_pred_poly, resid_poly, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predict (Linear Poly)")
plt.xlabel("Predict")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))

# ----------------------------------------
# 선형모델 예측
plt.scatter(y_te, y_te_pred_lin, label="Linear", alpha=0.6)

# 다항모델 예측
plt.scatter(y_te, y_te_pred_poly, label="Poly(hp, weight, deg=2)", alpha=0.6)

# 이상적 예측선
min_v, max_v = y_te.min(), y_te.max()
plt.plot([min_v, max_v], [min_v, max_v], "r--")

plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")
plt.legend()
plt.title("다항식 추가 전후 예측 비교 (Test)")
plt.tight_layout()
plt.show()
