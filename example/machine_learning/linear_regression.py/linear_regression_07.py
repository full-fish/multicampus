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
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 한글
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # 리눅스 계열 (예: 구글코랩, 우분투)
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

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
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

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
#
pipe_linear = Pipeline([("prep", preprocess_linear), ("model", LinearRegression())])

pipe_linear.fit(x_tr, y_tr)


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

lin_model = LinearRegression()
lasso_model = LassoCV(
    alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000, random_state=42
)
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

models = {
    "linear_base": Pipeline([("prep", preprocess_linear), ("model", lin_model)]),
    "ridge_base": Pipeline([("prep", preprocess_linear), ("model", ridge_model)]),
    "lasso_base": Pipeline([("prep", preprocess_linear), ("model", lasso_model)]),
    "linear_poly": Pipeline([("prep", preprocess_poly), ("model", lin_model)]),
    "ridge_poly": Pipeline([("prep", preprocess_poly), ("model", ridge_model)]),
    "lasso_poly": Pipeline([("prep", preprocess_poly), ("model", lasso_model)]),
}
# lasso_linear_y_tr_pred = models["lasso_base"].predict(x_tr)
# lasso_linear_resid = y_tr - lasso_linear_y_tr_pred


# ---------------------------------------------------------
# D) 학습 및 평가
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
    }


results = []
train_residuals = {}
test_residuals = {}
train_preds = {}
for name, model in models.items():
    model.fit(x_tr, y_tr)
    y_te_pred = model.predict(x_te)  # model이 pipe임
    metrics = get_metrics(y_te, y_te_pred)
    metrics["model_name"] = name
    results.append(metrics)

    y_tr_pred = model.predict(x_tr)
    train_preds[name] = y_tr_pred
    train_residuals[name] = y_tr - y_tr_pred
    test_residuals[name] = y_te - y_te_pred

print("results", results)
print("train_residuals", train_residuals)
results_df = pd.DataFrame(results)[["model_name", "R2", "MAE", "RMSE"]]
print("\n=== 모델별 Test 성능 비교 ===")
print(results_df)

# 시각화

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

metrics = ["R2", "MAE", "RMSE"]
colors = ["red", "blue", "green"]

for i, row in enumerate(results):
    ax = axes[i]
    values = [row[m] for m in metrics]
    ax.barh(metrics, values, color=colors, alpha=0.8)
    # 가장 큰 값 기준으로 해서 보기 좋게
    ax.set_xlim(
        0,
        max(results_df["R2"].max(), results_df["MAE"].max(), results_df["RMSE"].max())
        * 1.1,
    )
    ax.set_title(row["model_name"])
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 각 막대 끝에 값 표시
    for j, v in enumerate(values):
        ax.text(v + 0.02, j, f"{v:.3f}", va="center")

plt.suptitle("모델별 성능 비교 (R² / MAE / RMSE)", fontsize=14)
plt.tight_layout()
plt.show()

# 잔차 그래프
fig, axes = plt.subplots(6, 3, figsize=(16, 12))
axes = axes.reshape(6, 3)

for i, name in enumerate(models.keys()):
    resid = train_residuals[name]
    y_pred = train_preds[name]

    # ① horsepower vs Residuals
    axes[i, 0].scatter(x_tr["horsepower"], resid, alpha=0.6)
    axes[i, 0].axhline(0, linestyle="--", color="red", linewidth=1)
    axes[i, 0].set_xlabel("horsepower")
    axes[i, 0].set_ylabel("Residuals" if i == 0 else "")
    axes[i, 0].set_title(f"{name}: Residuals vs horsepower")

    # ② weight vs Residuals
    axes[i, 1].scatter(x_tr["weight"], resid, alpha=0.6)
    axes[i, 1].axhline(0, linestyle="--", color="red", linewidth=1)
    axes[i, 1].set_xlabel("weight")
    axes[i, 1].set_title(f"{name}: Residuals vs weight")

    # ③ Predicted mpg vs Residuals
    axes[i, 2].scatter(y_pred, resid, alpha=0.6)
    axes[i, 2].axhline(0, linestyle="--", color="red", linewidth=1)
    axes[i, 2].set_xlabel("Predicted mpg")
    axes[i, 2].set_title(f"{name}: Residuals vs Predicted")

plt.suptitle("모델별 잔차 비교 (훈련셋 기준)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
