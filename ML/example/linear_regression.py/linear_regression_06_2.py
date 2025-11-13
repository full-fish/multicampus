import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. 데이터 불러오기
df = sns.load_dataset("mpg").dropna().copy()

target = "mpg"
y = df[target]
X = df.drop(columns=[target])

# 2. 컬럼 분리
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 다항식으로 키울 애들 (mpg에서 비선형일 가능성 있는 것들)
poly_features = ["displacement", "horsepower", "weight", "acceleration"]
# 실제 있는 컬럼만 남기기 (혹시 없을 수도 있으니까)
poly_features = [c for c in poly_features if c in num_cols]

# 나머지 숫자
num_rest = [c for c in num_cols if c not in poly_features]

# 3. 학습/테스트 분리
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 4-1. 기본 버전: 다항식 없이 숫자 스케일 + 범주형 원핫
basic_ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

basic_model = Pipeline(
    steps=[
        ("prep", basic_ct),
        ("reg", LinearRegression()),
    ]
)

basic_model.fit(x_tr, y_tr)
pred_basic = basic_model.predict(x_te)

# 4-2. 다항식 버전: 일부 숫자만 2차항으로 늘리고, 나머지는 스케일, 범주형은 원핫
poly_ct = ColumnTransformer(
    transformers=[
        (
            "poly_num",
            Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler()),
                ]
            ),
            poly_features,
        ),
        ("num_rest", StandardScaler(), num_rest),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

poly_model = Pipeline(
    steps=[
        ("prep", poly_ct),
        ("reg", LinearRegression()),
    ]
)

poly_model.fit(x_tr, y_tr)
pred_poly = poly_model.predict(x_te)


# 5. 성능 비교
def print_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print()


print_metrics("기본 모델 (스케일 + 원핫)", y_te, pred_basic)
print_metrics("다항식 모델 (일부 숫자 2차)", y_te, pred_poly)

# 6. 시각적으로도 한 번
plt.figure(figsize=(6, 5))
plt.scatter(y_te, pred_basic, label="basic", alpha=0.6)
plt.scatter(y_te, pred_poly, label="poly(2)", alpha=0.6)
min_v, max_v = y_te.min(), y_te.max()
plt.plot([min_v, max_v], [min_v, max_v], "r--")
plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")
plt.title("기본 vs 다항식 예측 비교")
plt.legend()
plt.tight_layout()
plt.show()
