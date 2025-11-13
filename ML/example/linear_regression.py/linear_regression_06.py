# linear_regression() 사용
# horsepower, weight 선형성검증하고 다항식 추가해서 성능 개선
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

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

# ----------------------------------------

df = sns.load_dataset("mpg").dropna().copy()
print(df.info())

X = df[["horsepower", "weight"]]
y = df["mpg"]

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 산점도
sns.regplot(x="horsepower", y="mpg", data=df, ax=axes[0, 0], line_kws={"color": "red"})
sns.regplot(x="weight", y="mpg", data=df, ax=axes[0, 1], line_kws={"color": "red"})
axes[0, 0].set_title("horsepower vs mpg")
axes[0, 1].set_title("weight vs mpg")

# 잔차
sns.residplot(
    x="horsepower",
    y="mpg",
    data=df,
    ax=axes[1, 0],
    lowess=True,
    line_kws={"color": "red"},
)
sns.residplot(
    x="weight", y="mpg", data=df, ax=axes[1, 1], lowess=True, line_kws={"color": "red"}
)
axes[1, 0].set_title("horsepower residuals")
axes[1, 1].set_title("weight residuals")

plt.tight_layout()
plt.show()

# 단순 선형회귀
lin_pipe = Pipeline(
    steps=[("scaler", StandardScaler()), ("lin_model", LinearRegression())]
)
lin_pipe.fit(x_tr, y_tr)
y_pred_lin = lin_pipe.predict(x_te)

r2 = r2_score(y_te, y_pred_lin)
mae = mean_absolute_error(y_te, y_pred_lin)
mse = mean_squared_error(y_te, y_pred_lin)
RMSE = np.sqrt(mse)
print("Linear Regression 성능:")
print("R2 :", r2)
print("mae:", mae)
print("mse :", mse)
print("RMSE :", RMSE)

# 다항식 추가
poly_pipe = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)
poly_pipe.fit(x_tr, y_tr)
y_pred_poly = poly_pipe.predict(x_te)

print("\nPoly Regression 성능:")
poly_r2 = r2_score(y_te, y_pred_poly)
poly_mae = mean_absolute_error(y_te, y_pred_poly)
poly_mse = mean_squared_error(y_te, y_pred_poly)
poly_RMSE = np.sqrt(poly_mse)
print("R2 :", poly_r2)
print("mae:", poly_mae)
print("mse :", poly_mse)
print("RMSE :", poly_RMSE)

print("\n 변경 값 (poly - line)")
print("R2 :", poly_r2 - r2)
print("mae:", poly_mae - mae)
print("mse :", poly_mse - mse)
print("RMSE :", poly_RMSE - RMSE)

plt.figure(figsize=(6, 4))
plt.scatter(y_te, y_pred_lin, label="Linear", alpha=0.6)
plt.scatter(y_te, y_pred_poly, label="Poly", alpha=0.6)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")
plt.legend()
plt.title("다항식 추가 전 후 비교")
plt.tight_layout()
plt.show()

"""
Linear Regression 성능:
R2 : 0.6514190280854426
mae: 3.5056538974903253
mse : 17.791776112838146
RMSE : 4.218029885247157

Poly Regression 성능:
R2 : 0.6883062442194608
mae: 3.0211866941739345
mse : 15.909031087262887
RMSE : 3.988612677017272

 변경 값 (poly - line)
R2 : 0.03688721613401824  좋아는 졌는데 큰 의미 없음
mae: -0.4844672033163908  0.4만큼 덜 틀림
mse : -1.8827450255752591  
RMSE : -0.2294172082298851  모델 오차가 10%정도 줄음. 유의미한 성능 향상

"""
