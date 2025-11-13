import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

"""
mpg 데이터에서 숫자 변수들 사이에 다중공선성이 있는지 보고,
서로 너무 비슷하게 움직이는(상관이 너무 높은) 변수들을 자동으로 골라내서 빼고,
빼고 난 다음 VIF가 얼마나 좋아졌는지 다시 보는 코드"""

df = sns.load_dataset("mpg").dropna().copy()
# print(df.head())
# print(df.info())

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
target = "mpg"
num_cols = [c for c in num_cols if c != target]

X = df[num_cols].copy()
y = df[target].copy()

print("Numeric features :", num_cols)
corr = X.corr(numeric_only=True).values
# 상관계수 보기
# plt.figure(figsize=(6, 5))
# sns.heatmap(corr, annot=True, cmap="coolwarm")
# plt.xticks(range(len(num_cols)), labels=num_cols, rotation=45)
# plt.yticks(range(len(num_cols)), labels=num_cols)
# plt.tight_layout()
# plt.show()


def compute_vif(df_num: pd.DataFrame) -> pd.DataFrame:
    Z = sm.add_constant(df_num.values, has_constant="skip")

    vifs = []
    for i in range(1, Z.shape[1]):
        vifs.append(variance_inflation_factor(Z, i))

    return pd.DataFrame({"feature": df_num.columns, "VIF": vifs}).sort_values(
        "VIF", ascending=False
    )


vif_before = compute_vif(X.astype(float))
print(vif_before)


def drop_high_corr_features(df_num: pd.DataFrame, thr: float = 0.9) -> list:
    cols = df_num.columns.tolist()

    while True:
        C = df_num[cols].corr(numeric_only=True)
        np.fill_diagonal(C.values, 0)
        max_corr = C.values.max()
        if max_corr < thr or len(cols) <= 1:
            break
        i, j = np.where(C.values == max_corr)  # (2,3), (3,4) -> i=[2,3], j=[3,4]
        i0, j0 = int(i[0]), int(j[0])
        col_i, col_j = cols[i0], cols[j0]
        mean_i = C.loc[col_i, cols].mean()
        mean_j = C.loc[col_j, cols].mean()

        drop_col = col_i if mean_i >= mean_j else col_j
        cols.remove(drop_col)
    return cols


reduced_cols = drop_high_corr_features(X, thr=0.90)
print("remain features : ", reduced_cols)
X_reduced = X[reduced_cols].copy()
vif_after = compute_vif(X_reduced.astype(float))
print(vif_after)

# 여기 까진 수동으로 한거

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lin_pipe = Pipeline(
    steps=[("scaler", StandardScaler()), ("lin_model", LinearRegression())]
)

ridge_pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)),
    ]
)

lin_pipe.fit(x_tr, y_tr)
ridge_pipe.fit(x_tr, y_tr)

y_pred_lin = lin_pipe.predict(x_te)
y_pred_ridge = ridge_pipe.predict(x_te)


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


print("Linear : ", metrics(y_te, y_pred_lin))
print("Ridge  : ", metrics(y_te, y_pred_ridge))

# KFold로 수동으로 돌려서 훈련 점수 확인 -> 과적합 줄었는지 확인
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores_lin = cross_val_score(lin_pipe, x_tr, y_tr, cv=cv, scoring="r2")
scores_ridge = cross_val_score(ridge_pipe, x_tr, y_tr, cv=cv, scoring="r2")
print("CV R2 Linear Mean: ", scores_lin.mean(), " Std: ", scores_lin.std())
print("CV R2 Ridge  Mean: ", scores_ridge.mean(), " Std: ", scores_ridge.std())
