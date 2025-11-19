import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = sns.load_dataset("mpg").dropna().copy()

y = df["mpg"]
X = df.drop(columns=["mpg", "name"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()

print("num_cols: ", num_cols)
print("cat_cols: ", cat_cols)

# ? 다중공정성 검사
X_num = df[num_cols].astype(float).copy()
X_num = sm.add_constant(X_num)

vif_values = []

for i in range(1, X_num.shape[1]):
    vif_values.append(variance_inflation_factor(X_num.values, i))
vif = pd.Series(vif_values, index=X_num.columns[1:]).sort_values(ascending=False)

print(vif)

# ?
# preprocess = ColumnTransformer(
#     transformers=[
#         ("num", "passthrough", num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#     ],
#     remainder="drop",
# )

# model = Pipeline(steps=[("prep", preprocess), ("reg", LinearRegression())])

# x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# model.fit(x_tr, y_tr)

# pred = model.predict(x_te)

# r2 = r2_score(y_te, pred)
# mae = mean_absolute_error(y_te, pred)
# mse = mean_squared_error(y_te, pred)
# rmse = np.sqrt(mse)
# print("R2 :", r2, "  mae :", mae, "  mse :", mse, "  rmse :", rmse)

# resid = y_te - pred

# plt.figure(figsize=(6, 4))
# plt.scatter(pred, resid, alpha=0.7)
# plt.axhline(0, linestyle="--")
# plt.xlabel("Predicted")
# plt.ylabel("Residual")
# plt.title("Residual VS Predicted")
# plt.tight_layout()
# plt.show()

# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model, x_tr, y_tr, cv=cv, scoring="r2")
# print("CV R2: ", cv_scores, "Mean: ", cv_scores.mean(), "Std: ", cv_scores.std())

# 비선형일것 같은 애들 2차항까지 만들기
# poly_features = ["displacement", "horsepower", "weight", "acceleration"]
# num_rest = [c for c in num_cols if c not in poly_features]

# poly_ct = ColumnTransformer(
#     transformers=[
#         (
#             "poly_num",
#             Pipeline(
#                 steps=[
#                     ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#                     ("scaler", StandardScaler()),
#                 ]
#             ),
#             poly_features,
#         ),
#         ("num_rest", StandardScaler(), num_rest),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#     ],
#     remainder="drop",
# )

# model = Pipeline(steps=[("prep", poly_ct), ("reg", LinearRegression())])

# model.fit(x_tr, y_tr)
# pred2 = model.predict(x_te)

# r2_poly = r2_score(y_te, pred2)
# mae_poly = mean_absolute_error(y_te, pred2)
# mse_poly = mean_squared_error(y_te, pred2)
# rmse_poly = np.sqrt(mse_poly)
# print(
#     "Poly R2 : ", r2_poly, "  mae : ", mae_poly, "  mse : ", mse_poly, "  rmse : ", rmse
# )
