import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")

plt.rcParams["axes.unicode_minus"] = False

titanic = sns.load_dataset("titanic")

use_cols = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "alone",
]

df = titanic[use_cols].copy()
# print(df)
# ============================================원시데이터 시각화====================================================
# 클래스 불균형 확인
# plt.figure()
# sns.countplot(x="survived", data=df)
# plt.title("클래스 분포(0=사망, 1=생존)")
# plt.show()

# num_cols = ['pclass','age','sibsp', 'parch', 'fare']
# df_num = df[num_cols + ['survived']].copy()

# for c in num_cols:
#     df_num[c] = df_num[c].fillna(df_num[c].median())

# print(df_num.isnull().sum())

# plt.figure(figsize=(7,5))
# corr = df_num.corr(numeric_only=True)
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("수치형 피쳐 상관관계")
# plt.show()

# plt.figure(figsize=(9,5))
# sns.boxplot(data=df, x="survived", y='age', hue="pclass")
# plt.title("나이의 박스플롯(이상치 관찰)")
# plt.show()

# plt.figure()
# sns.histplot(df_num['age'], bins=30)
# plt.title("나이 분포")
# plt.show()

# ===================================================================================

cat_cols = ["sex", "embarked", "alone"]
num_cols = ["pclass", "age", "sibsp", "parch", "fare"]

df[num_cols] = df[num_cols].apply(lambda s: s.fillna(s.median()))
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
Y = df["survived"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
    ],
    remainder="passthrough",  # "drop" : 삭제
)

pipe = Pipeline(
    steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))]
)

param_grid = [
    {
        "model__solver": ["lbfgs", "saga"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1, 3, 10],
        "model__class_weight": [None, "balanced"],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid.fit(x_train, y_train)
best_model = grid.best_estimator_

print("Best Params : ", grid.best_params_)
print("CV Best ROC_AUC : ", grid.best_score_)

y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:, 1]

print("accuraucy : ", accuracy_score(y_test, y_pred))
print("roc_auc : ", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report : \n", classification_report(y_test, y_pred, digits=3))

RocCurveDisplay.from_estimator(best_model, x_test, y_test)
plt.title("ROC Curve")
plt.show()
