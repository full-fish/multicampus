import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

data = load_breast_cancer(as_frame=True)
print(data.frame)
X = data.frame.drop(columns=["target"])
Y = data.frame["target"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="roc_auc")
cv_acc = cross_val_score(pipe, x_train, y_train, scoring="accuracy")
# print("cv_auc", f"{cv_auc.mean():.3f}")
# print("cv_acc", f"{cv_acc.mean():.3f}")

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
# predict_proba는 예측 결과를 클래스별 확률값 배열로 반환합니다.
# 이진 분류에서는 두 컬럼(클래스 0에 대한 확률, 클래스 1에 대한 확률)이 반환됩니다.
# [:, 1]은 각 샘플에서 클래스 1(양성 클래스, 즉 '암 있음')에 대한 확률만 추출하는 코드입니다.
# ROC-AUC 등 양성 클래스 기준의 평가에 이 확률값이 필요해서 아래와 같이 사용합니다.
y_pred_proba = pipe.predict_proba(x_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("acc : ", acc)
print("prec :", prec)
print("rec : ", rec)
print("auc : ", auc)

print(classification_report(y_test, y_pred, digits=3))
