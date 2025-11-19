import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
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
print("121212")
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

param_grid = {"clf__C": [0.01, 0.1, 0.3, 1, 3, 10]}
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",  # 하이퍼파라미터 선택 기준 점수
    cv=cv,  # 검증 방식
    n_jobs=-1,  # 가능한 코어 활용(그리드 탐색에만 적용)
    refit=True,  # 최고 점수 모델로 자동 재학습
    return_train_score=True,
)

grid.fit(x_train, y_train)


print("Best Params : ", grid.best_params_)
print("Best CV ROC AUC : ", grid.best_score_)
cvres = (
    pd.DataFrame(grid.cv_results_)
    .loc[:, ["params", "mean_test_score", "mean_train_score"]]
    .sort_values("mean_test_score", ascending=False)
)
print(cvres.head(10))

best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:, 1]

acc = accuracy_score(y_test, y_pred)  # 정확도
prec = precision_score(y_test, y_pred)  # 정밀도
rec = recall_score(y_test, y_pred)  # 재현율
auc = roc_auc_score(y_test, y_pred_proba)

print("acc : ", acc)
print("prec :", prec)
print("rec : ", rec)
print("auc : ", auc)

print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(pd.DataFrame(cm, index=["악성", "양성"], columns=["악성", "양성"]))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print(fpr)
print(tpr)
print(thresholds)

plt.plot(fpr, tpr, label=f"AUC = {auc: 3f}")
plt.tight_layout()
plt.show()
