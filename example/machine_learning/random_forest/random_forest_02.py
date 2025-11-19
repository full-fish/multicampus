from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine

bc = load_breast_cancer()
X = bc.data
Y = bc.target

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(X)
print("--")
print(Y)

pipe = Pipeline(
    [
        (
            "model",
            RandomForestClassifier(
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "model__n_estimators": [300],
    # "model__n_estimators": list(range(100, 301, 100)),
    "model__max_depth": [None, 3, 5, 7, 10],
    "model__min_samples_leaf": list(range(1, 5)),
    "model__max_features": [None, "sqrt", "log2"],
}

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
print("최적 하이퍼파라미터:", grid.best_params_)
print("CV 최고 점수(roc_auc):", grid.best_score_)

best_estimator = grid.best_estimator_
best_model = best_estimator.named_steps["model"]
print("best_estimator", best_estimator)
print("best_model", best_model)

pred = best_estimator.predict(x_test)
print("proba", best_estimator.predict_proba(x_test))

proba = best_estimator.predict_proba(x_test)[:, 1]
print("OOB score :", best_model.oob_score_)
print("accuracy :", accuracy_score(y_test, pred))
print("ROC AUC :", roc_auc_score(y_test, proba))
print(
    "Classification Report:\n",
    classification_report(y_test, pred, target_names=bc.target_names),
)

# 7. 중요도 비교 (MDI vs permutation) -- RandomForest 안에 들어있는 실제 모델 꺼내기

imp_mdi = pd.Series(
    best_model.feature_importances_, index=bc.feature_names
).sort_values(ascending=False)

perm = permutation_importance(
    best_estimator,
    x_test,
    y_test,
    scoring="roc_auc",
    n_repeats=50,
    random_state=42,
    n_jobs=-1,
)
imp_perm_mean = pd.Series(perm.importances_mean, index=bc.feature_names)
imp_perm_std = pd.Series(perm.importances_std, index=bc.feature_names)

mdi_top10 = imp_mdi.head(10).iloc[::-1]
mda_top10 = imp_perm_mean.sort_values(ascending=False).head(10).iloc[::-1]
mda_std_top10 = imp_perm_std[mda_top10.index]

plt.figure(figsize=(8, 6))
plt.barh(mdi_top10.index, mdi_top10.values)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(mda_top10.index, mda_top10.values, xerr=mda_std_top10.values)
plt.tight_layout()
plt.show()



compare = pd.DataFrame(
    {
        "MDI importance": imp_mdi,
        "MDA importance": imp_perm_mean,
        "MDA importance std": imp_perm_std,
    }
).sort_values("MDA importance", ascending=False)

print(compare)
