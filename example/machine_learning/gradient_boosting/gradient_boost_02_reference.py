# ==============================================
# 실습 1 (업그레이드): 유방암 데이터 + 하이퍼파라미터 튜닝
# - 모델: HistGradientBoostingClassifier
# - 튜닝: GridSearchCV (ROC-AUC 기준 refit, 다중 스코어 기록)
# - 포인트: early_stopping=True + validation_fraction(내부 검증)
# ==============================================
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

# 1) 데이터 로드
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 2) 홀드아웃 분할(테스트 고정)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) 베이스라인 (참조용)
base = HistGradientBoostingClassifier(
    early_stopping=True,  # 폴드 내부에서 조기 종료
    validation_fraction=0.1,
    n_iter_no_change=10,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
)


base.fit(X_tr, y_tr)
proba_base = base.predict_proba(X_te)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)  # 임계값 튜닝

print("[Baseline]")
print("Accuracy :", accuracy_score(y_te, pred_base))
print("F1       :", f1_score(y_te, pred_base))
print("ROC-AUC  :", roc_auc_score(y_te, proba_base))
print()

num_selector = selector(dtype_include=["int64", "float64"])
cat_selector = selector(dtype_include=["object", "category"])

num_pipe = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    [
        ("num", num_pipe, num_selector),
        ("cat", cat_pipe, cat_selector),
    ],
    remainder="drop",
)

pipe = Pipeline(
    [
        # ("preprocess", preprocess),
        (
            "model",
            HistGradientBoostingClassifier(
                early_stopping=True,  # 폴드 내부에서 조기 종료
                validation_fraction=0.1,  # 기본값
                n_iter_no_change=10,  # 기본값
                random_state=42,
            ),
        )
    ]
)

# 4) 그리드 정의
param_grid = [
    {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        # 필요 시 추가: "model__min_samples_leaf": [10, 20, 30]
    }
]

# 5) 교차검증 설정 (층화 + 셔플)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6) 다중 스코어 기록, 최종 refit은 ROC-AUC
scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "accuracy": "accuracy",
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit="roc_auc",  # 최종 선택 기준
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
)

# 7) 학습(튜닝)
search.fit(X_tr, y_tr)


# 8) CV 결과 표 정리
def tidy_cv_results(gscv):
    df = pd.DataFrame(gscv.cv_results_)
    keep = [
        "param_model__learning_rate",
        "param_model__max_leaf_nodes",
        "param_model__l2_regularization",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_accuracy",
        "std_test_accuracy",
        "rank_test_roc_auc",  # GridSearchCV의 cv_results_ 테이블에서, 각 하이퍼파라미터 조합의 mean_test_roc_auc가 몇 위인지를 나타내는 순위 컬럼
    ]
    return df[keep].sort_values("rank_test_roc_auc").reset_index(drop=True)
    # 데이터프레임의 인덱스를 0부터 다시 매기고, 기존 인덱스는 버린다
    # reset_index()만 쓰면: 기존 인덱스가 새 컬럼으로 들어가고, 인덱스는 0..n-1로 재설정


cv_table = tidy_cv_results(search)
print("[Top 10 CV rows by ROC-AUC]")
print(cv_table.head(10).to_string(index=False))
print()

print("[Best Params by ROC-AUC]")
print(search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)
print()

# 9) 테스트셋 평가(베스트 모델)
best = search.best_estimator_
y_proba = best.predict_proba(X_te)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("[Best Model on Test]")
print("Accuracy :", accuracy_score(y_te, y_pred))
print("F1       :", f1_score(y_te, y_pred))
print("ROC-AUC  :", roc_auc_score(y_te, y_proba))
print("\nClassification Report\n", classification_report(y_te, y_pred, digits=3))


# 10) 순열 중요도(테스트셋, ROC-AUC 기준)
perm = permutation_importance(
    best, X_te, y_te, scoring="roc_auc", n_repeats=30, random_state=42, n_jobs=-1
)

imp_mean = perm.importances_mean
imp_std = perm.importances_std
rank = np.argsort(-imp_mean)

print("\n[Permutation Importance - Top 10]")
for idx in rank[:10]:
    print(f"{X.columns[idx]:25s} mean={imp_mean[idx]:.4f}  std={imp_std[idx]:.4f}")
