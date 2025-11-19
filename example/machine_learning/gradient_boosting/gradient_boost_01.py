from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

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

# 데이터 로드
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

num_pipe = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ],
)

preprocess = ColumnTransformer(
    [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
)

pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("clf", HistGradientBoostingClassifier(random_state=42, early_stopping=False)),
    ]
)

# 홀드아웃 분할(검증 세트 포함: early_stopping용)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 모델 정의
# clf = HistGradientBoostingClassifier(
#     learning_rate=0.08,  # 작게 → 과적합 완화, 더 많은 트리 필요 [0.03, 0.05, 0.08, 0.1, 0.15]
#     max_leaf_nodes=31,  # 트리 복잡도(비슷한 역할: max_depth)
#     l2_regularization=0.0, #[0.0, 0.01, 0,1, 1.0]
#     early_stopping=True,
#     random_state=42,
# )

# clf = HistGradientBoostingClassifier(
#     random_state=42,
#     early_stopping=False,  # 튜닝할 땐 보통 끔
# )

param_grid = {
    "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__max_depth": [None, 3, 5],
    "clf__l2_regularization": [0.0, 0.01, 0.1, 1.0],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# 학습
grid.fit(X_tr, y_tr)

print("베스트 ROC-AUC (CV 평균):", grid.best_score_)
print("베스트 하이퍼파라미터:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

# 평가
best_estimator = grid.best_estimator_
best_model = best_estimator.named_steps["clf"]

#! permutation importance 출력 할 때 원핫 인코딩된 피쳐들은 길이가 달라 질 수 있으니 전처리에서 feature name을 가져와야함
preprocess = best_estimator.named_steps["preprocess"]
feature_names = []

for name, trans, cols in preprocess.transformers_:
    # 남은 컬럼(remainder)은 스킵
    if name == "remainder":
        continue

    # 컬럼이 아예 없으면(이번 케이스처럼 범주형 없음) 스킵
    if cols is None or len(cols) == 0:
        continue

    # 파이프라인으로 묶여 있는 경우 (num_pipe, cat_pipe)
    if hasattr(trans, "named_steps"):
        last_step = list(trans.named_steps.values())[-1]

        # 마지막 단계가 OHE처럼 이름을 만들어주는 경우
        if hasattr(last_step, "get_feature_names_out"):
            feature_names.extend(last_step.get_feature_names_out(cols))
        else:
            # StandardScaler처럼 이름 안 만들어주면 원래 컬럼명 그대로
            feature_names.extend(cols)
    else:
        # 파이프라인이 아닌 단일 변환기인 경우
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

feature_names = np.array(feature_names)
#! --

y_pred = best_estimator.predict(X_te)
y_proba = best_estimator.predict_proba(X_te)[:, 1]
print("Accuracy:", accuracy_score(y_te, y_pred))
print("ROC-AUC :", roc_auc_score(y_te, y_proba))
print()
print(classification_report(y_te, y_pred, digits=3))

acc = accuracy_score(y_te, y_pred)  # 정확도
prec = precision_score(y_te, y_pred, average="macro")  # 정밀도
rec = recall_score(y_te, y_pred, average="macro")  # 재현율
f1 = f1_score(y_te, y_pred, average="macro")
auc = roc_auc_score(y_te, y_proba, multi_class="ovr")

print("acc", acc)
print("prec", prec)
print("rec", rec)
print("f1", f1)
print("auc", auc)
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1,
    "ROC-AUC": auc,
}

# DataFrame 변환
df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])

# 그래프
plt.figure(figsize=(6, 4))
sns.barplot(data=df_metrics, x="Score", y="Metric", palette="crest")

plt.title("모델 성능 지표")
plt.xlim(0, 1.05)
plt.xlabel("Score")
plt.ylabel("Metric")
plt.tight_layout()
plt.show()

#! MDI는 HistGradientBoostingClassifier에서 미지원. 트리 기반이지만 히스토그램 기반이라 그럼

# 6) 피처 중요도(순열 중요도: 검증/테스트 기반 권장)
perm = permutation_importance(
    best_estimator,
    X_te,
    y_te,
    scoring="roc_auc",
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)
importances_mean = perm.importances_mean
importances_std = perm.importances_std

# ? 아래 두개 그래프 같은거임 후자가 좀더 커스텀이 됨
mda_df = pd.Series(importances_mean, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
top_n = 10
plt.barh(
    mda_df.head(top_n).index[::-1],  # 위에서 10개 뽑고 뒤집어서 위가 가장 중요하게
    mda_df.head(top_n).values[::-1],
)
plt.title("MDA (Permutation Importance)")
plt.xlabel("Importance (ROC-AUC 감소량)")
plt.tight_layout()
plt.show()

# ?
# 중요도 상위 10개 출력
rank = np.argsort(-importances_mean)
for idx in rank[:10]:
    print(
        f"{feature_names[idx]:25s}  mean={importances_mean[idx]:.4f}  std={importances_std[idx]:.4f}"
    )
top_n = 10
top_idx = rank[:top_n]

# 중요도 데이터 추출
top_features = feature_names[top_idx]
top_importances = importances_mean[top_idx]
top_std = importances_std[top_idx]

# 그래프 시각화
plt.figure(figsize=(8, 5))
plt.barh(range(top_n), top_importances, xerr=top_std, align="center")
plt.yticks(range(top_n), top_features)
plt.gca().invert_yaxis()  # 높은 중요도가 위로 오도록
plt.xlabel("Permutation Importance (ROC-AUC 감소량)")
plt.title("상위 10개 피처 중요도")
plt.tight_layout()
plt.show()
