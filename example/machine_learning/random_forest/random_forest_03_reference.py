# =========================================================
# Penguins 데이터 분류 파이프라인 (시각화 → RF+GridSearch → 평가 → 변수중요도)
# =========================================================
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")

plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 0) 데이터 로드 & 기본 정리
# -----------------------------
# df = sns.load_dataset("penguins").dropna().reset_index(drop=True)  # 결측 제거(간단히)
df = sns.load_dataset("penguins")

# 타켓/특징 분리
X = df.drop(columns=["species"])
Y = df["species"]  # 3-class: Adelie / Chinstrap / Gentoo

num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()


# 결측 대치(모델 학습용)
df[num_cols] = df[num_cols].apply(lambda s: s.fillna(s.median()))
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])


print(f"[INFO] 수치형 특성({len(num_cols)}): {num_cols}")
print(f"[INFO] 범주형 특성({len(cat_cols)}): {cat_cols}")

# =========================================================
# 1) 원시 데이터 시각화 (여러 시각)
# =========================================================
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x="species")
plt.title("[원시] 클래스 분포")
plt.tight_layout()
plt.show()

# 수치형 변수들의 히스토그램
df[num_cols].hist(bins=20, figsize=(10, 8))
plt.suptitle("[원시] 수치형 변수 히스토그램", y=1.02)
plt.tight_layout()
plt.show()

# 박스플롯(클래스별 분포 비교)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="species", y="body_mass_g", hue="species")
plt.xticks(rotation=30)
plt.title("[원시] 클래스별 수치형 박스플롯")
plt.tight_layout()
plt.show()

# 상관행렬(수치형만)
plt.figure(figsize=(7, 5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("[원시] 수치형 상관행렬")
plt.tight_layout()
plt.show()

# =========================================================
# 2) 전처리 파이프라인 + 모델 (RandomForest)
# =========================================================

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
    ],
    remainder="passthrough",
)

rf = RandomForestClassifier(random_state=42)

pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", rf),
    ]
)

# =========================================================
# 3) GridSearchCV 하이퍼파라미터 탐색
#   - 멀티 지표(scoring dict)로 평가, refit은 'f1_macro' 기준
# =========================================================
param_grid = [
    {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.8],  # 특성 수 대비 랜덤 후보 개수
    }
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",  # 각 클래스를 한 번씩 양성으로 간주하여 클래스별 F1을 모두 계산 → 산술평균(동일 가중치) 한 값
    "f1_weighted": "f1_weighted",  # 클래스 표본 수로 가중 평균
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1_macro",  # 최종 모델 선택 기준
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)


# df = sns.load_dataset("penguins")

# X = df.drop(columns=["species"])
# Y = df["species"] # 3-class: Adelie / Chinstrap / Gentoo

# X_tr, X_te, y_tr, y_te = train_test_split(
#     X, Y, test_size=0.2, stratify=Y, random_state=42
# )

# X_tr[num_cols] = X_tr[num_cols].apply(lambda s: s.fillna(s.median()))
# for c in cat_cols:
#     X_tr[c] = X_tr[c].fillna(X_tr[c].mode()[0])

# X_tr = pd.get_dummies(X_tr[cat_cols + num_cols], columns=cat_cols, drop_first=True)

# X_te[num_cols] = X_te[num_cols].apply(lambda s: s.fillna(s.median()))
# for c in cat_cols:
#     X_te[c] = X_te[c].fillna(X_te[c].mode()[0])

# X_te = pd.get_dummies(X_te[cat_cols + num_cols], columns=cat_cols, drop_first=True)


gs.fit(X_tr, y_tr)

print("\n[GridSearchCV] 최적 파라미터(=refit 기준 f1_macro):")
print(gs.best_params_)
print(f"[GridSearchCV] CV best f1_macro: {gs.best_score_:.4f}")

best_model = gs.best_estimator_

# =========================================================
# 4) 테스트 평가 지표 출력
# =========================================================
y_pred = best_model.predict(X_te)
y_proba = best_model.predict_proba(X_te)  # 멀티클래스 확률

acc = accuracy_score(y_te, y_pred)
f1_macro = f1_score(y_te, y_pred, average="macro")  # 클래스별 F1의 동일 가중 평균
f1_weighted = f1_score(y_te, y_pred, average="weighted")  # 클래스 표본 수로 가중 평균

print("\n[TEST] 분류 리포트")
print(classification_report(y_te, y_pred, digits=4))

print("[TEST] 요약 지표")
print(f"- Accuracy     : {acc:.4f}")
print(f"- F1 (macro)   : {f1_macro:.4f}")
print(f"- F1 (weighted): {f1_weighted:.4f}")

# 멀티클래스 ROC-AUC (One-vs-Rest)
classes = np.unique(Y)  # Y.unique()
auc_ovr = roc_auc_score(y_te, y_proba, multi_class="ovr")
auc_ovo = roc_auc_score(y_te, y_proba, multi_class="ovo")
print(f"- ROC-AUC (OVR, macro): {auc_ovr:.4f}")
print(f"- ROC-AUC (OVO, macro): {auc_ovo:.4f}")

# 혼동행렬 시각화
cm = confusion_matrix(y_te, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", values_format="d")
plt.title("[TEST] 혼동행렬")
plt.tight_layout()
plt.show()

# =========================================================
# 5) 변수 중요도 (MDI + Permutation Importance)
#   - (A) MDI: RandomForest의 feature_importances_ (전처리 후 특성 기준)
#   - (B) Permutation Importance: 테스트셋에서 f1_macro 기준
# =========================================================

# ----- (A) MDI (모델 내장 중요도)
# 전처리 후 피처 이름 추출
prep = best_model.named_steps["prep"]
feature_names = X_te.columns
print("\n\nfeature_names : ", feature_names)

rf_best = best_model.named_steps["model"]
mdi = pd.Series(rf_best.feature_importances_, index=feature_names).sort_values(
    ascending=False
)

TOPN = 10
plt.figure(figsize=(8, 7))
mdi.head(TOPN).iloc[::-1].plot(kind="barh")
plt.title(f"[MDI] 모델 내장 변수중요도 TOP{TOPN}")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\n[MDI] 상위 중요도 TOP10")
print(mdi.head(10).round(4))

X_te_tx = prep.transform(X_te)
# ----- (B) Permutation Importance (테스트셋, f1_macro)
# 파이프라인 전체에 대해 순열 중요도(전처리 포함)
perm = permutation_importance(
    rf_best, X_te_tx, y_te, scoring="f1_macro", n_repeats=30, random_state=42, n_jobs=-1
)

perm_mean = pd.Series(perm.importances_mean, index=feature_names).sort_values(
    ascending=False
)
perm_std = pd.Series(perm.importances_std, index=feature_names).loc[perm_mean.index]

plt.figure(figsize=(8, 7))
perm_mean.head(TOPN).iloc[::-1].plot(kind="barh", xerr=perm_std.head(TOPN).iloc[::-1])
plt.title(f"[Permutation Importance] (f1_macro, TEST) TOP{TOPN}")
plt.xlabel("Mean Δscore (± std)")
plt.tight_layout()
plt.show()

print("\n[Permutation Importance] 상위 중요도 TOP10 (f1_macro, TEST)")
out = pd.DataFrame(
    {
        "mean": perm_mean.head(10).round(4),
        "std": perm_std.head(10).round(4),
    }
)
print(out)
