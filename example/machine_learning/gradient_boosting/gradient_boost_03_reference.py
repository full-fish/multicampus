# =============================
# 0) 라이브러리
# =============================
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
import platform

if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42

# =============================
# 1) 데이터 로드 & 기본 전처리
# =============================
df = sns.load_dataset("mpg")

# 고카디널리티 텍스트 name은 제거
df = df.drop(columns=["name"], errors="ignore")

df['cylinders'] = df['cylinders'].astype('category')

num_selector = selector(dtype_include=["int64","float64"])
cat_selector = selector(dtype_include=["object","category"])


# 결측 대치(모델 학습용)
df[num_selector] = df[num_selector].apply(lambda s: s.fillna(s.median()))
for c in df[cat_selector].columns:
    df[c] = df[c].fillna(df[c].mode()[0])
    

# =============================
# 2) 초기 시각화
# =============================
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.countplot(data=df, x="origin")
plt.title("Target 분포 (origin)")

plt.subplot(1,3,2)
sns.boxplot(data=df, x="origin", y="mpg")
plt.title("origin vs mpg")

plt.subplot(1,3,3)
sns.scatterplot(data=df, x="weight", y="mpg", hue="origin")
plt.title("weight vs mpg (by origin)")
plt.tight_layout()
plt.show()

# =============================
# 3) 피처/타깃 분리
# =============================
y = df["origin"]
X = df.drop(columns=["origin"])

# =============================
# 4) 전처리 파이프라인 (SimpleImputer)
# =============================
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # ("scaler", StandardScaler())
])
categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # 최신 sklearn에선 sparse_output=False 권장(밀집 행렬 반환)
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(transformers=[
    ("num", numeric_tf, num_selector),
    ("cat", categorical_tf, cat_selector)
])

# =============================
# 5) 모델 & 전체 파이프라인
# =============================
gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", gb)
])

# =============================
# 6) 하이퍼파라미터 탐색 (GridSearchCV)
# =============================
# 권장 탐색: max_depth = {2, 3, 4, 5} → 잎 ≈ {4, 8, 16, 32}
param_grid = {   
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],  # 가장 중요한 튜닝 하이퍼 파라미터       
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc_ovr",      # 다중분류 AUC(One-vs-Rest)
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# =============================
# 7) 학습/평가
# =============================
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
grid.fit(X_tr, y_tr)

best = grid.best_estimator_
print("Best params:", grid.best_params_)
print("CV best AUC(ovr):", round(grid.best_score_, 4))

# 테스트 성능
y_pred  = best.predict(X_te)
y_proba = best.predict_proba(X_te)  # shape: (n_samples, n_classes)
# 멀티클래스 AUC(ovr, macro)
test_auc = roc_auc_score(y_te, y_proba, multi_class="ovr")
print("\nTest AUC(ovr, macro):", round(test_auc, 4))
print(classification_report(y_te, y_pred, digits=3))


# =============================
# 8) 변수 중요도 (MDA: 순열 중요도, 원본 컬럼 단위 집계)
# =============================
perm = permutation_importance(
    best, X_te, y_te,
    scoring="roc_auc_ovr",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_mean = pd.Series(perm.importances_mean, index=X_te.columns)

imp_mean = perm.importances_mean
imp_std  = perm.importances_std
rank = np.argsort(-imp_mean)

print("\n[Permutation Importance - Top 10]")
for idx in rank[:10]:
    print(f"{X.columns[idx]:25s} mean={imp_mean[idx]:.4f}  std={imp_std[idx]:.4f}")