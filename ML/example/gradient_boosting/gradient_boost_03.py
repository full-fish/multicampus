# origin을 target
# cylinders을 as type해서 범주형으로 바꿔서(몇개 없기 때문에) df['name'] = df['name'].astype('category')

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
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

df = sns.load_dataset("mpg")
print(df)
print(df.info())
df["cylinders"] = df["cylinders"].astype("category")
print(df)
print(df.info())

X = df.drop(columns=["origin"])
y = df["origin"]

# 홀드아웃
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # origin이 타깃이라 클래스 비율 맞춤
)

# 전처리 파이프라인
num_selector = selector(dtype_include=["int64", "float64"])
cat_selector = selector(dtype_include=["object", "category"])

num_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    [
        ("num", num_pipe, num_selector),
        ("cat", cat_pipe, cat_selector),
    ],
    remainder="drop",
)

# 전체 파이프라인
pipe = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "model",
            HistGradientBoostingClassifier(
                early_stopping=False,  # 튜닝 중엔 끄는 게 깔끔
                random_state=42,
            ),
        ),
    ]
)

# 그리드 정의
param_grid = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__max_depth": [None, 3, 5],
    "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 이중 / 다중 스코어 기록, 최종 refit은 ROC-AUC
scoring = {}
print("y_tr.nunique()", y_tr.nunique())  # 3

if y_tr.nunique() == 2:
    # 이진분류
    scoring = {
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
    }
    refit_metric = "roc_auc"
else:
    # 다중분류
    scoring = {
        "roc_auc_ovr": "roc_auc_ovr",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
    }

# origin은 다중분류라서 roc_auc는 ovr로
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit="roc_auc_ovr",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

# 학습
search.fit(X_tr, y_tr)

print("Best params:", search.best_params_)
print("Best CV roc_auc_ovr:", search.best_score_)

# 테스트 평가
best_est = search.best_estimator_
best_model = best_est.named_steps["model"]
y_proba = best_est.predict_proba(X_te)
y_pred = best_est.predict(X_te)

print("\n[Test]")

acc = accuracy_score(y_te, y_pred)  # 정확도
prec = precision_score(y_te, y_pred, average="macro")  # 정밀도
rec = recall_score(y_te, y_pred, average="macro")  # 재현율
f1 = f1_score(y_te, y_pred, average="macro")
auc = roc_auc_score(y_te, y_proba, multi_class="ovr")

print("Accuracy: ", acc)
print("prec", prec)
print("rec", rec)
print("F1 (macro): ", f1)
print("ROC-AUC (ovr): ", auc)
print("\nClassification report\n", classification_report(y_te, y_pred, digits=3))

metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1,
    "ROC-AUC": auc,
}

# DataFrame 변환
df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])

# 모델 성능 지표 그래프
plt.figure(figsize=(6, 4))
sns.barplot(data=df_metrics, x="Score", y="Metric", palette="crest")
plt.title("모델 성능 지표")
plt.xlim(0, 1.05)
plt.xlabel("Score")
plt.ylabel("Metric")
plt.tight_layout()
plt.show()

# 전처리 후 피처 이름 가져오기
preprocess_fitted = best_est.named_steps["preprocess"]
feature_names = []

for name, trans, cols in preprocess_fitted.transformers_:
    if name == "remainder":
        continue
    if cols is None or len(cols) == 0:
        continue
    if hasattr(trans, "named_steps"):
        last_step = list(trans.named_steps.values())[-1]
        if hasattr(last_step, "get_feature_names_out"):
            feature_names.extend(last_step.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    else:
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

feature_names = np.array(feature_names)

# permutation importance
X_te_transformed = preprocess_fitted.transform(X_te)
perm = permutation_importance(
    best_model,
    X_te_transformed,
    y_te,
    scoring="roc_auc_ovr",
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

imp_mean = perm.importances_mean
imp_std = perm.importances_std

# 시리즈로 정리
mda_df = pd.Series(perm.importances_mean, index=feature_names).sort_values(
    ascending=False
)

print("\n[Permutation importance top 10]")
print(mda_df.head(10))

# 순열 중요도
plt.figure(figsize=(7, 5))
top_n = 10
plt.barh(
    mda_df.head(top_n).index[::-1],
    mda_df.head(top_n).values[::-1],
)
plt.title("MDA (Permutation Importance) - mpg")
plt.xlabel("Importance (roc_auc_ovr 감소)")
plt.tight_layout()
plt.show()

# 성능 지표에서는 Recall값이 0.72로 약간 놓치는 양성이 있음
# displacement가 0.38로 지대한 영향을 미치고 그 다음은 horsepower인데 0.04로 큰 영향을 미치지 않음
