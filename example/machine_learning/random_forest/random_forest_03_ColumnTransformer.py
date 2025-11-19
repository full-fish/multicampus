from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.inspection import permutation_importance
import platform

# 한글 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터
df = sns.load_dataset("penguins")

# 2. 결측치 처리 (df 전체에서 먼저)
num_cols_all = df.select_dtypes(include=["number"]).columns
cat_cols_all = df.select_dtypes(exclude=["number"]).columns

for col in num_cols_all:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols_all:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 3. X, y 분리
X = df.drop(columns=["species"])
y = df["species"]

# 4. 숫자/범주형 다시 나누기 (이제 여기서 뽑은 걸 ColumnTransformer에 넣을 거임)
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

# 5. ColumnTransformer 정의
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# 6. 파이프라인 (전처리 → 모델)
pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)

# 7. 시각화 (여기는 네 원래 코드 스타일 유지해도 됨: 인코딩 전 X로 그린다)
fig, axes = plt.subplots(4, 6, figsize=(20, 12))
corr = df[num_cols].corr()

sns.countplot(data=df, x="species", ax=axes[0, 0])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1])

# 숫자형만 히스토그램, 박스플롯
for idx, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[1, idx])
    axes[1, idx].set_title(f"{col}의 분포")

for idx, col in enumerate(num_cols):
    sns.boxplot(data=df, x="species", y=col, ax=axes[2, idx])
    axes[2, idx].set_title(f"{col}의 이상치 및 분포")

# 8. train/test split은 원본 X로
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. 교차검증 객체
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 10. GridSearchCV (이제 estimator가 pipe임)
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None, "sqrt", "log2"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc_ovr",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)
grid.fit(x_train, y_train)

print("Best Params :", grid.best_params_)
print("Best CV ROC AUC :", grid.best_score_)

best_estimator = grid.best_estimator_
best_model = best_estimator.named_steps["model"]
best_preprocessor = best_estimator.named_steps["preprocessor"]

# 11. 예측
y_pred = best_estimator.predict(x_test)
y_proba = best_estimator.predict_proba(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print("Test accuracy:", acc)
print("Test ROC AUC (ovr):", auc)
print("Classification report:\n", classification_report(y_test, y_pred))

# 12. 성능 막대 그래프 (네 원래 코드 그대로)
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1,
    "ROC AUC": auc,
}
df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
sns.barplot(data=df_metrics, x="Score", y="Metric", ax=axes[3, 2], palette="crest")
axes[3, 2].set_title("모델 성능 지표")
axes[3, 2].set_xlim(0, 1.05)
for i, v in enumerate(df_metrics["Score"]):
    axes[3, 2].text(v, i, f"{v:.3f}", va="center")

# 13. MDI (RandomForest 내장 중요도) → 전처리된 컬럼 이름 필요
feat_names = best_preprocessor.get_feature_names_out()
importances = best_model.feature_importances_
mdi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
    "importance", ascending=False
)

axes[3, 0].barh(mdi_df.head(10)["feature"][::-1], mdi_df.head(10)["importance"][::-1])
axes[3, 0].set_title("MDI (Mean Decrease in Impurity)")

# 14. MDA (Permutation Importance) → 파이프라인에 원본 X 넣었으니 이름은 원본 X.columns로
perm = permutation_importance(
    best_estimator,
    x_test,
    y_test,
    n_repeats=30,
    scoring="roc_auc_ovr",
    random_state=42,
    n_jobs=-1,
)
mda_df = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

axes[3, 1].barh(mda_df.head(10).index[::-1], mda_df.head(10).values[::-1])
axes[3, 1].set_title("MDA (Permutation Importance)")

# 15. 빈 축 숨기기
for ax in axes.flat:
    if not ax.has_data():
        ax.set_visible(False)

plt.tight_layout()
plt.show()
