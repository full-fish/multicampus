from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
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

# species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
df = sns.load_dataset("penguins")

# 수치형 컬럼과 범주형 컬럼 구분
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(exclude=["number"]).columns

# 수치형은 중앙값으로 결측치 채우기
for col in num_cols:
    df.fillna({col: df[col].median()}, inplace=True)

# 범주형 → 최빈값으로 결측치 채우기
for col in cat_cols:
    df.fillna({col: df[col].mode()[0]}, inplace=True)
print(df)

X = df.drop(columns=["species"])
X_encoded = pd.get_dummies(data=X, dtype="int", drop_first=False)
Y = df["species"]

# 원시데이터 시각화
fig, axes = plt.subplots(4, 6, figsize=(20, 12))
corr = df[num_cols].corr()

sns.countplot(data=df, x="species", ax=axes[0, 0])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1])

for index, col in enumerate(X.columns):
    sns.histplot(data=df, x=col, kde=True, ax=axes[1, index])
    axes[1, index].set_title(f"{col}의 분포")
for index, col in enumerate(X.columns):
    sns.boxplot(data=df, x="species", y=col, ax=axes[2, index])
    axes[2, index].set_title(f"{col}의 이상치 및 분포")


x_train, x_test, y_train, y_test = train_test_split(
    X_encoded, Y, test_size=0.2, random_state=42, stratify=Y
)

#  ColumnTransformer()  -수치형 컬럼 스케일링
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
            ),
        ),
    ]
)


# 교차 검증 객체 생성
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# gridsearchCV
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None, "sqrt", "log2"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc_ovr",  # 하이퍼파라미터 선택 기준 점수
    cv=cv,  # 검증 방식
    n_jobs=-1,  # 가능한 코어 활용(그리드 탐색에만 적용)
    refit=True,  # 최고 점수 모델로 자동 재학습
    return_train_score=True,
)
grid.fit(x_train, y_train)
print("Best Params : ", grid.best_params_)
print("Best CV ROC AUC : ", grid.best_score_)


best_estimator = grid.best_estimator_
best_model = best_estimator.named_steps["model"]

y_pred = best_estimator.predict(x_test)
y_proba = best_estimator.predict_proba(x_test)

# 다중분류 average="macro" 기본 값은 binary인데 이건 이진 분류에서만
acc = accuracy_score(y_test, y_pred)  # 정확도
prec = precision_score(y_test, y_pred, average="macro")  # 정밀도
rec = recall_score(y_test, y_pred, average="macro")  # 재현율
f1 = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print("Test accuracy:", acc)
print("Test ROC AUC (ovr):", auc)
print("Classification report:\n", classification_report(y_test, y_pred))

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

# MDI
importances = best_model.feature_importances_
feature_names = x_train.columns
mdi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
mdi_df.sort_values("importance", ascending=False, inplace=True)

# MDA
perm = permutation_importance(
    best_estimator,
    x_test,
    y_test,
    n_repeats=30,
    scoring="roc_auc_ovr",
    random_state=42,
    n_jobs=-1,
)
mda_df = pd.Series(perm.importances_mean, index=x_train.columns).sort_values(
    ascending=False
)

# 시각화
axes[3, 0].barh(mdi_df.head(10)["feature"][::-1], mdi_df.head(10)["importance"][::-1])
axes[3, 0].set_title("MDI (Mean Decrease in Impurity)")

axes[3, 1].barh(mda_df.head(10).index[::-1], mda_df.head(10).values[::-1])
axes[3, 1].set_title("MDA (Permutation Importance)")
for ax in axes.flat:  # 1차원으로 평탄화
    if not ax.has_data():  # 빈 거면 안뜨게
        ax.set_visible(False)
plt.tight_layout()
plt.show()
