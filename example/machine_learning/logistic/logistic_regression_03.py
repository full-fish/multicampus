from __future__ import division
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
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

df = sns.load_dataset("titanic")
# 결측치 제거. 결측치 있으면 스케일러가 못 다룸
df = df.dropna(subset=["age", "embarked", "sex", "class"])
print(df)
# 원핫 인코딩
df_encoded = pd.get_dummies(df, drop_first=True)
print("df_encoded", df_encoded)
#! "alive_yes"이거 가 survived랑 같은 데이터라 있으면 데이터누수 나서 정확도 1이 나옴
X = df_encoded.drop(columns=["survived", "alive_yes"])
Y = df_encoded["survived"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 객실 등급이 좋을 수록 살 가능성이 높음
sns.barplot(data=df, x="class", y="survived", estimator="mean", ax=axes[0, 0])
axes[0, 0].set_title("Class별 생존률")
axes[0, 0].set_ylabel("생존률")
axes[0, 0].set_xlabel("객실 등급")

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
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("acc : ", acc)
print("prec :", prec)
print("rec : ", rec)
print("f1 : ", f1)
print("auc : ", auc)

metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1,
    "ROC AUC": auc,
}
df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])

sns.barplot(data=df_metrics, x="Score", y="Metric", ax=axes[0, 1], palette="crest")
axes[0, 1].set_title("모델 성능 지표")
axes[0, 1].set_xlim(0, 1.05)

for i, v in enumerate(df_metrics["Score"]):
    axes[0, 1].text(v, i, f"{v:.3f}", va="center")


print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(
    pd.DataFrame(
        cm, index=["실제: 사망", "실제: 생존"], columns=["예측: 사망", "예측: 생존"]
    )
)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print(fpr)
print(tpr)
print(thresholds)
# AUC 값이 0.858이 나오는데 성능이 좋은 편임
# 생존자 1명과 사망자 1명을 뽑았을때 85% 확률로 생존자 예측
axes[1, 0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[1, 0].plot([1, 0], [1, 0], "k--", alpha=0.5)
axes[1, 0].set_title("ROC 곡선")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend(loc="lower right")

# 피처 중요도 분석
coefs = best_model.named_steps["clf"].coef_[0]  # 로지스틱 회귀 계수
features = X.columns

importance = pd.DataFrame({"feature": features, "coef": coefs})
importance["abs_coef"] = importance["coef"].abs()
importance = importance.sort_values("abs_coef", ascending=False)

print(importance.head(15))

# survived에 영향을 가장 많이 주는건 성별임을 알 수 있음
# 성인 남자 일수록 살 가능성이 덕음
sns.barplot(
    data=importance.head(15), x="coef", y="feature", palette="vlag", ax=axes[1, 1]
)
axes[1, 1].set_title("로지스틱 회귀 - 피처 중요도(상위 15개)")
axes[1, 1].axvline(0, color="gray", linestyle="--")
axes[1, 1].set_xlabel("계수 (coef)")
axes[1, 1].set_ylabel("")
plt.tight_layout()
plt.show()
