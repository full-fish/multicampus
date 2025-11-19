import numpy as np
import matplotlib.pyplot as plt
from pandas.io.sql import com
from scipy.integrate._ivp.radau import P
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

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

wine = load_wine()
X, Y = wine.data, wine.target
print(
    wine.feature_names
)  # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
print(wine.target_names)  # ['class_0' 'class_1' 'class_2']

# !----------------------------------------
df = pd.DataFrame(X, columns=wine.feature_names)
df["target"] = Y
df["target_name"] = df["target"].map(dict(enumerate(wine.target_names)))

# 데이터 불균형 확인 시각화
# df["target"].value_counts().sort_index().plot(kind="bar")
# plt.xticks([0, 1, 2], labels=wine.target_names)
# plt.show()


# sns.countplot(data=df, x="target_name")
# plt.show()

# 단변량 분포
# feat = "alcohol"
# sns.histplot(data=df, x=feat, hue="target_name", bins=20)

# 분포 및 이상치 확인
# feat = "alcohol"
# sns.boxplot(data=df, x="target_name", y=feat)
# plt.show()

# 두 특징간의 관계 및 분포
# xfeat, yfeat = "alcohol", "color_intensity"
# sns.scatterplot(data=df, x=xfeat, y=yfeat, hue="target_name")

# 모든 특징간의 상관관계 분포
corr = df.drop(columns=["target", "target_name"]).corr()
sns.heatmap(corr, cmap="coolwarm")
plt.show()

# !----------------------------------------


# df = pd.DataFrame(X[:, [1, 3]], columns=["malic_acid", "alcalinity_of_ash"])
# df = pd.DataFrame(X, columns=wine.feature_names)
# df["class"] = [wine.target_names[i] for i in Y]
# print(df)

# # 3️⃣ 필요한 컬럼만 선택
# df_selected = df[["malic_acid", "alcalinity_of_ash", "class"]]

# # 4️⃣ 산점도 그리기
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=df_selected, x="malic_acid", y="alcalinity_of_ash", hue="class", palette="Set2"
# )

# plt.title("와인 데이터: 말산 vs 회분 알칼리도")
# plt.xlabel("말산 (malic_acid)")
# plt.ylabel("회분 알칼리도 (alcalinity_of_ash)")
# plt.grid(True)
# plt.show()
# corr = df["malic_acid"].corr(df["alcalinity_of_ash"])
# print("malic_acid vs alcalinity_of_ash 상관계수:", corr)  # 0.29
np.divide
corr = df.corr(numeric_only=True)

abs_corr = corr.abs()

# (2) 자기 자신(1.0)은 제외하고 최대값 찾기
# 각 변수별로 가장 높은 상관을 가지는 다른 변수 찾기
max_corr = abs_corr.apply(lambda x: x.drop(x.name).idxmax())
max_corr_value = abs_corr.apply(lambda x: x.drop(x.name).max())

# 결과 정리
result = pd.DataFrame(
    {"가장 연관된 변수": max_corr, "상관계수(절댓값)": max_corr_value}
)

print(result.sort_values(by="상관계수(절댓값)", ascending=False).head(10))
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=20, stratify=Y
)
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=20, stratify=y_train
)
scaler = StandardScaler().fit(x_tr)
x_tr_scaled = scaler.transform(x_tr)
x_val_scaled = scaler.transform(x_val)
best_cfg = None
best_acc = -1.0
for k in range(3, 101, 2):
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
            model.fit(x_tr_scaled, y_tr)
            pred = model.predict(x_val_scaled)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best_cfg = {"k": k, "metric": metric, "weights": weights}

print(f"[검증] 최고 정확도 = {best_acc: .3f} 설정 = {best_cfg}")

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train_scaled, y_train)

# 모델 만들었으니 테스트
y_pred = knn.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cm = confusion_matrix(y_test, y_pred)
dfcm = pd.DataFrame(cm, index=wine.target_names, columns=wine.target_names)
sns.heatmap(data=dfcm, annot=True, fmt="d", cmap="Blues")
plt.show()
