import numpy as np
import matplotlib.pyplot as plt
from pandas.io.sql import com
from scipy.integrate._ivp.radau import P
from sklearn.datasets import load_iris
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


# 1. 데이터 로드
iris = load_iris()  # Bunch 객체

X, Y = iris.data, iris.target
X_2, Y_2 = load_iris(return_X_y=True, as_frame=True)
print(X_2)
print("--------------------------------")
print(Y_2)
print("--------------------------------")
print("--------------------------------")
# 위의 2줄이 같은 값이긴 한데 형태가 다름 처음껀 리스트 형식 밑에는 데이터프레임 형식
print(X)
print("--------------------------------")
print(Y)
print("==붓꽃 데이터셋 정보==")
print(f"데이터 개수: ,{len(X)}")
print(f"특성(features): {iris.feature_names}")
print(f"클래스(species): {iris.target_names}")
df = pd.DataFrame(X[:, [2, 3]], columns=iris.feature_names[2:4])
df["species"] = [iris.target_names[i] for i in Y]

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)

sns.scatterplot(
    data=df, x=iris.feature_names[2], y=iris.feature_names[3], hue="species", ax=ax1
)
ax1.set_xlabel("꽃받침 길이(cm)")
ax1.set_ylabel("꽃받침 너비(cm)")
ax1.set_title("붗꽃 데이터 분포")
ax1.grid(True)


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=33, stratify=Y
)  # 훈련/테스트 데이터 분리 (80:20), stratify=Y 는 데이터 분포를 유지하기 위해 사용
print(df)
print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

scaler = StandardScaler().fit(x_train)
print("scaler", scaler)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("x_train_scaled", x_train_scaled)
print("x_test_scaled", x_test_scaled)

k = 33
knn = KNeighborsClassifier(n_neighbors=k)
print("knn", knn)
knn.fit(x_train_scaled, y_train)
print("knn", knn)

y_pred = knn.predict(x_test_scaled)
print("y_pred", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"==KNN(k={k} 모델 성능)")
print(f"정확도(accuracy): {accuracy: .2f}")
print("분류 리포트: ")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(y_pred)
print(y_test)

print("----혼동행렬-----")
cm = confusion_matrix(y_test, y_pred)
dfcm = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

ax2 = fig.add_subplot(122)
sns.heatmap(data=dfcm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("예측값")
ax2.set_ylabel("실제값")
ax2.set_title("Confusion Matrix")


k_range = range(3, 51)
accuracies = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(x_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(x_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

ax3 = fig.add_subplot(133)
ax3.plot(k_range, accuracies, marker="o")
ax3.set_xlabel("k 값")
ax3.set_ylabel("정확도")
ax3.set_title("k값에 따른 분류")
ax3.grid(True)
ax3.set_xticks(range(1, 31, 2))
print(accuracies)
plt.tight_layout()
plt.show()

print("----------새로운 데이터 붓꽃 예측-----------------")
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)

pred = knn.predict(new_flower_scaled)
probabilites = knn.predict_proba(new_flower_scaled)
print(pred)  # 0이라고 나오는것의 이름은 새로운 꽃 저 크기라면 setosa이다
print(probabilites)  # setosa가 될 확률은 1, versicolor와 virginica이건 0
