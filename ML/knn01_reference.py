import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")

plt.rcParams["axes.unicode_minus"] = False

#
iris = load_iris()  # Bunch객체 리턴
# print(iris)
X = iris.data
Y = iris.target

# X,Y = load_iris(return_X_y=True, as_frame=True)
# print(X)
# print(Y)

print("==붓꽃 데이터셋 정보==")
print(f"데이터 개수: {len(X)}")
print(f"특성(features): {iris.feature_names}")
print(f"클래스(species): {iris.target_names}")

df = pd.DataFrame(X[:, [2, 3]], columns=iris.feature_names[2:4])
df["species"] = [iris.target_names[i] for i in Y]

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)

sns.scatterplot(
    data=df, x=iris.feature_names[2], y=iris.feature_names[3], hue="species", ax=ax1
)
ax1.set_xlabel("꽃잎 길이(cm)")
ax1.set_ylabel("꽃잎 너비(cm)")
ax1.set_title("붓꽃 데이터 분포")
ax1.grid(True)


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=33, stratify=Y
)


scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

k = 22
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train_scaled, y_train)


y_pred = knn.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"==KNN(k={k} 모델 성능)")
print(f"정확도(accuracy): {accuracy:.2f}")
print()
print("분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
dfcm = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

ax2 = fig.add_subplot(122)
sns.heatmap(data=dfcm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("예측 값")
ax2.set_ylabel("실제 값")
ax2.set_title("Confusion Matrix")

k_range = range(3, 31)
accuracies = []

for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(x_train_scaled, y_train)
    y_pred_tmp = knn_tmp.predict(x_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred_tmp))

ax3 = fig.add_subplot(133)
ax3.plot(k_range, accuracies, marker="o")
ax3.set_xlabel("K 값")
ax3.set_ylabel("정확도")
ax3.set_title("K 값에 따른 모델 정확도")
ax3.grid(True)
ax3.set_xticks(range(1, 31, 2))

plt.tight_layout()
plt.show()


print("\n==새로운 데이터 붓꽃 예측===")
new_flower = np.array([[2.1, 1.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)
pred = knn.predict(new_flower_scaled)
probabilites = knn.predict_proba(new_flower_scaled)
print(pred)
print(probabilites)
