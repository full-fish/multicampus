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


X = iris.data  # 특성 (꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)
Y = iris.target  # 타겟 (0: Setosa, 1: Versicolor, 2: Virginica)
print("=== 붓꽃 데이터셋 정보 ===")
print(f"데이터 개수: {len(X)}")
print(f"특성(features): {iris.feature_names}")
print(f"클래스(species): {iris.target_names}")
print()

# 훈련/테스트 데이터 분리 (80:20)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)
# 훈련/검증 데이터 분리(75:25)
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=42
)
# 데이터 정규화 (거리 기반 알고리즘이므로 스케일링 중요!)
scaler = StandardScaler().fit(x_tr)
x_tr_scaled = scaler.transform(x_tr)
x_val_scaled = scaler.transform(x_val)
# 검증을 통한 하이퍼 파라미터 찾기
best_cfg = None
best_acc = -1.0

for k in [3, 5, 7, 9, 11]:
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            model.fit(x_tr_scaled, y_tr)
            pred = model.predict(x_val_scaled)
            acc = accuracy_score(y_val, pred)
            print("k:", k, "acc", acc)
            if acc > best_acc:
                best_acc = acc
                best_cfg = {"k": k, "metric": metric, "weights": weights}
print(f"[검증] 최고 정확도={best_acc:.3f}, 설정={best_cfg}")
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_model = KNeighborsClassifier(
    n_neighbors=best_cfg["k"], metric=best_cfg["metric"], weights=best_cfg["weights"]
)

# 성능 평가
knn_model.fit(x_train_scaled, y_train)
pred = knn_model.predict(x_test_scaled)
last_acc = accuracy_score(y_test, pred)
print(f"[테스트] 정확도: {last_acc:.3f}")
print(classification_report(y_test, pred, digits=3))
# 혼동 행렬(Confusion Matrix) 시각화
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot()
cm = confusion_matrix(y_test, pred)
dfcm = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
sns.heatmap(dfcm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_ylabel("실제 값")
ax1.set_xlabel("예측 값")
plt.tight_layout()
plt.show()

# # 새로운 샘플 예측 예제
# print("\n=== 새로운 붓꽃 예측 ===")
# new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # 새로운 붓꽃 데이터
# new_flower_scaled = scaler.transform(new_flower)
# prediction = knn_model.predict(new_flower_scaled)
# probabilities = knn_model.predict_proba(new_flower_scaled)
# print(
#     f"입력 데이터: 꽃받침 길이={new_flower[0][0]}, 꽃받침 너비={new_flower[0][1]}, "
#     f"꽃잎 길이={new_flower[0][2]}, 꽃잎 너비={new_flower[0][3]}"
# )
# print(f"예측 결과: {iris.target_names[prediction[0]]}")
# print(f"각 클래스별 확률:")
# for i, prob in enumerate(probabilities[0]):
#     print(f"  {iris.target_names[i]}: {prob:.2%}")
# print(probabilities)
