# 필요한 라이브러리를 임포트합니다.
# 1. load_digits: scikit-learn에 내장된 손글씨 숫자 데이터셋을 로드하는 함수
# 2. TSNE: 비선형 차원 축소 알고리즘인 t-SNE 모델
# 3. matplotlib.pyplot: 시각화를 위한 라이브러리 (보통 plt로 줄여서 사용)
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1) 데이터 로드 및 준비
# --------------------
# 손글씨 숫자 데이터 (0~9)를 로드합니다. 각 이미지는 8x8 픽셀(64차원)입니다.
digits = load_digits()

print("\n\ndigis\n", digits)

# 원본 고차원 데이터 (64차원)
X = digits.data
# 데이터의 실제 라벨 (0부터 9까지의 숫자) -> 시각화 시 색상 지정에 사용됨
y = digits.target

# 2) t-SNE를 이용한 차원 축소
# --------------------
# TSNE 모델 객체를 생성하고 하이퍼 파라미터를 설정합니다.
tsne = TSNE(
    n_components=2,  # 축소할 차원 수: 2차원으로 설정하여 2D 평면에 시각화
    perplexity=30,  # 이웃의 개수 설정 (5~50 권장): 로컬 구조를 보존하며 변환
    learning_rate=200,  # 학습률: 최적화 단계에서 이동 폭을 결정
    random_state=42,  # 결과 재현성을 위한 시드 고정
)

# 64차원 데이터 X를 2차원으로 변환합니다. (t-SNE 학습 및 변환 동시 수행)
X_2d = tsne.fit_transform(X)

# 3) 결과 시각화
# --------------------
# 그래프 크기 설정
plt.figure(figsize=(8, 6))

# 산점도(Scatter Plot)를 이용하여 변환된 데이터를 시각화합니다.
# X_2d[:, 0] : x축 좌표 (첫 번째 t-SNE 차원)
# X_2d[:, 1] : y축 좌표 (두 번째 t-SNE 차원)
# c=y : 각 점의 색상을 라벨(y, 즉 실제 숫자)에 따라 다르게 지정하여 클러스터링을 쉽게 확인
# s=10 : 점의 크기 설정
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=10, cmap="viridis")

# 그래프 제목 설정
plt.title("t-SNE visualization of digits (64D to 2D)")

# 범례(Legend) 추가: 어떤 색이 어떤 숫자를 의미하는지 표시
plt.colorbar(scatter, ticks=range(10), label="Digit Label")

# 그래프 출력
plt.show()
