import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 문서 데이터 정의
docs = [
    "배송이 빠르고 포장이 꼼꼼해서 만족합니다",
    "배송이 너무 느리고 포장도 엉망이에요",
    "가격은 저렴한데 품질이 좋아요",
    "가격이 비싼 편이지만 디자인이 예쁩니다",
    "배송도 빠르고 품질도 좋아서 또 구매하고 싶어요",
]

# 2. TF-IDF 벡터화 객체 설정
# TfidfVectorizer는 텍스트를 TF-IDF 기반의 실수 벡터로 변환합니다.
tfidf = TfidfVectorizer(
    # 전체 문서 중 80% 이상의 문서에 나타나는 단어는 무시 (너무 흔한 단어 제외)
    max_df=0.8,
    # 최소 1개 이상의 문서에 나타나는 단어만 포함
    min_df=1,
    # 토큰 패턴 정의: 한글 단어를 포함하여 문자를 토큰으로 인식
    token_pattern=r"(?u)\b\w+\b",
)

# 3. 문서-단어 행렬 X 생성
# fit: docs를 분석하여 단어장(Vocabulary) 구축 및 IDF 계산
# transform: 구축된 단어장과 IDF를 이용해 docs를 TF-IDF 벡터로 변환
X = tfidf.fit_transform(docs)

# X는 희소 행렬(Sparse Matrix)이며, shape는 (문서 수, 단어 수)입니다.
print("TF-IDF 행렬 크기:", X.shape)

# 4. 전체 문서 간 코사인 유사도 계산
# X와 X를 비교하여, 모든 문서 쌍 간의 유사도를 계산합니다.
# 결과 cos_sim은 (문서 수, 문서 수) 크기의 대칭 행렬이 됩니다.
cos_sim = cosine_similarity(X, X)

# 5. 결과를 DataFrame으로 만들어 보기 쉽게 출력
# pandas DataFrame을 사용하여 유사도 행렬을 시각화합니다.
cos_df = pd.DataFrame(cos_sim, columns=[f"d{i}" for i in range(len(docs))])

print("\n코사인 유사도 행렬:")
print(cos_df.round(2))

# 6. 기준 문서 설정 및 가장 유사한 문서 찾기
idx = 0  # 첫 번째 리뷰를 기준 문서(d0)로 설정합니다.
similarities = cos_sim[
    idx
]  # 코사인 유사도 행렬에서 0번째 행(d0과 다른 문서들의 유사도)을 가져옵니다.

# numpy를 사용하여 유사도 배열을 내림차순으로 정렬합니다.
# np.argsort(-similarities)를 통해 내림차순 인덱스를 얻습니다.
# [1]을 선택하는 이유: [0]은 자기 자신(idx)이므로 유사도가 1.0으로 당연히 가장 높습니다.
# 두 번째로 큰 값(가장 비슷한 다른 문서)의 인덱스를 찾습니다.
most_sim_idx = np.argsort(-similarities)[1]

print("\n[기준 문서]")
print(docs[idx])

print("\n[가장 비슷한 문서]")
print(docs[most_sim_idx])

print("\n유사도:", similarities[most_sim_idx])
