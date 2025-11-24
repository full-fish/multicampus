# TF-IDF로 문서를 벡터화한 뒤, 코사인 유사도로 가장 비슷한 문서를 찾는 예제

from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도 계산
import numpy as np  # argmax 등 수치 연산

# 비교할 문서(코퍼스) 리스트
docs = [
    "오늘 날씨가 좋아서 산책을 했다",
    "오늘은 비가 와서 우울하다",
    "점심에 맛있는 파스타를 먹었다",
    "저녁에 산책하면서 음악을 들었다",  # 따옴표 통일 (직접 입력 시 스마트쿼트 주의)
]

# TF-IDF 벡터라이저 생성 (기본 설정: 공백 기반 토큰화)
vectorizer = TfidfVectorizer()
# print("vectorizer", vectorizer.get_feature_names_out())

# 문서 코퍼스를 학습(fit)하고, TF-IDF 희소행렬로 변환(transform)
X = vectorizer.fit_transform(docs)  # 형태: (문서 수, 단어 수)
print("vectorizer", vectorizer.get_feature_names_out())

# 검색 쿼리(비교 대상) 문장
query = ["산책을 하면서 날씨를 즐겼다"]

# 코퍼스에서 학습한 같은 단어 사전으로 쿼리를 TF-IDF 벡터로 변환
X_query = vectorizer.transform(query)  # 형태: (1, 단어 수)

# 쿼리와 각 문서 간 코사인 유사도 계산 → 결과 형태: (1, 문서 수)
sim = cosine_similarity(X_query, X)

# 유사도 행렬 출력 (한 줄짜리 벡터)
print("similarity:", sim)

# 가장 유사도가 높은 문서의 인덱스 선택
most_sim_idx = np.argmax(sim[0])

# 원 쿼리 문장 출력
print("Query:", query[0])

# 가장 유사한 문서 출력
print("Most similar doc:", docs[most_sim_idx])
