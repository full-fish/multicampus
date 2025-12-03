from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# --- 1. 데이터 준비 및 설정 ---
# 1) 예제 리뷰 데이터
docs = [
    "배송이 빠르고 포장이 꼼꼼해서 만족합니다",
    "배송이 너무 느리고 포장도 엉망이에요",
    "가격은 저렴한데 품질이 좋아요",
    "가격이 비싼 편이지만 디자인이 예쁩니다",
    "배송도 빠르고 품질도 좋아서 또 구매하고 싶어요",
]
# 2) TF-IDF 벡터화 객체 초기화
# max_df=0.8: 전체 문서의 80% 이상에 나타나는 단어는 제거 (너무 흔한 단어)
# min_df=1: 최소 1개 문서에 나타나는 단어만 사용
# token_pattern: 한글 포함 모든 단어(토큰)를 인식하도록 설정
tfidf = TfidfVectorizer(
    max_df=0.8, min_df=1, token_pattern=r"(?u)\b\w+\b"  # 한글 포함 토큰
)
# 데이터를 학습하고 TF-IDF 행렬 X를 생성 (희소 행렬 형태)
X = tfidf.fit_transform(docs)  # shape: (문서 수, 단어 수)

# --- 2. K-means 모델 학습 ---
k = 2  # 군집 개수 설정
# n_init="auto": 중심 초기화를 여러 번 시도하여 가장 좋은 결과를 선택 (최신 sklearn 권장)
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

# TF-IDF 행렬 X를 이용하여 K-means 모델 학습
kmeans.fit(X)
# 각 문서가 할당된 클러스터 레이블 (0 또는 1)
labels = kmeans.labels_
print("문서별 클러스터 라벨:", labels)

# --- 3. 클러스터링 결과 확인 ---
# 문서와 클러스터 정보를 같이 보기 위해 DataFrame 생성
df_cluster = pd.DataFrame({"doc_id": range(len(docs)), "text": docs, "cluster": labels})

print("\n[문서별 클러스터 할당 결과]")
print(df_cluster)

# 클러스터별 문서 묶어서 출력하여 클러스터링 결과 시각화
for c in range(k):
    print(f"\n=== 클러스터 {c} ===")
    # 현재 클러스터 c에 속하는 문서만 필터링하여 출력
    for text in df_cluster[df_cluster["cluster"] == c]["text"]:
        print("-", text)

# --- 4. 문서 및 클러스터별 대표 키워드 추출 함수 ---
# 단어집 (TF-IDF 벡터라이저의 피처 이름) 먼저 가져오기
feature_names = tfidf.get_feature_names_out()


# 개별 문서의 상위 키워드를 추출하는 함수
def get_top_keywords_for_doc(tfidf_ventor, top_n=5):
    # print(tfidf_ventor.toarray()) # 문서의 전체 TF-IDF 벡터 출력 (주석 처리)
    # 희소 행렬을 넘파이 배열로 변환하고 1차원으로 평탄화
    vec = tfidf_ventor.toarray().flatten()
    # 가중치가 큰 순서대로 인덱스 top_n개를 추출 (음수(-vec)로 변환 후 argsort는 내림차순 효과)
    top_idx = np.argsort(-vec)[:top_n]
    # 키워드 이름과 가중치를 묶어 반환
    return list(zip(feature_names[top_idx], vec[top_idx]))


# 0번 문서의 상위 3개 키워드 추출 및 출력 예시
top_keyword_doc0 = get_top_keywords_for_doc(X[0], top_n=3)
print("\n[0번 문서 대표 키워드]")
for word, score in top_keyword_doc0:
    print(f"{word} : {score:.3f}")


# 클러스터별 대표 키워드를 추출하는 함수 (평균 벡터 사용)
def get_top_keywords_for_cluster(X, labels, cluster_id, top_n=5):
    # 현재 클러스터 ID와 일치하는 문서들을 찾는 마스크 생성
    mask = labels == cluster_id

    # 1. 마스킹된 희소 행렬을 밀집 배열(toarray())로 변환 후
    # 2. 단어 축(axis=0)을 따라 평균을 계산하여 클러스터 대표 벡터를 생성 (안정성 확보)
    cluster_vec = X[mask].toarray().mean(axis=0)

    # 3. 평균 벡터 처리 (넘파이 배열이므로 flatten 해도 무방)
    vec = cluster_vec.flatten()
    # 4. 가중치가 큰 순서대로 인덱스 top_n개를 추출
    top_idx = np.argsort(-vec)[:top_n]

    # 5. 키워드와 평균 가중치를 묶어 반환
    return list(zip(feature_names[top_idx], vec[top_idx]))


# 모든 클러스터에 대해 대표 키워드 추출 및 출력
for c in range(k):
    # 현재 클러스터 c의 상위 3개 대표 키워드를 추출
    top_keywords = get_top_keywords_for_cluster(X, labels, cluster_id=c, top_n=3)
    print(f"\n==== 클러스터 {c} 대표 키워드 ====")
    for word, score in top_keywords:
        print(f"{word} : {score:.3f}")
