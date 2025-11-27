import numpy as np  # 배열 및 수학 연산을 위한 라이브러리 (결과 출력에 사용)
from sklearn.feature_extraction.text import (
    CountVectorizer,
)  # 텍스트를 단어 빈도 벡터로 변환하는 클래스
from sklearn.decomposition import LatentDirichletAllocation  # LDA 모델을 구현한 클래스

# 분석할 문서(doc) 리스트. LDA의 입력 데이터가 됩니다.
docs = [
    "배송이 빠르고 포장이 깔끔해서 좋았어요",
    "배송이 느리고 박스가 찢어져 와서 별로였어요",
    # .... (여기에 더 많은 문서가 있다고 가정합니다.)
]

# --- 1. 단어 카운트 벡터화 (피처 추출) ---

# CountVectorizer 객체 생성. 텍스트를 단어 빈도 기반의 행렬로 변환합니다.
vectorizer = CountVectorizer()

# fit_transform을 통해 다음 두 가지 작업을 동시에 수행합니다.
# 1. fit: docs를 분석하여 단어장(Vocabulary)을 구축합니다. (어떤 단어가 있는지 파악)
# 2. transform: docs를 단어장 기반의 빈도 행렬(문서-단어 행렬)로 변환합니다.
# X는 희소 행렬(Sparse Matrix) 형태로 저장됩니다.
X = vectorizer.fit_transform(docs)

# 문서-단어 행렬의 크기를 출력합니다.
# (문서 수, 단어 수). LDA 모델의 입력 차원입니다.
print("문서-단어 행렬 크기:", X.shape)

# --- 2. LDA 모델 설정 및 학습 ---

n_topics = 3  # 추출하고자 하는 토픽의 개수 K를 설정합니다.

# LatentDirichletAllocation 객체 생성
lda = LatentDirichletAllocation(
    n_components=n_topics,  # 추출할 토픽 개수 K를 지정합니다.
    learning_method="batch",  # 학습 방법 지정: 'batch'는 전체 데이터를 한 번에 보고 학습합니다.
    random_state=42,  # 재현성을 위한 난수 시드 설정
)

# 학습 (fit) 및 변환 (transform)을 동시에 수행합니다.
# fit: X를 이용해 토픽-단어 분포 (phi: lda.components_)를 추정합니다.
# transform: 학습된 phi를 이용해 X를 문서-토픽 분포 (theta: doc_topic)로 변환합니다.
# doc_topic은 각 문서별 토픽 비중이 저장된 행렬입니다.
doc_topic = lda.fit_transform(X)  # shape: (n_docs, n_topics)

# (참고용 주석)
# print(doc_topic.sum(axis=1)) # 각 행(문서)의 합은 1에 가까워야 합니다.
# print(lda.components_) # 토픽-단어 분포 행렬 (phi)

# --- 3. 결과 분석 및 출력 ---


# 토픽별 상위 단어를 출력하는 함수를 정의합니다.
# model.components_는 토픽-단어 분포 (phi)를 나타냅니다.
# 이는 토픽 k에서 단어 w가 나타날 확률을 나타냅니다.
def print_topics(model, feature_names, n_top_words=5):
    # model.components_는 (n_topics, n_features) 크기의 행렬입니다.
    # 각 행은 하나의 토픽을, 각 열은 단어(feature)를 나타냅니다.
    for topic_idx, topic in enumerate(model.components_):

        # topic: 각 단어별 "카운트 비슷한 값" (클수록 그 토픽에서 더 중요한 단어)
        # 1. topic.argsort(): 토픽에 대한 단어들의 중요도를 오름차순으로 정렬한 인덱스 배열을 반환합니다.
        # 2. [::-1]: 인덱스 배열을 내림차순으로 뒤집습니다. (가장 중요한 단어가 앞으로 옴)
        # 3. [:n_top_words]: 앞에서부터 n_top_words 개만 선택합니다.
        top_indices = topic.argsort()[::-1][:n_top_words]

        # 선택된 인덱스를 이용해 실제 단어(feature_names)를 추출합니다.
        top_words = [feature_names[i] for i in top_indices]

        # 토픽 번호와 상위 단어를 출력합니다.
        print(f"Topic {topic_idx}: {', '.join(top_words)}")


# CountVectorizer가 학습한 단어장(feature) 이름을 가져옵니다.
feature_names = vectorizer.get_feature_names_out()

# 정의한 함수를 호출하여 토픽별 상위 5개 단어를 출력합니다.
print_topics(lda, feature_names, n_top_words=5)
print("\n\ndoc_topic\n", doc_topic)

# 문서별 토픽 비중(theta: doc_topic)을 확인합니다.
for i, topic_dist in enumerate(doc_topic):
    # 각 문서가 3개의 토픽에 대해 어떤 비율로 구성되어 있는지 출력합니다.
    # np.round를 사용하여 소수점 셋째 자리까지 반올림하여 출력합니다.
    print(f"문서 {i} 토픽 분포:", np.round(topic_dist, 3))
