# 필요한 라이브러리 임포트
import re  # 정규 표현식 처리를 위한 라이브러리
from sklearn.svm import LinearSVC  # 선형 SVM 분류기(확률 출력 X)
from sklearn.calibration import (
    CalibratedClassifierCV,
)  # SVM 결과를 확률로 보정해 주는 래퍼
from sklearn.pipeline import Pipeline  # 여러 전처리/모델을 순차적으로 묶는 도구
from sklearn.metrics import (
    classification_report,
)  # 분류 리포트(precision, recall, f1 등) 출력
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)  # 데이터 분할과 하이퍼파라미터 탐색
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # 텍스트를 TF-IDF 벡터로 변환
import pandas as pd  # 데이터프레임 처리용 라이브러리
import numpy as np  # 수치 연산용 라이브러리
from sklearn.decomposition import (
    TruncatedSVD,
)  # LSA(잠재 의미 분석)를 위한 Truncated SVD
from sklearn.manifold import TSNE  # 비선형 차원 축소 알고리즘 (시각화용)
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 데이터 로드 및 전처리
# --------------------------------------------------------------------------------

# 영화 리뷰 데이터 CSV 파일 읽기
# 파일 경로와 컬럼 이름 지정
df = pd.read_csv(
    "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/text_preprocess/movie_reviews.csv",
    header=0,  # 첫 줄을 헤더(제목)로 간주
    names=[
        "id",
        "review",
        "label",
    ],  # 컬럼 이름 강제 지정: 'id', 'review'(텍스트), 'label'(긍정/부정)
)

df = df.dropna()  # 결측값(NA) 포함된 행 제거

# label 컬럼을 정수형으로 강제 변환 (0, 1 레이블을 확실히 정수로 맞춰 줌)
df["label"] = df["label"].astype(int)

print(df.head())  # 데이터 상위 5개 행 출력 (데이터 구조 확인)
# print(df["label"].value_counts()) # 레이블 개수 확인 (필요시만 사용)
print(
    df["label"].value_counts(normalize=True)
)  # 레이블 비율 확인 (0/1 비율이 균등한지 확인)

# 2. 전체 데이터에서 층화추출(stratify)로 샘플 추출 (대규모 데이터셋에서 샘플링)
# --------------------------------------------------------------------------------

# 전체 df에서 1000개의 샘플만 추출하여 df_sample에 저장 (나머지는 _ 변수에 저장)
_, df_sample = train_test_split(
    df,
    test_size=1000,  # 총 데이터 중 1000개를 샘플로 사용
    stratify=df["label"],  # 레이블 비율(긍정/부정)을 유지하면서 샘플 분할 (층화 추출)
    shuffle=True,  # 데이터를 섞어서 추출 (기본값)
    random_state=42,  # 재현 가능한 결과를 위한 시드 고정
)

print(len(df_sample))  # 샘플 데이터 크기(1000) 출력
print(len(_))  # 나머지 데이터 크기 출력

# 3. 텍스트 정제 함수 정의 및 적용
# --------------------------------------------------------------------------------


# 한국어 텍스트 클리닝 함수 정의
def simple_korean_clean(text):
    # 1. 한글, 숫자, 공백 외의 모든 문자(특수문자, 영문 등)를 공백으로 대체
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    # 2. 연속된 공백(하나 이상의 \s)을 하나의 공백으로 압축하고, 양쪽 끝 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 'review' 컬럼에 클리닝 함수 적용하여 새로운 'clean' 컬럼 생성
df_sample["clean"] = df_sample["review"].astype(str).apply(simple_korean_clean)
# 클리닝된 텍스트 목록을 list 형태로 변환 (Vectorization 입력 형식)
texts = df_sample["clean"].tolist()

# 4. LSA 파이프라인 정의 및 적용 (TF-IDF -> SVD)
# --------------------------------------------------------------------------------

# LSA(Truncated SVD)를 통해 축소할 주제(토픽)의 개수 설정
n_topics = 5

# TfidfVectorizer와 TruncatedSVD를 순차적으로 묶는 파이프라인 정의
pipe_lsa = Pipeline(
    steps=[
        # 1단계: TF-IDF 행렬 변환기 (고차원 특징 추출)
        (
            "tfidf",
            TfidfVectorizer(
                max_df=0.7,  # 문서 70% 이상에 등장하는 단어 제외
                min_df=5,  # 문서 5개 미만에 등장하는 단어 제외
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        # 2단계: Truncated SVD (LSA 수행, 5차원으로 차원 축소)
        (
            "svd",
            TruncatedSVD(n_components=n_topics, random_state=42),
        ),
    ]
)

# 파이프라인 실행: 텍스트를 TF-IDF 행렬로 변환하고 5차원의 LSA 주제 공간으로 최종 변환
X_lsa = pipe_lsa.fit_transform(texts)

# 학습 완료된 파이프라인 내부의 개별 객체(변환기)를 추출
tfidf = pipe_lsa.named_steps["tfidf"]  # TfidfVectorizer 객체 추출
svd = pipe_lsa.named_steps["svd"]  # TruncatedSVD 객체 추출

# 5. LSA 결과 크기 및 주제별 핵심 단어 확인
# --------------------------------------------------------------------------------

print("TF-IDF shape : ", tfidf.transform(texts).shape)  # TF-IDF 행렬의 크기 (고차원)
print("LSA shape : ", X_lsa.shape)  # LSA 최종 행렬의 크기 (저차원: 1000 x 5)

terms = tfidf.get_feature_names_out()  # TF-IDF에 사용된 전체 단어 사전 추출

# 각 주제(토픽)별 상위 10개 단어 출력
for topic_idx, comp in enumerate(svd.components_):
    # comp.argsort()[::-1][:10] : 가중치가 가장 높은 상위 10개 단어의 인덱스 추출
    term_idx = comp.argsort()[::-1][:10]
    print(f"\n[토픽 {topic_idx}]")
    # 인덱스를 이용해 실제 단어(terms)를 찾아 쉼표로 연결하여 출력
    print(", ".join(terms[i] for i in term_idx))

# 6. LSA 결과에 t-SNE 적용 및 시각화
# --------------------------------------------------------------------------------

# LSA로 축소된 5차원 데이터를 t-SNE를 이용해 2차원으로 시각화
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_lsa)

# 2차원 좌표를 데이터프레임에 저장
df_sample["tsne_x"] = X_2d[:, 0]
df_sample["tsne_y"] = X_2d[:, 1]

# 산점도(Scatter Plot) 시각화
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    df_sample["tsne_x"],
    df_sample["tsne_y"],
    c=df_sample["label"],  # c=df["label"] 오류 수정: 샘플 데이터의 라벨 사용
    s=15,
    alpha=0.7,
)

# 그래프 꾸미기
plt.title("문서들의 2D LSA공간 (t-SNE)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.colorbar(
    scatter, ticks=[0, 1], label="Label (0: Negative, 1: Positive)"
)  # 컬러바에 라벨 추가
plt.tight_layout()
# plt.show() # 주석 처리됨: 시각화 창을 띄우는 함수

# 7. 특정 영역의 데이터 추출 및 확인
# --------------------------------------------------------------------------------

# t-SNE Y축 좌표가 -40 이상 -30 이하인 데이터만 추출 (특정 클러스터 확인용)
extract_df = df_sample[(df_sample["tsne_y"] >= -40) & (df_sample["tsne_y"] <= -30)]

print(len(extract_df))  # 추출된 데이터의 개수 출력

# 추출된 데이터 중 10개를 무작위로 샘플링하여 출력 (데이터 내용 확인)
random10_df = extract_df.sample(n=10, random_state=42)
print("\n\nrnadome10_df\n", random10_df)
