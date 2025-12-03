import sys
import os

# 현재 파일의 절대 경로를 기준으로 최상위 패키지 경로를 계산
target_path = "/Users/choimanseon/Documents/multicampus/example/nlp"

if target_path not in sys.path:
    sys.path.append(target_path)

# 이제부터는 임포트가 작동해야 합니다.
from stat_nlp.common.preprocess_tidf import get_vectored_value

# ----------------------------------------

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
import matplotlib.pyplot as plt  # 시각화 라이브러리
from umap import UMAP
from sklearn.cluster import KMeans

from stat_nlp.common.preprocess_tidf import get_vectored_value
from sklearn.metrics import silhouette_score

# 한글 폰트 설정
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # 리눅스 (예: Google Colab, Ubuntu)
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

#!
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def make_scatter(n_topics, df_sample, X_tfidf):
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)

    neighbors_list = [5, 10, 15]
    plt.figure(figsize=(15, 4))

    for i, n_nb in enumerate(neighbors_list, start=1):
        # i: 서브플롯 위치 (1, 2, 3)
        # n_nb: n_neighbors 값 (5, 10, 15)

        # UMAP 모델 정의: n_neighbors만 루프마다 다르게 설정
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_nb,
            min_dist=0.5,
            metric="cosine",
            random_state=42,
        )

        # LSA 결과를 UMAP으로 변환 (루프마다 다시 학습/변환)
        X_umap = reducer.fit_transform(X_lsa)

        # i 번째 서브플롯에 그림을 그리기 시작
        plt.subplot(1, 3, i)

        # 산점도 시각화
        sc = plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=df_sample["label"],  # 라벨을 사용하여 색상 지정
            s=1,
            cmap="bwr",
            alpha=0.7,
        )

        # 그래프 제목 설정 (오류 수정: plt.title() 제거)
        plt.title(f"UMAP (n_neighbors = {n_nb})")
        plt.xticks([])  # x축 눈금 제거
        plt.yticks([])  # y축 눈금 제거
        # --------------------------------------------------------------------------------------

    plt.suptitle("UMAP : n_neighbors 따른 시각화 비교", y=1.02)  # 전체 제목 설정

    # 모든 서브플롯에 대해 한 번만 컬러바와 레이아웃을 설정합니다.
    # 컬러바는 산점도 밖에 위치하므로, 필요한 경우 마지막 서브플롯 코드 뒤에 추가합니다.
    # plt.colorbar(sc)

    plt.tight_layout()
    plt.show()


def elbow(X_tfidf):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

        kmeans.fit(X_tfidf)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertias, marker="o")
    plt.xlabel("k 클러스터 수")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method")
    plt.show()


def silhouette(X_tfidf):
    k_range = range(2, 11)
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

        labels = kmeans.fit_predict(X_tfidf)
        score = silhouette_score(X_tfidf, labels)
        sil_scores.append(score)
        print(f"k={k}, silhouette_score = {score:.3f}")

    plt.plot(k_range, sil_scores, marker="o")
    plt.xlabel("k cluster count")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.show()
