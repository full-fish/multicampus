import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from konlpy.tag import Okt

okt = Okt()
# N-gram, TF-IDF용 라이브러리
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.decomposition import (
    TruncatedSVD,
)  # LSA(잠재 의미 분석)를 위한 Truncated SVD

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

# ----------------------------------------
"""
1. 여성의류 쇼핑몰 .json 데이터 활용
2. 만족, 불만족, 보통의 데이터를 나누어 토픽 뽑기(LSA)
   -토픽당 주요 단어 20개 뽑아 출력해보기
3. LSA가 뽑은 토픽을 UMAP을 활용해 2D 산점도로 그려서 군집 형태 분석"""
base_dir = Path(r"stat_nlp/naive_bayes_svm/Sample/02.라벨링데이터")

# 불용어
with open(
    "stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())
stopwords.update(["하다", "하"])
"""
##! 1. 데이터 로드 및 기본 정보 확인
""" ""

df = []
i = 1
TARGET = 2000
counts = {1: 0, 0: 0, -1: 0}
while True:
    path = f"{base_dir}/쇼핑몰/01. 패션/1-1. 여성의류/1-1.여성의류({i}).json"
    try:
        print(i, "번 파일 읽음")
        temp = pd.read_json(path)

        if all(value >= TARGET for value in counts.values()):
            print("모든 클래스 600개씩 수집 완료")
            break

        for _, row in temp.iterrows():
            if pd.isna(row["GeneralPolarity"]):
                continue
            if counts[int(row["GeneralPolarity"])] < TARGET:
                counts[int(row["GeneralPolarity"])] += 1
                df.append(row.to_dict())

        i += 1
    except Exception as e:
        print(i, "번 파일에서 에러 발생")
        print("에러 내용:", e)
        break
df = pd.DataFrame(df)
df = df.dropna(subset=["RawText", "GeneralPolarity", "ReviewScore", "RDate"])
print("\ndf.head()", df.head())


def preprocess(text):
    str_reg = re.sub(r"[^가-힝0-9a-zA-Z\s]", "", text).lower()
    pos = okt.pos(str_reg, norm=True, stem=True, join=True)
    pos = [word.split("/") for word in pos]
    filtered_pos = [
        word
        for word, tag in pos
        if word and word not in stopwords and tag in ["Noun", "Verb", "Adjective"]
    ]
    return filtered_pos


n_topics = 3
neighbors_list = [5, 10, 30]
df["polarity_map"] = df["GeneralPolarity"].map({-1: 0, 0: 1, 1: 2})

print("\n==========================================")
print("[전체 데이터] LSA 분석 시작")
print("==========================================")

texts_all = df["RawText"].astype(str).tolist()

pipe_lsa = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=preprocess,
                token_pattern=None,
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=5,
            ),
        ),
        (
            "svd",
            TruncatedSVD(n_components=n_topics, random_state=42),
        ),
    ]
)

X_lsa_all = pipe_lsa.fit_transform(texts_all)
tfidf = pipe_lsa.named_steps["tfidf"]
svd = pipe_lsa.named_steps["svd"]
terms = tfidf.get_feature_names_out()

print(f"LSA 행렬 크기 (축소된 차원): {X_lsa_all.shape}")


print(f"\n[LSA 토픽별 주요 단어 (Top 20) - 전체 데이터]")
for topic_idx, comp in enumerate(svd.components_):
    term_idx = comp.argsort()[::-1][:20]
    top_terms = [terms[i] for i in term_idx]
    print(f"토픽 {topic_idx + 1}: {', '.join(top_terms)}")


# UMAP 시각화 (감성별 군집 분석)
plt.figure(figsize=(15, 5))

for i, n_nb in enumerate(neighbors_list, start=1):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_nb,
        min_dist=0.5,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    X_umap = reducer.fit_transform(X_lsa_all)

    plt.subplot(1, 3, i)

    scatter = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=df["polarity_map"],
        s=3,
        cmap="bwr",
        alpha=0.7,
    )

    plt.title(f"UMAP (n_neighbors = {n_nb})", fontsize=12)
    plt.xticks([])
    plt.yticks([])


plt.suptitle(
    "전체 데이터 UMAP 시각화",
    y=1.05,
    fontsize=14,
)
plt.tight_layout()
plt.show()
