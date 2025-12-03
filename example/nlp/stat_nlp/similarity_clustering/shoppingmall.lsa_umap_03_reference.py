import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import platform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.model_selection import train_test_split


if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


okt = Okt()


def preprocess_text(text: str, local_stopwords: str) -> list:
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    morphs = okt.pos(text, stem=True)

    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            if word not in local_stopwords and len(word) > 1:
                tokens.append(word)

    return tokens


def get_vectorize_value(
    filename: str,
    file_type: str,
    vectorize_column_name: str,  # 벡터화할 컬럼 이름(함수의 재사용성을 위해: 여러파일에서 벡터대상 컬럼이 다를 수 있으므로)
    label_column_name: str,  # 샘플링을 할때 층화추출을 하고 싶다면 그 기준이되는 칼럼이름
    add_stopwords: set,  # 불용어에 추가단어가 있을때
    sample_size: int,
) -> tuple:
    if file_type == "csv":
        df = pd.read_csv(filename, encoding="utf-8")
    elif file_type == "json":
        df = pd.read_json(filename, encoding="utf-8")
    else:
        print("지원하지 않는 파일타입입니다.")
        return

    df = df.dropna()
    TEXT_COL = vectorize_column_name

    if label_column_name is not None:
        stratify_column = df[label_column_name]
    else:
        stratify_column = None

    sample_flag = True

    if len(df) <= sample_size:
        sample_flag = False

    if sample_flag:
        _, df_sample = train_test_split(
            df,
            test_size=sample_size,
            stratify=stratify_column,
            shuffle=True,
            random_state=42,
        )
    else:
        df_sample = df.copy()

    df_sample = df_sample.reset_index(drop=True)

    with open(
        r"D:\multicompus\exam\nlp\stat_nlp\common\stopwords-ko.txt", encoding="utf-8"
    ) as f:
        stopwords = set(w.strip() for w in f if w.strip())

    if add_stopwords is not None:
        stopwords.update(add_stopwords)

    def tokenizer(text: str):
        return preprocess_text(text, stopwords)

    tfidf = TfidfVectorizer(
        max_df=0.8, min_df=1, token_pattern=None, tokenizer=tokenizer
    )

    X_tfidf = tfidf.fit_transform(df_sample[TEXT_COL])
    return df_sample, X_tfidf, tfidf


add_word = set(["하다", "입다", "이다", "않다", "입다"])

df, X_tfidf, tfidf = get_vectorize_value(
    filename=r"D:\multicompus\exam\nlp\stat_nlp\naive_bayes_linearsvc\woman_wear_balanced_stream_600_each.json",
    file_type="json",
    vectorize_column_name="RawText",
    label_column_name="GeneralPolarity",
    add_stopwords=add_word,
    sample_size=1800,
)

n_topics = 5

svd = TruncatedSVD(n_components=n_topics, random_state=42)

print("\n[LSA] 전체 데이터에 대해 TF-IDF → SVD 학습 중...")
X_lsa = svd.fit_transform(X_tfidf)

print("TF-IDF shape :", X_tfidf.shape)
print("LSA shape    :", X_lsa.shape)

reducer = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.2, metric="cosine", random_state=42
)

print("\n[UMAP] 2차원 임베딩 중...")
X_umap = reducer.fit_transform(X_lsa)

df["umap_x"] = X_umap[:, 0]
df["umap_y"] = X_umap[:, 1]

# plt.figure(figsize=(7, 6))

# label_info = [
#     (-1, "부정(-1)", "tab:blue"),
#     (0,  "중립(0)",  "tab:gray"),
#     (1,  "긍정(1)",  "tab:red"),
# ]

# for label, name, color in label_info:
#     mask = (df["GeneralPolarity"] == label)
#     plt.scatter(
#         df.loc[mask, "umap_x"],
#         df.loc[mask, "umap_y"],
#         s=8,
#         alpha=0.7,
#         label=name,
#         c=color
#     )

# plt.title("UMAP : GeneralPolarity에 따른 전체 리뷰 시각화")
# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.tight_layout()
# plt.show()


# -----------------------------
# UMAP 좌표 기준 K-Means 군집 & 군집별 샘플 보기
# -----------------------------
n_clusters = 6  # 군집 개수(원하는 대로 조정 가능)
print(f"\n[K-Means] UMAP 좌표로 {n_clusters}개 군집 생성 중...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_umap)
# kmeans.fit(X_umap)만 하고 kmeans.labels_를 확인해도 같은 결과

# (선택) 군집별 색으로 다시 시각화
plt.figure(figsize=(7, 6))
for c in range(n_clusters):
    mask = df["cluster"] == c
    plt.scatter(
        df.loc[mask, "umap_x"],
        df.loc[mask, "umap_y"],
        s=8,
        alpha=0.7,
        label=f"cluster {c}",
    )

plt.title("UMAP + KMeans 군집 시각화")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7. 각 군집에서 예시 리뷰 몇 개씩 출력
# -----------------------------
SAMPLES_PER_CLUSTER = 5  # 군집당 보고 싶은 예시 개수
print(df.head())
# for c in range(n_clusters):
#     sub = df[df["cluster"] == c]

#     print("\n" + "=" * 80)
#     print(f"[클러스터 {c}] 샘플 수: {len(sub)}")
#     print("감성 라벨 분포:")
#     print(sub["GeneralPolarity"].value_counts())

#     if len(sub) == 0:
#         continue

#     # 군집에서 몇 개 샘플 뽑기
#     sample_n = min(SAMPLES_PER_CLUSTER, len(sub))
#     samples = sub.sample(sample_n, random_state=42)

#     for i, (_, row) in enumerate(samples.iterrows(), start=1):
#         print("-" * 80)
#         print(f"[예시 {i}] 라벨={row['GeneralPolarity']}")
#         print(row["RawText"])
