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


if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

okt = Okt()


df = pd.read_json(
    r"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\woman_wear_sample_600.json",
    encoding="utf-8",
)

df = df.dropna(subset=["RawText", "GeneralPolarity"])
df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)

X_text = df["RawText"]
y = df["GeneralPolarity"]

print("\n레이블 분포(클래스별 샘플 수)")
print(y.value_counts())


with open(
    r"/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())

add_word = set(["하다", "입다", "이다", "않다", "입다"])
stopwords.update(add_word)


def preprocess_text(text: str) -> list:
    """한글 리뷰를 정규화 + 형태소 분석해서 (명사/동사/형용사) 토큰만 반환"""
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

    morphs = okt.pos(text, norm=True, stem=True)
    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords and len(word) > 1:
                tokens.append(word)
    return tokens


n_topics = 5

pipe_lsa = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                max_df=0.8,
                min_df=3,
                token_pattern=None,  # 직접 tokenizer 사용
                tokenizer=preprocess_text,
            ),
        ),
        ("svd", TruncatedSVD(n_components=n_topics, random_state=42)),
    ]
)

print("\n[LSA] 전체 데이터에 대해 TF-IDF → SVD 학습 중...")
X_lsa = pipe_lsa.fit_transform(X_text)
tfidf = pipe_lsa.named_steps["tfidf"]
svd = pipe_lsa.named_steps["svd"]

print("TF-IDF shape :", tfidf.transform(X_text).shape)
print("LSA shape    :", X_lsa.shape)


reducer = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.2, metric="cosine", random_state=42
)

print("\n[UMAP] 2차원 임베딩 중...")
X_umap = reducer.fit_transform(X_lsa)


df["umap_x"] = X_umap[:, 0]
df["umap_y"] = X_umap[:, 1]


plt.figure(figsize=(7, 6))

label_info = [
    (-1, "부정(-1)", "tab:blue"),
    (0, "중립(0)", "tab:gray"),
    (1, "긍정(1)", "tab:red"),
]

for label, name, color in label_info:
    mask = df["GeneralPolarity"] == label
    plt.scatter(
        df.loc[mask, "umap_x"],
        df.loc[mask, "umap_y"],
        s=8,
        alpha=0.7,
        label=name,
        c=color,
    )

plt.title("UMAP : GeneralPolarity에 따른 전체 리뷰 시각화")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

# # -----------------------------
# # 6. UMAP 좌표 기준 K-Means 군집 & 군집별 샘플 보기
# # -----------------------------
# n_clusters = 6   # 군집 개수(원하는 대로 조정 가능)
# print(f"\n[K-Means] UMAP 좌표로 {n_clusters}개 군집 생성 중...")

# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
# df["cluster"] = kmeans.fit_predict(X_umap)

# # (선택) 군집별 색으로 다시 시각화
# plt.figure(figsize=(7, 6))
# for c in range(n_clusters):
#     mask = (df["cluster"] == c)
#     plt.scatter(
#         df.loc[mask, "umap_x"],
#         df.loc[mask, "umap_y"],
#         s=8,
#         alpha=0.7,
#         label=f"cluster {c}"
#     )

# plt.title("UMAP + KMeans 군집 시각화")
# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.tight_layout()
# plt.show()

# # -----------------------------
# # 7. 각 군집에서 예시 리뷰 몇 개씩 출력
# # -----------------------------
# SAMPLES_PER_CLUSTER = 5  # 군집당 보고 싶은 예시 개수

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
