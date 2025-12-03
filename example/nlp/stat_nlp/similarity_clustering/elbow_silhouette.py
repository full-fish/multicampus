import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from stat_nlp.common.graph import elbow, silhouette

# 코드 상단에 추가되어야 할 import 구문
from sklearn.metrics import silhouette_score

df = pd.read_csv(
    r"/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/movie_reviews.csv",
    encoding="utf-8",
)
df = df.dropna()
_, df_sample = train_test_split(
    df, test_size=10000, stratify=df["label"], shuffle=True, random_state=42
)
df_sample = df_sample.reset_index(drop=True)
okt = Okt()
with open(
    r"/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())
add_stopwords = set(
    [
        "그리고",
        "그래서",
        "하지만",
        "그러나",
        "너무",
        "정말",
        "진짜",
        "조금",
        "또한",
        "그냥",
        "매우",
        "에서",
        "에게",
        "미만",
        "이상",
        "같은",
        "하다",
        "되다",
    ]
)
stopwords.update(add_stopwords)


def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    morphs = okt.pos(text, stem=True)
    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords and len(word) > 1:
                tokens.append(word)
    return tokens


TEXT_COL = "document"

tfidf = TfidfVectorizer(
    max_df=0.8, min_df=1, token_pattern=None, tokenizer=preprocess_text
)

X_tfidf = tfidf.fit_transform(df_sample[TEXT_COL])

inertias = []

elbow(X_tfidf)
silhouette(X_tfidf)
# #! 엘보우
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

#     kmeans.fit(X_tfidf)
#     inertias.append(kmeans.inertia_)
# plt.plot(range(1, 11), inertias, marker="o")
# plt.xlabel("k 클러스터 수")
# plt.ylabel("Inertia (SSE)")
# plt.title("Elbow Method")
# plt.show()

# #! 실루엣
# k_range = range(2, 11)
# sil_scores = []
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

#     labels = kmeans.fit_predict(X_tfidf)
#     score = silhouette_score(X_tfidf, labels)
#     sil_scores.append(score)
#     print(f"k={k}, silhouette_score = {score:.3f}")

# plt.plot(k_range, sil_scores, marker="o")
# plt.xlabel("k cluster count")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Method")
# plt.show()
