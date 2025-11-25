"""
# 여성의류 리뷰 데이터 EDA 실습 노트북

이 노트북에서는 여성의류 라벨링 데이터(JSON)를 이용해 다음과 같은 탐색적 데이터 분석(EDA)을 수행합니다.

1. 데이터 기본 구조 및 결측치 확인
2. 리뷰 길이(문자 수, 단어 수) 분포 분석
3. 리뷰 날짜 분포 분석
4. ReviewScore 분포 및 길이와의 관계
5. GeneralPolarity(전체 감성) 분포 분석
6. Aspect 기반 EDA (Aspect 빈도, Aspect별 감성 분포, 사이즈/가격 vs ReviewScore)
7. WordCloud / N-gram / TF-IDF 기반 텍스트 분석
8. Aspect별 대표 문장 살펴보기

> **주의:** 아래 경로(`JSON_PATH`)를 본인의 파일 경로에 맞게 수정한 뒤 실행하세요."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from konlpy.tag import Okt
from wordcloud import WordCloud

okt = Okt()
# N-gram, TF-IDF용 라이브러리
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

WORDCLOUD_FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
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

# JSON 파일 경로를 본인 환경에 맞게 수정하세요.
base_dir = Path(r"stat_nlp/naive_bayes_svm/Sample/02.라벨링데이터")  # 예시

# 불용어
with open(
    "stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())

"""
##! 1. 데이터 로드 및 기본 정보 확인
""" ""

df = []
i = 1
TARGET = 600
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

"""
##! 2. 리뷰 길이 분석 (문자 수, 단어 수)

- RawText 길이(문자 수)
- RawText 단어 수
- 길이 분포 히스토그램 / 박스플롯
""" ""

df["char_len"] = df["RawText"].str.len()
df["word_len"] = df["RawText"].str.split().str.len()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
df["char_len"].hist(bins=50, color="skyblue", edgecolor="black")
plt.title("RawText의 글자 개수", fontsize=15)
plt.xlabel("글자 개수")
plt.ylabel("리뷰 수")
plt.grid()

plt.subplot(1, 2, 2)
plt.boxplot(df["char_len"].dropna())
plt.title("RawText의 글자 개수", fontsize=15)
plt.xlabel("글자 개수")
plt.tight_layout()
# plt.show()


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
df["word_len"].hist(bins=50, color="lightcoral", edgecolor="black")
plt.title("RawText의 단어 개수", fontsize=15)
plt.xlabel("단어 개수")
plt.ylabel("리뷰 수")
plt.grid()

plt.subplot(1, 2, 2)
plt.boxplot(df["word_len"].dropna())
plt.title("RawText의 단어 개수", fontsize=15)
plt.xlabel("단어 개수")
plt.tight_layout()
# plt.show()

"""
##! 3. 리뷰 날짜(RDate) 분포

- 날짜 컬럼을 datetime으로 변환
- 전체 기간 동안 리뷰가 어떻게 분포하는지 확인
""" ""
df["RDate_datetime"] = pd.to_datetime(df["RDate"], format="%Y%m%d", errors="coerce")

df = df.dropna(subset=["RDate_datetime"])
# print(df.head().to_markdown())
review_counts_daily = df.groupby(df["RDate_datetime"].dt.date).size()

plt.figure(figsize=(15, 6))

review_counts_daily.plot(kind="line", color="darkblue", linewidth=1)
plt.title("전체 기간 동안 리뷰 수 분포", fontsize=15)
plt.xlabel("날짜")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
# plt.show()

"""
##! 4. ReviewScore 분포 및 길이와의 관계
""" ""
# 리뷰 개수 분포
df["ReviewScore"] = df["ReviewScore"].astype(int)
review_score_counts = df["ReviewScore"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

review_score_counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("리뷰 점수 분포", fontsize=15)
plt.xlabel("점수")
plt.xticks(rotation=0)
plt.ylabel("리뷰 개수1")
plt.grid()
plt.tight_layout()
# plt.show()

# 글자 수와 리뷰 점수 관계
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(
    x="ReviewScore", y="char_len", data=df, hue="ReviewScore", palette="viridis"
)
plt.title("ReviewScore별 리뷰 길이(문자 수) 분포", fontsize=15)
plt.xlabel("리뷰 점수 (ReviewScore)")
plt.ylabel("리뷰 길이 (문자 수)")
plt.grid()
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.boxplot(
    x="ReviewScore", y="word_len", data=df, hue="ReviewScore", palette="viridis"
)
plt.title("ReviewScore별 리뷰 길이(단어 수) 분포", fontsize=15)
plt.xlabel("리뷰 점수 (ReviewScore)")
plt.ylabel("리뷰 길이 (단어 수)")
plt.grid()
plt.tight_layout()

"""
##! 5. GeneralPolarity(전체 감성) 분포
- -1: 부정, 0: 중립, 1: 긍정
""" ""
df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)
generalPolarity_sorted = df["GeneralPolarity"].value_counts().sort_index()
generalPolarity_total_count = df["GeneralPolarity"].value_counts().sum()


def pie_format(percent, allvals):
    """
    퍼센트(pct)를 입력받아 '개수 (비율%)' 형식의 문자열을 반환합니다.
    """
    absolute = int(np.round(percent / 100.0 * allvals))

    return f"{absolute} ({percent:.1f}%)"


plt.figure(figsize=(7, 7))
plt.pie(
    generalPolarity_sorted,
    labels=["부정", "중립", "긍정"],
    autopct=lambda percent: pie_format(percent, generalPolarity_total_count),
    startangle=90,
    colors=["lightcoral", "lightgray", "lightgreen"],
    wedgeprops={"edgecolor": "black"},
)
plt.tight_layout()
# plt.show()

"""
##! 6. Aspect 기반 EDA

- Aspects 컬럼을 행 단위로 풀어서 하나의 DataFrame(`aspect_df`) 생성
- Aspect 종류별 빈도
- Aspect별 감성 분포(-1/0/1)
- "사이즈" / "가격"에 대한 감성과 ReviewScore 관계 보기"""
# aspect_df = [row for row in df["Aspects"]]
# print("\naspect_df", (aspect_df))
temp = df[["RawText", "GeneralPolarity", "ReviewScore", "RDate", "Aspects"]].copy()

# 한 리뷰 안의 여러 Aspect를 행으로 분리
# explode 하면 리스트 개수만큼 행이 생김
temp = temp.explode("Aspects").dropna(subset=["Aspects"])
# json_normaliz: obj를 컬럼화해서 펼침
aspect_info = pd.json_normalize(temp["Aspects"])
# print("\n", temp.iloc[0])
# print("\n", aspect_info.iloc[0])
# print("\ntmr", len(temp), len(aspect_info))
aspect_df = pd.concat(
    [
        temp.drop(columns=["Aspects"]).reset_index(drop=True),
        aspect_info.reset_index(drop=True),
    ],
    axis=1,
)
# print("\naspect_df.head()", aspect_df.head())
aspect_counts = aspect_df["Aspect"].value_counts()
SentimentPolarity_counts = aspect_df["SentimentPolarity"].value_counts().sort_index()
top_N = 20
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
aspect_counts.head(top_N).sort_values().plot(kind="barh")

plt.title(f"Aspect별 언급 개수 (상위 {top_N}개)", fontsize=15)
plt.xlabel("언급 개수")
plt.ylabel("Aspect")
plt.grid(axis="x")
plt.tight_layout()

# pie 그래프
plt.subplot(1, 2, 2)

SentimentPolarity_sorted = aspect_df["SentimentPolarity"].value_counts().sort_index()
SentimentPolarity_total_count = aspect_df["SentimentPolarity"].value_counts().sum()
plt.pie(
    SentimentPolarity_sorted,
    labels=["부정", "중립", "긍정"],
    autopct=lambda percent: pie_format(percent, SentimentPolarity_total_count),
    startangle=90,
    colors=["lightcoral", "lightgray", "lightgreen"],
    wedgeprops={"edgecolor": "black"},
)
plt.tight_layout()
# plt.show()

# 사이즈" / "가격"에 대한 감성과 ReviewScore 관계 보기
target_aspects_df = aspect_df[aspect_df["Aspect"].isin(["사이즈", "가격"])].copy()

score_analysis = target_aspects_df.groupby(["Aspect", "SentimentPolarity"])[
    "ReviewScore"
].mean()
print("\nscore_analysis\n", score_analysis)

score_pivot = score_analysis.unstack()  # 2차원 형태로 펼침

# 4. 시각화
plt.figure(figsize=(10, 6))

score_pivot.plot(
    kind="bar",
    rot=0,
    ax=plt.gca(),
    color=[
        "lightcoral",
        "lightgray",
        "lightgreen",
    ],  # 부정(-1), 중립(0), 긍정(1)에 맞춤
    edgecolor="black",
)

plt.title("'사이즈'/'가격' Aspect 감성별 평균 ReviewScore", fontsize=15)
plt.xlabel("Aspect 감성 분류 (-1:부정, 0:중립, 1:긍정)", fontsize=12)
plt.ylabel("평균 ReviewScore", fontsize=12)
plt.legend(title="Aspect")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
"""
##! 7. 텍스트 기반 분석: N-gram(CountVectorizer) / TF-IDF(긍/부정 상위 20개 단어 추출)
""" ""


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


def get_top_ngrams(df_series, top_n=20):
    vectorizer = CountVectorizer(
        tokenizer=preprocess, token_pattern=None, ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df_series)
    feature_names = vectorizer.get_feature_names_out()

    word_scores = X.sum(axis=0).A1

    scores_series = pd.Series(word_scores, index=feature_names)
    return scores_series.nlargest(top_n)


def get_top_tfidf(df, polarity, top_n=10):
    df_filtered = df[df["GeneralPolarity"] == polarity]["RawText"]

    vectorizer = TfidfVectorizer(
        tokenizer=preprocess, token_pattern=None, ngram_range=(1, 1)
    )
    X = vectorizer.fit_transform(df_filtered)
    feature_names = vectorizer.get_feature_names_out()

    word_scores = X.sum(axis=0).A1

    scores_series = pd.Series(word_scores, index=feature_names)
    return scores_series.nlargest(top_n)


print("--- N-gram 빈도 분석 ---")
top_ngrams = get_top_ngrams(df["RawText"], 10)
print(top_ngrams.to_markdown(floatfmt=".0f"))

print("\n--- TF-IDF 분석 - 긍정 리뷰 ---")
top_tfidf_pos = get_top_tfidf(df, 1, 10)
print(top_tfidf_pos.to_markdown(floatfmt=".3f"))

print("\n--- [TF-IDF 분석 - 부정 리뷰 ---")
top_tfidf_neg = get_top_tfidf(df, -1, 10)
print(top_tfidf_neg.to_markdown(floatfmt=".3f"))

"""
##! 8. Aspect별 대표 문장 살펴보기(문장 길이 기준)
- 사이즈 부정 문장 Top 20
- 가격 긍정 문장 Top 20
""" ""
aspect_df["char_len"] = aspect_df["RawText"].str.len()
print("\n\n\n", aspect_df.head())

size_neg_top20 = (
    aspect_df[
        (aspect_df["Aspect"] == "사이즈") & (aspect_df["SentimentPolarity"] == "-1")
    ]
    .sort_values("char_len", ascending=False)
    .head(20)[["RawText"]]
)
cost_pos_top20 = (
    aspect_df[(aspect_df["Aspect"] == "가격") & (aspect_df["SentimentPolarity"] == "1")]
    .sort_values("char_len", ascending=False)
    .head(20)[["RawText"]]
)

print(size_neg_top20)
print(cost_pos_top20)

##! wordcloud
# df의 RawText 컬럼 전체를 하나의 긴 문자열(corpus)로 결합합니다.
text_corpus = " ".join(df["RawText"].astype(str))

# WordCloud 객체 생성
wordcloud = WordCloud(
    font_path=WORDCLOUD_FONT_PATH,  # 🌟 수정된 맥용 폰트 경로 사용 🌟
    stopwords=stopwords,
    background_color="white",
    width=800,
    height=600,
    max_words=100,
    scale=2,
)

# 4. WordCloud 생성
wordcloud.generate(text_corpus)

# 5. 시각화
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("전체 리뷰 텍스트 WordCloud", fontsize=15)
plt.tight_layout()
plt.show()
