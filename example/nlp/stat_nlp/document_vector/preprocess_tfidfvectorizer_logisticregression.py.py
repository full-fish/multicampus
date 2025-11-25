import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

"""
1. movie_reviews.csv 읽어서 DataFrame (결측치 삭제)
2. train_test_split() 함수로 층화추출로 샘플 10000개 추출
3. 샘플 10000개에 대해서 train, test로 분할
4. 텍스트전처리 (함수로 구현)
  - 영문자(소문자)
  - 한글/영문/숫자/공백 외 특수문자 제거
  - 형태소 분석 + 품사 태깅
  - 사용할 품사만 선택 (명사, 동사, 형용사)
  - 불용어 제거
5. TfdifVectorizer()로 훈련/테스트 데이터 벡터화
6. 모델 훈련(LogisticRegression)
7. 모델 테스트 검증 점수 출력"""

# 1. movie_reviews.csv 읽어서 DataFrame (결측치 삭제)
df = pd.read_csv(
    "stat_nlp/document_vector/movie_reviews.csv",
    header=0,
    names=["id", "review", "label"],
)

df = df.dropna()

# label을 정수형으로 변환 (0, 1). 지금 정수형이긴 하지만 혹시 모르니까
df["label"] = df["label"].astype(int)

print(df.head())
# print(df["label"].value_counts()) # 개수 확인
print(df["label"].value_counts(normalize=True))  # 비율 확인

# 2. train_test_split() 함수로 층화추출로 샘플 10000개 추출
_, df_sample = train_test_split(
    df,
    test_size=10000,
    stratify=df["label"],
    shuffle=True,  # 기본값이라 사실 안 써도 됨
    random_state=42,
)
print(len(df_sample))
print(len(_))

X = df_sample["review"]
y = df_sample["label"]

# 3. 샘플 10000개에 대해서 train, test로 분할
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 0.2 → 20퍼센트 test
    stratify=y,  # 샘플 안에서도 0/1 비율 유지
    shuffle=True,
    random_state=42,
)

"""
4. 텍스트전처리 (함수로 구현)
  - 영문자(소문자)
  - 한글/영문/숫자/공백 외 특수문자 제거
  - 형태소 분석 + 품사 태깅
  - 사용할 품사만 선택 (명사, 동사, 형용사)
  - 불용어 제거"""

okt = Okt()
# 불용어 가져오기
with open(
    "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    basic_stopwords = set(w.strip() for w in f if w.strip())

possible_pos = ["Noun", "Verb", "Adjective"]


def preprocess(text):
    str_reg = re.sub(r"[^가-힝0-9a-zA-Z\s]", "", text).lower()
    pos = okt.pos(str_reg, norm=True, stem=True, join=True)
    pos = [word.split("/") for word in pos]
    filtered_pos = [
        word
        for word, tag in pos
        if word and word not in basic_stopwords and tag in possible_pos
    ]
    return filtered_pos


# 5. TfdifVectorizer()로 훈련/테스트 데이터 벡터화
vectorizer = TfidfVectorizer(
    tokenizer=preprocess,
    token_pattern=None,  # 기본 정규식 토큰 분리 끔
    ngram_range=(1, 2),
)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# 6. 모델 훈련(LogisticRegression)
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train_vec, y_train)

# 7. 모델 테스트 검증 점수 출력
y_pred = clf.predict(x_test_vec)
print("y_pred", y_pred)

print(classification_report(y_test, y_pred, digits=3))


# 문장 받아서 긍정 부정 출력
test_sents = [
    "와 진짜 너무 재밌고 감동적이었다",
    "스토리가 너무 지루하고 시간 아깝네",
    "배우 연기는 좋았는데 결말이 별로다",
    "미친 영화다... 올해 최고",
    "연기 연습좀 해야할듯",
    "친구 추천해줘야 겠다",
]


def predict_sentences(sent_list):
    vec = vectorizer.transform(sent_list)
    preds = clf.predict(vec)
    # 긍정 확률 (1일 확률)
    probs = clf.predict_proba(vec)[:, 1]

    for sent, label, prob in zip(sent_list, preds, probs):
        print("문장:", sent)
        print("예측 라벨:", label, "(1=긍정, 0=부정)")
        print(f"긍정 확률: {prob:.3f}")
        print()


predict_sentences(test_sents)
