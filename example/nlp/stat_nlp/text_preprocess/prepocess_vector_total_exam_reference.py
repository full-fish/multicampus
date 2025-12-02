# 1. 라이브러리 임포트
# ============================================================
import re
import numpy as np
import pandas as pd

from konlpy.tag import Okt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(
    "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/document_vector/movie_reviews.csv"
)
df = df.dropna()

_, df_sample = train_test_split(
    df,
    test_size=10000,
    stratify=df["label"],
    shuffle=True,  # 기본값이라 사실 안 써도 됨
    # random_state=33
)
X = df_sample["document"]
y = df_sample["label"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# ============================================================
# 3. 한글 전처리 함수 정의
#    - 정제/정규화
#    - 토큰화(형태소 분석)
#    - 불용어 제거
#    - 어간(또는 표제어) 처리 (Okt의 stem=True 사용)
# ============================================================
okt = Okt()

# 아주 간단한 한글 불용어 목록 (필요에 따라 확장 가능)
korean_stopwords = set(
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


def preprocess_text(text: str) -> list:
    # 1) 소문자 변환 (영어가 껴 있을 수도 있으니)
    text = text.lower()

    # 2) 한글/영문/숫자/공백 외 특수문자 제거
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

    # 3) 형태소 분석 + 품사 태깅 (stem=True: 기본형으로)
    morphs = okt.pos(text, stem=True)

    # 4) 사용할 품사만 선택 (명사, 동사, 형용사)
    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            # 불용어 제거
            if word not in korean_stopwords and len(word) > 1:
                tokens.append(word)

    # 5) 토큰들을 공백으로 다시 결합 → TfidfVectorizer에 넣기 좋게 문자열로
    # return " ".join(tokens)
    return tokens


# 전처리 결과 확인 (상위 3개만)
print("=== 전처리 예시 ===")
for t in x_train[:3]:
    print("원문 :", t)
    print("전처리:", preprocess_text(t))
    print("-" * 40)

# ============================================================
# 4. 전체 데이터 전처리 + TF-IDF 벡터화
# ============================================================
processed_texts = [preprocess_text(t) for t in X]

tfidf = TfidfVectorizer(
    token_pattern=None,
    tokenizer=preprocess_text,
    max_features=1000,
    ngram_range=(1, 2),  # 유니그램 + 바이그램
)

x_train_vec = tfidf.fit_transform(x_train)
x_test_vec = tfidf.transform(x_test)


# print("\nTF-IDF 행렬 shape:", x_train_vec)
print("단어집 일부:", tfidf.get_feature_names_out()[:20])

# ============================================================
# 5. 학습/검증 데이터 분리
# ============================================================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# ============================================================
# 6. 분류 모델(로지스틱 회귀) 학습 및 평가
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train_vec, y_train)

y_pred = clf.predict(x_test_vec)

print("\n=== 모델 평가 ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))


# ============================================================
# 7. 새 리뷰 문장에 대한 감성 예측 함수
# ============================================================
def predict_sentiment(text: str):

    # 2) 기존 TF-IDF 벡터라이저로 변환 (fit X, transform O)
    vec = tfidf.transform([text])
    # 3) 예측
    pred = clf.predict(vec)[0]
    print("pred", pred)

    prob = clf.predict_proba(vec)[0][pred]

    label = "긍정" if pred == 1 else "부정"
    return label, prob  # 전처리된 문장도 같이 반환해서 확인해보자


# ============================================================
# 8. 예측 테스트
# ============================================================
test_texts = [
    "연기가 너무 좋고 스토리가 감동적이어서 또 보고 싶어요.",
    "시간이 아까울 정도로 지루하고 재미가 없었어요.",
    "그냥 그랬어요. 크게 인상적인 부분은 없었습니다.",
]

print("\n=== 새 문장 예측 ===")
for t in test_texts:
    label, prob = predict_sentiment(t)
    print(f"원문: {t}")
    print(f"예측: {label} (확률: {prob:.3f})")
    print("-" * 40)
