from sklearn.feature_extraction.text import CountVectorizer

docs = ["오늘 날씨 정말 좋다", "오늘 기분 정말 좋다", "오늘은 기분이 좋지 않다"]
# 기본 설정: 공백 기준 토큰화, 소문자 변환(영문 기준)
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(docs)  # 희소행렬(sparse matrix)
print("X.shape", X.shape)  # (문서 수, 단어 수)
print("X", X)

# 단어집(어휘) 확인
print(vectorizer.get_feature_names_out())

# 희소행렬 → 밀집행렬로 보기
print(X.toarray())


vectorizer = CountVectorizer(
    max_features=1000,  # 단어 수 상한
    min_df=2,  # 최소 2개 문서에 등장한 단어만 사용
    stop_words=["오늘", "정말"],  # 제거할 단어들
)
