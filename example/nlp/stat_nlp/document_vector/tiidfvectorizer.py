from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["오늘 날씨 날씨 정말 좋다", "오늘 기분 정말 좋다", "오늘은 기분이 좋지 않다"]

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(docs)
print("vocab : ", tfidf.get_feature_names_out())
print("shape : ", X_tfidf.shape)
print("X_tfidf : \n", X_tfidf.toarray().round(3))
