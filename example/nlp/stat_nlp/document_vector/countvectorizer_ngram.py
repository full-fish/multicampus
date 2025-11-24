from sklearn.feature_extraction.text import CountVectorizer

docs = ["오늘 날씨 정말 좋다", "오늘 기분 정말 좋다", "오늘은 기분이 좋지 않다"]
# 1) unigram
cv_uni = CountVectorizer(ngram_range=(1, 1))
X_uni = cv_uni.fit_transform(docs)

print("unigram vocab:", cv_uni.get_feature_names_out())
print(X_uni.toarray())
# 2) bigram
cv_bi = CountVectorizer(ngram_range=(2, 2))
X_bi = cv_bi.fit_transform(docs)
print("bigram vocab:", cv_bi.get_feature_names_out())
print(X_bi.toarray())
# 3) uni+bi 같이
cv_uni_bi = CountVectorizer(ngram_range=(1, 2))
X_uni_bi = cv_uni_bi.fit_transform(docs)
print("uni+bi vocab:", cv_uni_bi.get_feature_names_out())
print(X_uni_bi.toarray())
