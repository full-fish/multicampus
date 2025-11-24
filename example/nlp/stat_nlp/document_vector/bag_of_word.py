import numpy as np

docs = ["오늘 날씨 정말 좋다", "오늘 기분 정말 좋다", "오늘은 기분이 좋지 않다"]
# 1) 공백 기준 토큰화
tokenized_docs = [doc.split() for doc in docs]
print("tokenized_docs", tokenized_docs)
# [['오늘', '날씨', '정말', '좋다'], ...]
# 2) 전체 단어집(어휘집) 만들기
vocab = sorted({word for doc in tokenized_docs for word in doc})
print("vocab:", vocab)
# 3) 단어 → 인덱스 매핑
word_to_idx = {word: i for i, word in enumerate(vocab)}
print("word_to_idx", word_to_idx)

bow_vectors = []

for doc in tokenized_docs:
    vec = np.zeros(len(vocab), dtype=int)  # [0,0,0,0,0,0,0,0,0]
    for word in doc:
        idx = word_to_idx[word]
        vec[idx] += 1
    bow_vectors.append(vec)
print("bow_vectors", bow_vectors)
