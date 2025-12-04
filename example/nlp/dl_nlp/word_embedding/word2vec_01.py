from gensim.models import Word2Vec
import torch
import torch.nn as nn

sentences = [
    ["이", "영화", "정말", "최고", "였다"],
    ["배우", "연기", "가", "최고", "이다"],
    ["스토리", "가", "지루하다"],
    ["내용", "이", "지루하고", "별로", "였다"],
    ["이", "브랜드", "디자인", "이", "세련되다"],
    ["이", "브랜드", "가격", "은", "비싸다"],
]

model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    workers=1,  # 병렬 처리에 사용할 CPU 코어 개수
    epochs=100,  # 전체 말뭉치를 몇 번 반복해서 학습할지
)

# print("단어 '최고' 벡터 크기 : ", model.wv["최고"].shape)
# print("벡터 앞 5개 : ", model.wv["최고"][:5])

# print("\n[단어 '최고'와 비슷한 단어 Top-5]")
# for w, score in model.wv.most_similar("최고", topn=5):
#     print(f"{w:10s} 유사도:{score:.3f}")

# print("\n단어간 유사도")
# print("최고 vs 별로 : ", model.wv.similarity("최고", "별로"))
# print("지루하다 vs 별로 : ", model.wv.similarity("지루하다", "별로"))

word_index = model.wv.key_to_index
index_word = model.wv.index_to_key

# print(word_index)
# print(index_word)

vocab_size = len(word_index)
embed_dim = model.vector_size

print("vocab_size : ", vocab_size)
print("embed_dim : ", embed_dim)

pretrained_weights = model.wv.vectors
print("pretrained_weights.shape : ", pretrained_weights.shape)

# 임베딩 레이어 생성
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

word1 = "최고"
word2 = "지루하다"

idx1 = word_index[word1]
idx2 = word_index[word2]

idx_tensor = torch.tensor([idx1, idx2])
emb_vec = embedding(idx_tensor)

print("\n Pytorch Embedding으로 가져온 벡터 ")
print(f"{word1} 임베딩 벡터 : ", emb_vec[0][:5])
print(f"{word2} 임베딩 벡터 : ", emb_vec[1][:5])
