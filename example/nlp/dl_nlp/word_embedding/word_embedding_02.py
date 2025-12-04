from collections import (
    Counter,
)  # 단어의 등장 횟수(빈도수)를 쉽게 계산하기 위해 Counter 모듈을 임포트합니다.
import torch  # 텐서 연산과 딥러닝 모델 구축을 위해 PyTorch 라이브러리를 임포트합니다.
import torch.nn as nn  # 신경망 모듈(레이어)을 정의하기 위해 임포트합니다.
import torch.nn.functional as F  # 코사인 유사도 계산 함수(F.cosine_similarity)를 사용하기 위해 임포트합니다.

# 1) 간단한 말뭉치 (Corpus): 분석 대상이 되는 문장 데이터입니다.
sentences = [
    "이 영화 정말 최고였어요",
    "이 배우 연기가 최고입니다",
    "내용이 지루하고 별로였어요",
    "스토리가 지루하지만 배우는 좋았어요",
]
# ... (방법 1 주석 처리) ...

#! 방법2 이게 좋음 이유: 빈도수로 오름차순이 좋음----------------------------------------

# 2) 단어집(Vocab) 및 임베딩 레이어 생성 (방법 2: 빈도수 기반)
# ----------------------------------------

# 각 문장을 공백 기준으로 토큰(단어) 리스트로 분리하여 2차원 리스트를 생성합니다. (토큰화)
tokenized = [sentence.split() for sentence in sentences]

counter = Counter()  # 단어 빈도를 셀 Counter 객체를 초기화합니다.
for sent in tokenized:
    counter.update(sent)  # 각 문장의 단어 리스트를 Counter에 넣어 빈도를 누적합니다.

vocab = {}  # 단어집 {단어: 인덱스}를 저장할 딕셔너리를 초기화합니다.
# most_common()을 사용해 빈도수가 높은 순서대로 단어를 가져와 인덱스를 부여합니다.
# (실제 NLP에서 가장 흔하게 사용되는 방식입니다.)
for word, _ in counter.most_common():
    vocab[word] = len(
        vocab
    )  # 현재 vocab의 크기를 인덱스로 사용하여 0부터 순차적으로 부여합니다.

vocab_size = len(vocab)  # 단어집의 크기(고유 단어의 총 개수)를 저장합니다.
embed_dim = 8  # 임베딩 벡터의 차원(크기)을 8로 설정합니다.

# nn.Embedding 레이어를 정의합니다.
# num_embeddings: 단어집 크기 (vocab_size)
# embedding_dim: 임베딩 벡터 차원 (embed_dim)
embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
# 임베딩 레이어의 전체 정보(하이퍼파라미터)를 출력합니다.
print("\nembed\n", embed)
# 임베딩 가중치 표(행렬 W)의 초기 랜덤 값을 출력합니다.
print("\nembed.weight\n", embed.weight)
# 임베딩 가중치 표(행렬 W)의 shape을 출력합니다. (vocab_size, embed_dim) 형태입니다.
print("\nembed.weight.shape\n", embed.weight.shape)


# 3) 문장을 단어 인덱스 시퀀스로 변환
# ----------------------------------------


# 토큰화된 문장을 단어집(vocab)의 인덱스 리스트로 변환하는 함수를 정의합니다.
def sentence_to_indices(sentence_tokens, vocab):
    # 리스트 컴프리헨션을 사용하여 각 단어를 해당 인덱스로 변환합니다.
    return [vocab[w] for w in sentence_tokens]


# 최종적으로 생성된 단어집의 내용을 확인합니다.
print("\nvocab\n", vocab)

# 토큰화된 문장의 형태를 확인합니다.
print("\ntokenized\n", tokenized)

# 전체 토큰화된 문장들을 인덱스 시퀀스(정수 리스트)로 변환합니다.
indexed_sentences = [sentence_to_indices(s, vocab) for s in tokenized]
print("indexed_sentences :", indexed_sentences)


# 4) 문장 임베딩(단어 벡터)을 얻는 함수 정의
# ----------------------------------------


# 인덱스 리스트를 입력받아 문장 임베딩 벡터(평균)를 추출하는 함수를 정의합니다.
def get_sentence_embedding(idx_list):
    # 파이썬 리스트 형태의 인덱스 리스트를 PyTorch 텐서로 변환합니다.
    idx_tensor = torch.tensor(idx_list)

    # nn.Embedding 레이어에 인덱스 텐서를 전달하여 해당하는 단어 임베딩 벡터들을 조회합니다.
    # 결과 word_embeds의 shape은 (문장 길이, embed_dim)이 됩니다.
    word_embeds = embed(idx_tensor)

    # 단어 벡터들의 평균을 계산하여 문장 전체를 대표하는 하나의 벡터를 만듭니다. (Mean Pooling)
    # dim=0 (단어 차원)을 기준으로 평균을 구하여 (embed_dim) shape의 벡터를 반환합니다.
    doc_embed = word_embeds.mean(dim=0)
    return doc_embed


# 모든 문장에 대해 임베딩 벡터를 생성하고, torch.stack을 사용하여 하나의 텐서로 합칩니다.
# (문장 임베딩의 shape이 모두 (embed_dim,)로 동일하므로 stack 사용 가능)
sentence_embeddings = torch.stack(
    [get_sentence_embedding(idex_list) for idex_list in indexed_sentences]
)
# 최종 문장 임베딩 텐서의 내용을 출력합니다. shape은 (문장 개수, embed_dim)이 됩니다.
print("\nsentence_embeddings\n", sentence_embeddings)


# 5) 문장 유사도 측정 (코사인 유사도)
# ----------------------------------------


# 두 벡터(a, b)의 코사인 유사도를 계산하고 Python float 값으로 반환하는 함수를 정의합니다.
def consine_sim(a, b):
    # F.cosine_similarity(벡터1, 벡터2, 계산할 차원).item()으로 유사도 스칼라 값을 얻습니다.
    return F.cosine_similarity(a, b, dim=0).item()


print("\n문장유사도\n")
# 모든 가능한 문장 쌍(Pair)에 대해 반복합니다.
for i in range(len(sentences)):
    # j는 i보다 큰 인덱스부터 시작하여 중복 계산을 방지합니다.
    for j in range(i + 1, len(sentences)):
        # 두 문장 임베딩 벡터의 유사도를 계산합니다.
        sim = consine_sim(sentence_embeddings[i], sentence_embeddings[j])
        # 결과를 문장과 함께 소수점 셋째 자리까지 출력합니다.
        print(f"({i}) {sentences[i]} vs ({j}) {sentences[j]} -> 유사도 : {sim:.3f}")
