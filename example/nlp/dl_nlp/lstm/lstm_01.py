import torch
import torch.nn as nn

# 1. 하이퍼파라미터 및 데이터 정의

vocab_size = 4  # 전체 어휘의 개수 (단어 4개: 나는, 밥을, 라면을, 먹었다)
embed_dim = 4  # 단어를 표현할 임베딩 벡터의 차원 (input_size와 동일해야 함)
hidden_size = 3  # LSTM의 은닉 상태(Hidden state) 벡터의 크기
num_layers = 1  # 사용할 LSTM 레이어의 개수 (단층 LSTM)

# 단어와 인덱스를 매핑하는 사전(Vocabulary)
vocab = {"나는": 0, "밥을": 1, "라면을": 2, "먹었다": 3}

# 단어 인덱스로 표현된 두 개의 시퀀스(문장)
sent1 = [vocab["나는"], vocab["밥을"], vocab["먹었다"]]  # [0, 1, 3]
sent2 = [vocab["나는"], vocab["라면을"], vocab["먹었다"]]  # [0, 2, 3]

# 배치(Batch) 생성: (batch_size=2, seq_len=3)
# batch_size=2: 문장 2개
# seq_len=3: 각 문장의 길이 3
batch = torch.tensor([sent1, sent2])  # shape: (2, 3)
print("batch:", batch)
print("batch.shape:", batch.shape)
# ---

# 2. 임베딩 레이어 정의 및 통과

# nn.Embedding: 단어 인덱스를 밀집된 벡터(임베딩 벡터)로 변환
embedding = nn.Embedding(
    num_embeddings=vocab_size,  # 임베딩할 단어의 개수
    embedding_dim=embed_dim,  # 임베딩 벡터의 차원
)

# 단어 인덱스 → 임베딩 벡터로 변환
# batch (2, 3) → emb (2, 3, 4)
emb = embedding(batch)  # shape: (batch_size, seq_len, embed_dim)
print("\n[임베딩 통과 후]")
print("emb.shape:", emb.shape)  # (2, 3, 4)
# ---

# 3. LSTM 레이어 정의 및 통과

# nn.LSTM: 순환 신경망 레이어 정의
lstm = nn.LSTM(
    input_size=embed_dim,  # 각 타임스텝 입력 벡터 크기 (emb의 마지막 차원: 4)
    hidden_size=hidden_size,  # 은닉 상태 벡터 크기: 3
    num_layers=num_layers,  # LSTM 층 개수: 1
    batch_first=True,  # 입력/출력 텐서 형태를 (batch, seq_len, feature)로 설정
)

# LSTM 통과:
# lstm(입력 텐서) → (출력, (최종 은닉 상태, 최종 셀 상태))
output, (h_n, c_n) = lstm(emb)

# 결과 출력
print("\n[LSTM 결과]")
# output: 모든 시점(단어)에서의 Hidden state (h_t) 출력
print("output.shape:", output.shape)  # (batch, seq_len, hidden_size) → (2, 3, 3)
# h_n: 최종 시점(마지막 단어)의 Hidden state (h_T)
print("h_n.shape:", h_n.shape)  # (num_layers, batch, hidden_size) → (1, 2, 3)
# c_n: 최종 시점(마지막 단어)의 Cell state (c_T)
print("c_n.shape:", c_n.shape)  # (num_layers, batch, hidden_size) → (1, 2, 3)

print("\noutput:", output)
print("\nh_n:", h_n)
print("\nc_n:", c_n)
