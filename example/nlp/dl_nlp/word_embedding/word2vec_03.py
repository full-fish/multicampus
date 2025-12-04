from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 가변 길이 텐서를 배치로 만들 때 길이를 맞추기 위해 패딩하는 함수
from torch.nn.utils.rnn import pad_sequence

# -----------------------------------------------------------
# 1) 데이터 및 Word2Vec 학습 준비
# -----------------------------------------------------------

# 예제용 문장 (토큰화된 리스트)과 감성 라벨 (1=긍정, 0=부정)
sentences = [
    ["이", "영화", "정말", "최고", "였다"],  # 1
    ["배우", "연기", "가", "너무", "좋다"],  # 1
    ["스토리", "가", "지루하고", "별로", "였다"],  # 0
    ["내용", "이", "지루하다"],  # 0
    ["음악", "이", "감동적이고", "최고", "였다"],  # 1
    ["연출", "이", "허술하고", "지루하다"],  # 0
]
labels = [1, 1, 0, 0, 1, 0]

# Word2Vec 모델 학습 설정 및 실행
model = Word2Vec(
    sentences,
    vector_size=50,  # 임베딩 벡터의 차원 (embed_dim)
    window=3,  # 컨텍스트 윈도우 크기
    min_count=1,  # 최소 단어 출현 빈도
    sg=1,  # Skip-gram 방식 사용
    workers=1,
    epochs=200,
)

# 학습된 Word2Vec 단어장 정보 추출
word_index = model.wv.key_to_index
print("\nword_index\n", word_index)
index_word = model.wv.index_to_key
print("\nindex_word\n", index_word)

vocab_size = len(word_index)  # 단어장의 크기 (패딩 제외)
print("\nvocab_size\n", vocab_size)

embed_dim = model.vector_size  # 임베딩 차원 (50)
print("\nembed_dim\n", embed_dim)

# 학습된 Word2Vec 벡터 (가중치) 추출
pretrained_weights = model.wv.vectors

# 패딩 인덱스 정의: 기존 단어장 크기 다음의 인덱스를 사용
PAD_IDX = vocab_size
vocab_size_with_pad = vocab_size + 1

# 패딩 인덱스를 포함하도록 가중치 행렬을 확장 (Word2Vec 벡터 + 패딩 벡터)
extended_weights = np.zeros((vocab_size_with_pad, embed_dim), dtype=np.float32)
# 기존 Word2Vec 가중치를 복사
extended_weights[:vocab_size, :] = pretrained_weights
# 패딩 인덱스에 해당하는 벡터는 0.0으로 초기화
extended_weights[PAD_IDX, :] = 0.0


# -----------------------------------------------------------
# 2) PyTorch Dataset 및 DataLoader 정의
# -----------------------------------------------------------


class PaddedTextDataset(Dataset):
    """
    토큰화된 문장과 라벨을 받아 Word2Vec 인덱스 텐서와 라벨 텐서를 반환하는 Dataset
    """

    def __init__(self, tokenized_sentences, labels, word_index):
        super().__init__()
        self.sentences = tokenized_sentences
        self.labels = labels
        self.word_index = word_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokens = self.sentences[index]
        y = self.labels[index]

        # 단어 리스트를 Word2Vec 인덱스 리스트로 변환
        idxs = [self.word_index[w] for w in tokens]
        # 인덱스 리스트를 PyTorch 텐서로 변환
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        label_tensor = torch.tensor(y, dtype=torch.float32)
        return idx_tensor, label_tensor


def collate_fn_with_pad(batch):
    """
    DataLoader에서 사용될 콜레이트 함수: 배치 내 가변 길이 시퀀스에 패딩을 추가
    """
    # zip(*batch)를 사용하여 문장 인덱스 텐서와 라벨 텐서를 분리
    sequences, labels = zip(*batch)

    # pad_sequence를 사용하여 배치 내 텐서들을 패딩하여 크기를 통일
    # batch_first=True: [배치 크기, 시퀀스 길이] 형태로 만듦
    # padding_value=PAD_IDX: 패딩에 사용할 토큰 인덱스를 지정
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)

    # 라벨들을 하나의 텐서로 쌓아올림
    labels = torch.stack(labels)

    # [패딩된 시퀀스 텐서, 라벨 텐서] 반환
    return padded_seqs, labels


# -----------------------------------------------------------
# 3) PyTorch 모델 정의 (PaddedSentClassifier)
# -----------------------------------------------------------


class PaddedSentClassifier(nn.Module):
    """
    패딩된 문장 배치를 처리하고 평균 풀링으로 분류하는 신경망 모델
    """

    def __init__(self, embedding: nn.Embedding, pad_idx: int):
        super().__init__()
        self.embedding = embedding
        self.pad_idx = pad_idx
        # 임베딩 차원을 입력으로, 1차원 출력을 내는 완전 연결 레이어
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_batch):
        # idx_batch: [배치 크기, 시퀀스 길이] (토큰 인덱스)

        # 1. 임베딩 벡터 생성: [배치 크기, 시퀀스 길이, 임베딩 차원]
        emb = self.embedding(idx_batch)

        # 2. 마스크 생성: 패딩이 아닌 위치는 1.0, 패딩 위치는 0.0인 텐서 [배치 크기, 시퀀스 길이]
        mask = (idx_batch != self.pad_idx).float()

        # 3. 실제 길이 계산: 마스크의 합 = 실제 토큰 개수 [배치 크기, 1]
        lenghs = mask.sum(dim=1, keepdim=True)

        # 4. 임베딩 차원으로 마스크 확장: [배치 크기, 시퀀스 길이, 1]
        extended_mask = mask.unsqueeze(dim=2)

        # 5. 마스킹된 임베딩 생성: 패딩 위치의 임베딩을 0으로 만듦 [배치 크기, 시퀀스 길이, 임베딩 차원]
        masked_emb = emb * extended_mask

        # 6. 합산 풀링 (Sum Pooling): 모든 시퀀스 차원(dim=1)을 따라 합산 [배치 크기, 임베딩 차원]
        sum_emb = masked_emb.sum(dim=1)

        # 7. 평균 풀링 (Average Pooling): 합을 실제 길이로 나누어 문장 벡터 생성
        # lenghs.clamp(min=1.0)는 길이가 0인 경우(일반적으로 발생하지 않음) 0으로 나누는 것을 방지
        sent_vec = sum_emb / lenghs.clamp(min=1.0)  # [배치 크기, 임베딩 차원]

        # 8. 분류 (Fully Connected Layer): 문장 벡터를 입력으로 로짓 계산
        # squeeze(1)로 불필요한 차원 제거 (출력: [배치 크기])
        logits = self.fc(sent_vec).squeeze(1)
        return logits


# -----------------------------------------------------------
# 4) 모델 학습 함수 정의
# -----------------------------------------------------------


def train_model(model, loader, num_epochs=50, lr=0.01):
    """
    모델 학습을 수행하는 함수
    """
    # 손실 함수 정의: Sigmoid와 이진 교차 엔트로피를 결합 (안정적)
    criterion = nn.BCEWithLogitsLoss()

    # 옵티마이저 정의: 모델의 모든 학습 가능한 파라미터를 대상으로 Adam 사용
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        model.train()  # 모델을 학습 모드로 설정

        for x, y in loader:
            optimizer.zero_grad()  # 이전 스텝의 그래디언트 초기화
            logits = model(x)  # Forward Pass (예측 로짓)
            loss = criterion(logits, y)  # 손실 계산
            loss.backward()  # Backward Pass (그래디언트 계산)
            optimizer.step()  # 옵티마이저 스텝 (파라미터 업데이트)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:02d}] loss = {avg_loss:.4f}")


# -----------------------------------------------------------
# 5) 모델 인스턴스화 및 학습 실행
# -----------------------------------------------------------

# Dataset 및 DataLoader 인스턴스 생성
padded_dataset = PaddedTextDataset(sentences, labels, word_index)
padded_loader = DataLoader(
    padded_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_with_pad
)

# nn.Embedding 레이어 생성 (패딩 포함된 단어장 크기)
padded_embedding = nn.Embedding(vocab_size_with_pad, embed_dim)

# **사전 학습된 가중치(Word2Vec) 로드**
# torch.no_grad() 블록 안에서 수행하여 그래디언트 계산 없이 가중치 복사
with torch.no_grad():
    # numpy 배열을 torch 텐서로 변환 후, Embedding 레이어의 weight에 복사
    padded_embedding.weight.copy_(torch.from_numpy(extended_weights))

# 가중치 학습 가능 설정 (Word2Vec 임베딩을 Fine-tuning)
padded_embedding.weight.requires_grad = True

# 최종 분류 모델 인스턴스 생성 및 학습 시작
padded_model = PaddedSentClassifier(padded_embedding, pad_idx=PAD_IDX)
print("\n모델 학습 시작:")
train_model(padded_model, padded_loader, num_epochs=30, lr=0.01)

# 학습 로그 출력 (예시)
# [Epoch 01] loss = 0.6931
# [Epoch 10] loss = 0.6000
# [Epoch 20] loss = 0.5000
# [Epoch 30] loss = 0.4000
