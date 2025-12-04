from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1) 예제용 문장 + 감성 라벨 (1=긍정, 0=부정)
sentences = [
    ["이", "영화", "정말", "최고", "였다"],  # 1
    ["배우", "연기", "가", "너무", "좋다"],  # 1
    ["스토리", "가", "지루하고", "별로", "였다"],  # 0
    ["내용", "이", "지루하다"],  # 0
    ["음악", "이", "감동적이고", "최고", "였다"],  # 1
    ["연출", "이", "허술하고", "지루하다"],  # 0
]
labels = [1, 1, 0, 0, 1, 0]

# Word2Vec 모델 학습 설정
model = Word2Vec(
    sentences,
    vector_size=50,  # 임베딩 벡터의 차원 (embed_dim)
    window=3,  # 컨텍스트 윈도우 크기
    min_count=1,  # 최소 단어 출현 빈도
    sg=1,  # Skip-gram 방식 사용 (CBOW는 0)
    workers=1,  # 병렬 처리에 사용할 CPU 코어 개수
    epochs=200,  # 전체 말뭉치를 200번 반복 학습
)

# 단어장 정보 추출
word_index = model.wv.key_to_index
print("\nword_index\n", word_index)
index_word = model.wv.index_to_key
print("\nindex_word\n", index_word)

vocab_size = len(word_index)  # 단어장의 크기
print("\nvocab_size\n", vocab_size)

embed_dim = model.vector_size  # 임베딩 차원 (50)
print("\nembed_dim\n", embed_dim)

# 학습된 Word2Vec 벡터 (가중치) 추출
pretrained_weights = model.wv.vectors
# Word2Vec 벡터는 (단어장 크기, 임베딩 차원) 형태를 가짐 (예: (19, 50))
print("\npretrained_weights\n", pretrained_weights)
print("\npretrained_weights.len\n", len(pretrained_weights))


class SimpleTextDataset(Dataset):
    # PyTorch Dataset 클래스를 상속받아 텍스트 데이터를 단어 인덱스 텐서로 변환
    def __init__(self, sentences, labels, word_index):
        super().__init__()
        self.sentences = sentences  # 원본 문장 리스트
        self.labels = labels  # 라벨 리스트
        self.word_index = word_index  # 단어-인덱스 매핑

    def __len__(self):
        # 전체 데이터(문장)의 개수를 반환
        return len(self.sentences)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 (문장 인덱스 텐서, 라벨 텐서) 쌍을 반환
        sentence = self.sentences[idx]
        label = self.labels[idx]
        # 문장을 Word2Vec 인덱스 리스트로 변환
        indexed_sentence = [self.word_index[word] for word in sentence]

        # PyTorch 텐서로 변환 (문장: Long 타입, 라벨: Float32 타입)
        return torch.tensor(indexed_sentence, dtype=torch.long), torch.tensor(
            label, dtype=torch.float32
        )


def collate_fn(batch):
    # batch는 [(indexed_sentence_tensor, label_tensor), ...] 형태의 튜플 리스트입니다.

    # 1. 문장 텐서와 라벨 텐서를 분리
    sentences_tensors = [item[0] for item in batch]
    labels_tensors = [item[1] for item in batch]

    # 2. 패딩 적용: 배치 내 가장 긴 문장 길이에 맞춰 짧은 문장들을 0으로 채움
    # batch_first=True: (배치 크기, 문장 길이) 형태로 텐서를 구성
    padded_sentences = pad_sequence(
        sentences_tensors, batch_first=True, padding_value=0
    )

    # 3. 라벨 텐서들을 쌓아 올림
    labels_tensor = torch.stack(labels_tensors)

    return padded_sentences, labels_tensor


dataset = SimpleTextDataset(sentences, labels, word_index)

# 2. DataLoader를 생성할 때 collate_fn 인자를 정확하게 전달
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


class SimpleSentClassifier(nn.Module):
    # Word2Vec 임베딩을 활용한 간단한 문장 분류기 모델
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        # 미리 정의된 (사전 학습된 가중치가 있는) 임베딩 레이어를 저장
        self.embedding = embedding

        # 문장 분류를 위한 선형 레이어 (Fully Connected Layer) 정의
        # 입력 차원: 임베딩 차원 (50) | 출력 차원: 1 (이진 분류의 로짓)
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_tensor):
        # 입력: 단어 인덱스 텐서 (배치 크기, 문장 길이)

        # 배치 크기가 1인 경우, 차원을 (1, 문장 길이)로 확장
        if idx_tensor.dim() == 1:
            idx_tensor = idx_tensor.unsqueeze(0)

        # 1. 임베딩 레이어를 통과하여 단어 임베딩 획득
        emb = self.embedding(idx_tensor)
        # idx_tensor: (batch_size, seq_len) (예: 1, 5)
        # => emb: (batch_size, seq_len, embed_dim) (예: 1, 5, 50)

        # 2. 문장 벡터 생성: 문장 길이 차원(dim=1)을 따라 평균 (Mean Pooling)
        sent_vec = emb.mean(dim=1)  # (batch_size, embed_dim) (예: 1, 50)

        # 3. 선형 레이어를 통과하여 로짓 계산 후, 마지막 차원(1)을 제거 (.squeeze(1))
        # 출력: (batch_size) 형태의 로짓 텐서
        logit = self.fc(sent_vec).squeeze(1)

        return logit


def build_embedding_from_w2v(pretrained_weights, freeze: bool) -> nn.Embedding:
    # Word2Vec 가중치로 초기화된 nn.Embedding 레이어를 생성하는 유틸리티 함수

    # 가중치 형태에서 vocab_size와 embed_dim을 추출
    vocab_size, embed_dim = pretrained_weights.shape

    # nn.Embedding 레이어 초기화
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    # 그래디언트 계산 없이 (torch.no_grad()) Word2Vec 가중치를 임베딩 레이어에 복사
    with torch.no_grad():
        embedding.weight.copy_(torch.from_numpy(pretrained_weights))

    # freeze가 True면 가중치 고정 (학습하지 않음)
    embedding.weight.requires_grad = not freeze

    return embedding


def train_model(model, loader, num_epochs=50, lr=0.01):
    # 모델 학습 함수

    # 손실 함수 정의: 이진 분류를 위한 BCEWithLogitsLoss (Sigmoid 내장)
    criterion = nn.BCEWithLogitsLoss()

    # 옵티마이저 정의: requires_grad=True인 파라미터만 업데이트 대상으로 지정 (freeze된 임베딩 제외)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # 에포크 루프
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0

        # 배치 루프
        for x, y in loader:
            optimizer.zero_grad()  # 그래디언트 초기화
            logits = model(x)  # Forward Pass (예측 로짓 계산)
            loss = criterion(logits, y)  # 손실 계산
            loss.backward()  # Backward Pass (그래디언트 계산)
            optimizer.step()  # 옵티마이저 스텝 (파라미터 업데이트)

            total_loss += loss.item()  # 손실 누적

        # 에포크 종료 후 평균 손실 계산
        avg_loss = total_loss / len(loader)

        # 학습 로그 출력
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:02d}] loss = {avg_loss:.4f}")


print("\n==================================")
print("1) 임베딩 고정 freeze 버전")
print("==================================")

frozen_embedding = build_embedding_from_w2v(pretrained_weights, freeze=True)
model_frozen = SimpleSentClassifier(frozen_embedding)

print("임베딩 requires_grad : ", model_frozen.embedding.weight.requires_grad)
print("FC layer requires_grad : ", model_frozen.fc.weight.requires_grad)

train_model(model_frozen, loader, num_epochs=50, lr=0.01)

print("\n==================================")
print("1) 임베딩 미세조정(fine-tuning) 버전")
print("==================================")

frozen_embedding = build_embedding_from_w2v(pretrained_weights, freeze=False)
model_frozen = SimpleSentClassifier(frozen_embedding)

print("임베딩 requires_grad : ", model_frozen.embedding.weight.requires_grad)
print("FC layer requires_grad : ", model_frozen.fc.weight.requires_grad)

train_model(model_frozen, loader, num_epochs=50, lr=0.01)
