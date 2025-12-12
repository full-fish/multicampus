import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 1. 원본 텍스트 데이터와 레이블
raw_texts = [
    "영화가 정말 재미있고 감동적이었어요",  # 긍정 (1)
    "스토리가 지루하고 시간 낭비였어요",  # 부정 (0)
    "배우 연기가 훌륭하고 음악도 좋았어요",  # 긍정 (1)
    "내용이 별로고 전개가 너무 느렸어요",  # 부정 (0)
    "정말 최고의 영화였어요 또 보고 싶어요",  # 긍정 (1)
    "연출이 엉성하고 집중이 안 됐어요",  # 부정 (0)
]
raw_labels = [1, 0, 1, 0, 1, 0]


# 2. 간단한 토큰화 함수 정의
def simple_tokenize(text: str):
    # 특수문자를 공백으로 치환 (숫자, 영문, 한글, 공백 제외)
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    # 2개 이상의 연속된 공백을 하나로 치환하고 양쪽 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    # 공백 기준으로 토큰 분리
    tokens = text.split()
    return tokens


# 3. 토큰화 실행
tokenized_sentences = [simple_tokenize(s) for s in raw_texts]
print("토큰화된 문장들:\n", tokenized_sentences)

# 4. 특수 토큰 정의
PAD_TOKEN = "[PAD]"  # 패딩 토큰
UNK_TOKEN = "[UNK]"  # 미등록 단어 토큰

# 5. 단어 인덱스 딕셔너리 (word2idx) 초기화
word2idx = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
}
# Counter 활용해서 단어 빈도수 계산 및 word2idx 완성
counter = Counter()
for tokens in tokenized_sentences:
    counter.update(tokens)  # 모든 토큰의 빈도수를 계산

# 빈도수가 높은 순서대로 딕셔너리에 추가 (PAD, UNK 제외)
for word, _ in counter.most_common():
    if word not in word2idx:
        word2idx[word] = len(word2idx)  # 현재 딕셔너리 크기를 새 단어의 인덱스로 할당

vocab_size = len(word2idx)  # 전체 단어장의 크기
print("\nword2idx (단어 -> 인덱스):\n", word2idx)


# 6. 인덱스 단어 딕셔너리 (idx2word) 생성
idx2word = {idx: word for word, idx in word2idx.items()}
print("\nidx2word (인덱스 -> 단어):\n", idx2word)


# 7. 문장을 인덱스화하고 패딩 처리하는 함수
def encode_tokens(tokens, word2idx, max_len):
    # 단어를 인덱스로 변환. 단어장에 없으면 UNK_TOKEN 인덱스 사용
    idxs = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]
    idxs_len = len(idxs)

    # 문장 길이가 max_len보다 짧으면 PAD_TOKEN으로 채움
    if idxs_len < max_len:
        idxs += [word2idx[PAD_TOKEN]] * (max_len - idxs_len)
    # 문장 길이가 max_len보다 길면 잘라냄
    elif idxs_len > max_len:
        idxs = idxs[:max_len]

    return idxs


# 8. 최대 시퀀스 길이 계산 및 인코딩
MAX_LEN = max(len(tokens) for tokens in tokenized_sentences)
print(f"\n최대 시퀀스 길이 (max_len): {MAX_LEN}")
encode_sentences = [
    encode_tokens(tokens, word2idx, MAX_LEN) for tokens in tokenized_sentences
]
# print("\n인코딩 및 패딩된 문장들:\n", encode_sentences) # 확인용

# 9. 데이터셋 생성 (PyTorch 텐서로 변환)
X = torch.tensor(encode_sentences, dtype=torch.long)  # 입력 데이터 (문장 인덱스)
y = torch.tensor(raw_labels, dtype=torch.float32).unsqueeze(
    1
)  # 레이블 (이진 분류를 위해 (N, 1) 형태로 변환)

dataset = TensorDataset(X, y)

# 10. 데이터 로더 생성
BATCH_SIZE = 2
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 11. LSTM 기반 감성 분류 모델 정의
class LSTMSentimentClassifier(nn.Module):
    # 모델 초기화: 필요한 파라미터 정의
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, pad_idx=0):
        super().__init__()

        # 임베딩 레이어: 단어 인덱스를 벡터로 변환 (패딩 인덱스는 학습 제외)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 단어장의 크기
            embedding_dim=embed_dim,  # 임베딩 벡터의 차원
            padding_idx=pad_idx,  # 패딩 토큰의 인덱스 (0)
        )

        # LSTM 레이어: 시퀀스 데이터를 처리하고 문맥 정보를 추출
        self.lstm = nn.LSTM(
            input_size=embed_dim,  # 입력 차원 (임베딩 차원)
            hidden_size=hidden_size,  # 은닉 상태의 차원
            num_layers=num_layers,  # LSTM 레이어의 개수
            batch_first=True,  # 입력 텐서의 첫 번째 차원이 batch 크기임을 명시
            bidirectional=False,  # 단방향 LSTM 사용
        )

        # 출력 레이어: LSTM의 최종 은닉 상태를 입력받아 최종 예측값 (로짓)을 출력
        # hidden_size 차원의 벡터를 1차원 로짓으로 변환 (이진 분류)
        self.fc = nn.Linear(hidden_size, 1)

    # 순전파 (Forward propagation) 정의
    def forward(self, input_ids):  # input_ids: (batch, seq_len)

        # 1. 임베딩: 단어 인덱스를 임베딩 벡터로 변환
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # 2. LSTM 통과
        # output: 모든 시점의 은닉 상태 (batch, seq_len, hidden_size * num_directions)
        # (h_n, c_n): 마지막 시점의 은닉 상태와 셀 상태 (num_layers*num_directions, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(emb)

        # 3. 최종 은닉 상태 추출 및 사용
        # 단방향 LSTM이므로 h_n의 첫 번째 레이어 (0)를 사용
        # h_n shape: (1, batch, hidden_size) -> h_n[0] shape: (batch, hidden_size)
        last_hidden = h_n[0]

        # 4. 선형 레이어 (FC) 통과
        # last_hidden을 사용하여 최종 로짓 계산
        # fc(last_hidden) shape: (batch, 1)
        logits = self.fc(last_hidden)

        return logits


# 12. 모델 파라미터 및 모델 객체 생성
embed_dim = 16  # 임베딩 벡터 차원
hidden_size = 32  # LSTM 은닉 상태 차원
num_layers = 1  # LSTM 레이어 개수
pad_idx = word2idx[PAD_TOKEN]  # 패딩 토큰 인덱스 (0)

# 모델 인스턴스화
model = LSTMSentimentClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    pad_idx=pad_idx,
)

# 13. 손실 함수 (Loss function) 및 옵티마이저 (Optimizer) 정의
# 이진 분류 (0 또는 1)이므로 BCEWithLogitsLoss 사용
criterion = nn.BCEWithLogitsLoss()
# Adam 옵티마이저 사용
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, loader, criterion, optimizer, num_epochs=20):
    model.train()
    print("\n모델 학습 시작...")

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            # 손실 누적 (배치 크기 고려)
            total_loss += loss.item() * batch_X.size(0)

            # 정확도 계산
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_X.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        # ----------------------------------------------------

        print(
            f"[Epoch {epoch:02d}/{num_epochs}] Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}"
        )

    print("모델 학습 완료.")


print("\n=== 모델 학습 시작 ===")
train_model(model, train_loader, criterion, optimizer, num_epochs=20)


def predict_sentiment(text):
    model.eval()

    # 1. 토큰화
    tokens = simple_tokenize(text)

    # 2. 인덱스화 및 패딩 (MAX_LEN 사용)
    # 인덱스 리스트를 텐서로 변환하고 배치 차원 추가 (1, seq_len)
    input_idxs = encode_tokens(tokens, word2idx, MAX_LEN)
    print("\ninput_idxs\n", input_idxs)

    input_tensor = torch.tensor(input_idxs, dtype=torch.long).unsqueeze(0)

    # 학습이 아닐땐 미분이 필요없어서 with문 사용
    with torch.no_grad():
        # 3. 모델 예측 (로짓 획득)
        logits = model(input_tensor)

        # 4. 확률 및 라벨 변환
        prob = torch.sigmoid(logits).item()
        label = prob >= 0.5

    # 5. 결과 반환: 라벨(0 또는 1), 확률, 토큰
    return int(label), prob, tokens


test_setences = ["스토리가 정말 지루하고 재미없었어요", "완전 감동적이고 눈물났어요"]

for s in test_setences:
    label, prob, tokens = predict_sentiment(s)
    print(f"문장 : {s}")
    print(f"토큰: {tokens}")
    print(f"예측 라벨: , {label}(1:긍정, 0:부정), 확률: {prob:.4f}")
