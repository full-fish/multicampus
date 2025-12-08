import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from tqdm import tqdm  # 진행상황 바(Progress Bar) 출력용

# =============================================================================
# [STEP 1] 데이터 로드 및 전처리
# : 외부 라이브러리 함수 대신 직접 구현
# =============================================================================


# 간단한 전처리 함수 정의
def preprocess_text(text):
    # 특수문자 제거 등 필요한 전처리를 수행합니다.
    # 여기서는 간단히 문자열 변환 후 공백 기준 분리만 수행
    return str(text).split()


print("데이터 로딩 중...")
df = pd.read_csv("movie_reviews.csv")
df = df.dropna()
df["label"] = df["label"].astype(int)

# 텍스트를 토큰 리스트로 변환 (예: "이 영화" -> ["이", "영화"])
tokenized_sentences = [preprocess_text(review) for review in df["review"]]
labels = df["label"].values.astype(np.float32)

# 훈련/검증 데이터 분리
train_tokens, valid_tokens, y_train, y_valid = train_test_split(
    tokenized_sentences, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train 개수: {len(train_tokens)}, Valid 개수: {len(valid_tokens)}")


# =============================================================================
# [STEP 2] FastText 임베딩 학습 (사전 학습)
# =============================================================================

print("FastText 모델 학습 시작...")
EMBED_DIM = 100

ft_model = FastText(
    sentences=train_tokens,
    vector_size=EMBED_DIM,
    window=5,
    min_count=3,  # 3번 미만 등장한 희귀 단어는 학습에서 제외 (노이즈 감소)
    min_n=2,
    max_n=4,  # [Tip] 한국어에 맞는 자소/음절 n-gram 범위 설정
    workers=4,
    sg=1,  # Skip-gram
    epochs=10,
)

print(f"FastText 학습 완료. 단어장 크기: {len(ft_model.wv)}")


# =============================================================================
# [STEP 3] 임베딩 매트릭스 구축 (UNK, PAD 처리 핵심) ⭐
# : 이 부분이 'Pro' 버전의 핵심입니다.
# =============================================================================

# [국룰 인덱스 정의] 0번은 패딩(PAD), 1번은 모르는 단어(UNK)로 고정
PAD_IDX = 0
UNK_IDX = 1

word_to_idx = {}
idx_to_word = {}

# FastText에서 학습된 단어들을 가져와 2번부터 번호 부여
words = ft_model.wv.index_to_key
for i, w in enumerate(words, start=2):
    word_to_idx[w] = i
    idx_to_word[i] = w

vocab_size = len(word_to_idx) + 2  # PAD(1개) + UNK(1개) + 실제 단어 수
embed_dim = EMBED_DIM

# 1. 전체 가중치 행렬을 0으로 초기화
embedding_weights = np.zeros((vocab_size, embed_dim), dtype=np.float32)

# 2. [고급 기술] UNK 벡터 초기화
# 모르는 단어가 나왔을 때 0으로 두지 않고, '전체 단어의 평균' 느낌으로 초기화
# 이렇게 하면 모르는 단어가 나와도 모델이 덜 당황합니다.
all_vecs = []
for w in words:
    all_vecs.append(ft_model.wv[w])
all_vecs = np.stack(all_vecs, axis=0)
unk_vector = all_vecs.mean(axis=0)  # 전체 단어 벡터의 평균 계산

embedding_weights[UNK_IDX] = unk_vector  # 1번 인덱스에 할당

# 3. 실제 단어 벡터 채우기
for w, idx_ in word_to_idx.items():
    embedding_weights[idx_] = ft_model.wv[w]

# 4. PyTorch 임베딩 레이어 생성 및 가중치 복사
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
)
with torch.no_grad():
    embedding_layer.weight.copy_(torch.from_numpy(embedding_weights))


# =============================================================================
# [STEP 4] 데이터셋 준비 (속도 최적화)
# =============================================================================


# [최적화] 학습 도중에 문자열을 찾는 게 아니라, 미리 숫자로 다 바꿔둡니다.
def tokens_to_indices(tokens, word_to_idx, unk_idx=UNK_IDX):
    idxs = []
    for t in tokens:
        if t in word_to_idx:
            idxs.append(word_to_idx[t])
        else:
            # 사전에 없는 단어는 UNK 인덱스(1)로 변환! (무시하지 않음)
            idxs.append(unk_idx)

    # 문장이 비어있으면 UNK 하나라도 넣어서 에러 방지
    if len(idxs) == 0:
        idxs = [unk_idx]
    return torch.tensor(idxs, dtype=torch.long)


print("토큰 -> 인덱스 변환 중...")
X_train_idx = [tokens_to_indices(toks, word_to_idx) for toks in train_tokens]
X_valid_idx = [tokens_to_indices(toks, word_to_idx) for toks in valid_tokens]


class ReviewDataset(Dataset):
    def __init__(self, seq_list, labels):
        self.seq_list = seq_list
        self.labels = labels

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        return self.seq_list[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    seqs, labels = zip(*batch)
    # 배치 내에서 가장 긴 문장에 맞춰 패딩(0) 채우기
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded, labels


BATCH_SIZE = 64
train_dataset = ReviewDataset(X_train_idx, y_train)
valid_dataset = ReviewDataset(X_valid_idx, y_valid)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# =============================================================================
# [STEP 5] 모델 정의 (Masking 적용)
# =============================================================================


class FastTextSentClassifier(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, x):
        # 1. 임베딩: (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # 2. 패딩 마스크 생성: 0(PAD)인 부분은 계산에서 제외하기 위함
        # (batch, seq_len, 1)
        mask = (x != PAD_IDX).unsqueeze(-1).float()

        # 3. 평균 벡터 계산
        # 그냥 mean()을 쓰면 패딩(0)까지 평균에 포함되므로,
        # (실제 단어 합) / (실제 단어 개수)로 직접 계산합니다.
        emb_sum = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)  # 0으로 나누기 방지

        sent_vec = emb_sum / lengths

        # 4. 로짓 출력
        logit = self.fc(sent_vec).squeeze(1)
        return logit


# =============================================================================
# [STEP 6] 학습 설정 (차별적 학습률 적용) ⭐
# =============================================================================

model = FastTextSentClassifier(embedding_layer)
criterion = nn.BCEWithLogitsLoss()

# 파라미터 그룹 분리
pretrained_params = model.embedding.parameters()
head_params = model.fc.parameters()

# [고급 기술] Differential Learning Rate
# 임베딩은 이미 똑똑하니까 살살(1e-4) 학습시키고,
# 분류기(fc)는 처음 보는 거니까 좀 더 세게(1e-3) 학습시킵니다.
optimizer = torch.optim.Adam(
    [
        {"params": pretrained_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ]
)


# =============================================================================
# [STEP 7] 학습 및 평가 루프 (tqdm 사용)
# =============================================================================


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # 훈련 모드
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # tqdm으로 진행률 바 표시
    for batch_x, batch_y in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # 정확도 계산
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_model(model, loader, criterion):
    model.eval()  # 평가 모드 (필수!)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Valid", leave=False):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# 실행 Loop
EPOCHS = 10
print("\n=== PyTorch 학습 시작 ===")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc = eval_model(model, valid_loader, criterion)

    print(
        f"[Epoch {epoch}] "
        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
        f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}"
    )


# =============================================================================
# [STEP 8] 실전 예측 함수
# =============================================================================


def predict_sentiment(model, text: str):
    model.eval()
    tokens = preprocess_text(text)
    # 여기서는 UNK 처리가 자동으로 됩니다 (tokens_to_indices 함수 덕분)
    idxs = tokens_to_indices(tokens, word_to_idx)
    idxs = idxs.unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        logit = model(idxs)
        prob = torch.sigmoid(logit).item()

    label = "긍정" if prob >= 0.5 else "부정"
    return label, prob


print("\n=== 예측 테스트 ===")
test_text = "스토리가 정말 재미있고 배우들 연기도 좋았어요."
label, prob = predict_sentiment(model, test_text)
print(f"문장: {test_text}")
print(f"결과: {label} (확률: {prob:.4f})")

test_text2 = "돈 아깝고 지루해서 죽는 줄 알았음"
label2, prob2 = predict_sentiment(model, test_text2)
print(f"문장: {test_text2}")
print(f"결과: {label2} (확률: {prob2:.4f})")
