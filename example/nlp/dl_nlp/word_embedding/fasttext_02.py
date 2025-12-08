import re
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# [변경 포인트 1] Word2Vec 대신 FastText 임포트
from gensim.models import FastText

# =============================================================================
# [STEP 1] 데이터 로드 및 분할
# =============================================================================

# 1-1. CSV 파일 읽기 및 전처리
df = pd.read_csv(
    "movie_reviews.csv",
    header=0,
    names=["id", "review", "label"],
)
df = df.dropna()
df["label"] = df["label"].astype(int)

# 1-2. 전체 데이터 중 일부만 샘플링
_, df_sample = train_test_split(
    df,
    test_size=30000,
    stratify=df["label"],
    shuffle=True,
    random_state=42,
)

X = df_sample["review"]
y = df_sample["label"]

# 1-3. 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=42,
)

print(f"훈련 데이터 개수: {len(X_train)}")
print(f"테스트 데이터 개수: {len(X_test)}")


# =============================================================================
# [STEP 2] 텍스트 토큰화 (Tokenization)
# =============================================================================

X_train_tokens = [str(sent).split() for sent in X_train]
X_test_tokens = [str(sent).split() for sent in X_test]

print("토큰화 예시:", X_train_tokens[0])


# =============================================================================
# [STEP 3] FastText 임베딩 학습 (변경됨)
# : 단어를 n-gram으로 쪼개서 학습하므로 오타나 파생어에 강함
# =============================================================================

print("\n[FastText 학습 시작]...")

# [변경 포인트 2] FastText 모델 생성 및 학습
model = FastText(
    sentences=X_train_tokens,  # 훈련 데이터
    vector_size=50,  # 임베딩 차원
    window=3,
    min_count=2,  # 최소 2번 등장한 단어만 단어장에 등록
    min_n=2,  # [FastText 핵심] subword 최소 길이 (예: 2글자)
    max_n=4,  # [FastText 핵심] subword 최대 길이 (예: 4글자)
    sg=1,  # Skip-gram
    workers=4,  # CPU 병렬 처리 개수 (속도 향상)
    epochs=20,
)

# 3-2. 학습 결과 추출
# FastText.wv.vectors에는 이미 subword 정보가 합쳐진 '완성된 단어 벡터'가 들어있습니다.
word_index = model.wv.key_to_index
pretrained_weights = model.wv.vectors
vocab_size = len(word_index)
embed_dim = model.vector_size

print(f"단어 집합 크기: {vocab_size}")
print(f"임베딩 벡터 차원: {embed_dim}")


# =============================================================================
# [STEP 4] 임베딩 행렬 준비 (동일)
# =============================================================================

PAD_IDX = vocab_size
vocab_size_with_pad = vocab_size + 1

extended_weights = np.zeros((vocab_size_with_pad, embed_dim), dtype=np.float32)
extended_weights[:vocab_size, :] = pretrained_weights
extended_weights[PAD_IDX, :] = 0.0


# =============================================================================
# [STEP 5] PyTorch 데이터셋 & 로더 (동일)
# =============================================================================


class PaddedTextDataset(Dataset):
    def __init__(self, tokenized_sentences, labels, word_index, pad_idx):
        super().__init__()
        self.sentences = tokenized_sentences
        self.labels = labels
        self.word_index = word_index
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokens = self.sentences[index]
        y = self.labels[index]
        idxs = []
        for w in tokens:
            if w in self.word_index:
                idxs.append(self.word_index[w])
            # FastText 모델 자체는 OOV 벡터 생성이 가능하지만,
            # PyTorch nn.Embedding은 고정된 인덱스 테이블을 쓰므로
            # 여기서는 학습된 단어장에 없는 단어는 일단 제외합니다.

        if len(idxs) == 0:
            idxs.append(self.pad_idx)

        return torch.tensor(idxs, dtype=torch.long), torch.tensor(
            y, dtype=torch.float32
        )


def collate_fn_with_pad(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded_seqs, labels


train_dataset = PaddedTextDataset(X_train_tokens, y_train.values, word_index, PAD_IDX)
test_dataset = PaddedTextDataset(X_test_tokens, y_test.values, word_index, PAD_IDX)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_pad
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_pad
)


# =============================================================================
# [STEP 6] 모델 구조 정의 (동일)
# =============================================================================


class PaddedSentClassifier(nn.Module):
    def __init__(self, embedding: nn.Embedding, pad_idx: int):
        super().__init__()
        self.embedding = embedding
        self.pad_idx = pad_idx
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_batch):
        emb = self.embedding(idx_batch)
        mask = (idx_batch != self.pad_idx).float()
        lenghs = mask.sum(dim=1, keepdim=True)
        masked_emb = emb * mask.unsqueeze(dim=2)
        sum_emb = masked_emb.sum(dim=1)
        sent_vec = sum_emb / lenghs.clamp(min=1.0)
        logits = self.fc(sent_vec).squeeze(1)
        return logits


# =============================================================================
# [STEP 7] 학습 함수 (동일)
# =============================================================================


def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, patience=3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        # 훈련
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # 평가
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item()
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total * 100

        print(
            f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {accuracy:.2f}%"
        )

        # 조기 종료
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"    -> 최고 성능 갱신! (Best Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    -> 성능 개선 없음 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("\n[조기 종료] 학습을 중단합니다.")
                break

    model.load_state_dict(best_model_wts)


"""
과적합이 되면 "훈련 Loss"는 계속 떨어지지만, "평가(Test) Loss"는 오히려 다시 치솟아 오릅니다"""
"""
드롭아웃 (Dropout) 레이어의 차이

훈련 모드 (model.train()) 과적합(Overfitting)을 막기 위해 신경망의 일부 뉴런을 무작위로 꺼버립니다. 마치 운동할 때 모래주머니를 차고 하듯이, 일부러 뇌의 일부만 사용해서 어렵게 학습시킵니다. 그래서 같은 데이터를 넣어도 매번 결과가 미세하게 다르게 나올 수 있습니다.

평가 모드 (model.eval()) 모래주머니를 풀고 모든 뉴런을 100% 활성화합니다. 학습된 모든 능력을 동원해서 가장 정확하고 일관된 결과를 내놓습니다. 만약 평가 때도 훈련 모드를 쓰면, 멀쩡한 뉴런이 꺼져 있어서 실력이 제대로 안 나옵니다.

배치 정규화 (Batch Normalization) 레이어의 차이

훈련 모드 (model.train()) 지금 들어온 데이터 묶음(배치)만의 평균과 분산을 계산해서 정규화를 수행합니다. 그리고 이 값들을 기억해 둡니다(이동 평균).

평가 모드 (model.eval()) 지금 들어온 테스트 데이터의 통계치를 쓰지 않습니다. 대신 훈련할 때 기억해 뒀던 전체 데이터의 평균과 분산을 가져와서 적용합니다. 테스트 데이터는 1개만 들어올 수도 있는데, 1개 가지고 평균을 내면 값이 왜곡되기 때문입니다.

요약

model.train(): 학습을 위해 일부러 노이즈를 주거나(드롭아웃), 현재 데이터의 통계치에 민감하게 반응(배치 정규화)하는 유동적인 상태입니다.

model.eval(): 학습된 가중치와 통계치를 고정하고, 모델의 전력을 다해 정답을 맞히는 안정적인 상태입니다.

따라서 평가나 예측을 할 때는 반드시 model.eval()을 선언해야만 내가 학습시킨 모델의 진짜 성능을 확인할 수 있습니다."""

# =============================================================================
# [STEP 8] 실행
# =============================================================================

padded_embedding = nn.Embedding(vocab_size_with_pad, embed_dim)
with torch.no_grad():
    padded_embedding.weight.copy_(torch.from_numpy(extended_weights))
padded_embedding.weight.requires_grad = True

model_cls = PaddedSentClassifier(padded_embedding, pad_idx=PAD_IDX)

print("\n=== 모델 학습 시작 (FastText 기반) ===")
train_model(model_cls, train_loader, test_loader, num_epochs=300, lr=0.0005, patience=5)


# =============================================================================
# [STEP 9] 예측
# =============================================================================


def predict_sentiment(model, sentence, word_index, pad_idx):
    model.eval()
    tokens = str(sentence).split()
    idxs = []
    for w in tokens:
        if w in word_index:
            idxs.append(word_index[w])
        # 참고: FastText를 쓰더라도 nn.Embedding 테이블 방식에서는
        # 사전에 없는 단어(OOV)는 인덱스로 변환 불가능하여 제외됩니다.
        # 하지만 FastText 덕분에 학습 데이터 내의 희귀 단어들에 대한
        # 벡터 품질이 Word2Vec보다 훨씬 좋아져 성능 향상이 기대됩니다.
    print("\nidxs\n", idxs)

    if len(idxs) == 0:
        idxs.append(pad_idx)

    input_tensor = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    probability = probs.item()
    result = "긍정" if probability >= 0.5 else "부정"
    return result, probability


print("\n=== 예측 테스트 ===")
test_sentences = [
    "이 영화 정말 최고 였다",
    "스토리 가 지루하고 별로 였다",
    "배우 연기 가 너무 좋다",
    "내용 이 완전 지루하다",
    "시간 아까운 영화",
]

for sent in test_sentences:
    res, prob = predict_sentiment(model_cls, sent, word_index, PAD_IDX)
    print(f"문장: '{sent}'")
    print(f" -> 예측: {res} (확률: {prob:.4f})\n")
