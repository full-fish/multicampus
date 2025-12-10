import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# =============================================================================
# [STEP 1] 데이터 로드 및 분할
# : 원본 데이터를 불러와서 훈련용과 테스트용으로 나눕니다.
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

# 1-3. 훈련 데이터와 테스트 데이터 분리 (8:2 비율)
# 중요: 테스트 데이터는 학습에 전혀 관여하지 않아야 함
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
# : 문장을 단어들의 리스트로 변환합니다. (예: "이 영화" -> ["이", "영화"])
# =============================================================================

# 띄어쓰기 기준으로 자름 (실전에서는 형태소 분석기 사용 권장)
X_train_tokens = [str(sent).split() for sent in X_train]
X_test_tokens = [str(sent).split() for sent in X_test]

print("토큰화 예시:", X_train_tokens[0])


# =============================================================================
# [STEP 3] Word2Vec 임베딩 학습 (사전 학습)
# : 훈련 데이터의 단어들로 벡터 표현을 학습합니다.
# =============================================================================

# 3-1. Word2Vec 모델 생성 및 학습
model = Word2Vec(
    X_train_tokens,  # 훈련 데이터만 사용
    vector_size=50,  # 벡터 차원 크기
    window=3,
    min_count=2,  # 최소 2번 이상 나온 단어만 학습
    sg=1,  # Skip-gram
    workers=1,
    epochs=20,
)

# 3-2. 학습 결과 추출
word_index = model.wv.key_to_index  # 단어 -> 인덱스 사전
pretrained_weights = model.wv.vectors  # 학습된 벡터 행렬
vocab_size = len(word_index)
embed_dim = model.vector_size

print(f"단어 집합 크기: {vocab_size}")
print(f"임베딩 벡터 차원: {embed_dim}")


# =============================================================================
# [STEP 4] 임베딩 행렬 준비
# : PyTorch에서 사용할 수 있도록 Word2Vec 가중치에 패딩용 벡터를 추가합니다.
# =============================================================================

PAD_IDX = vocab_size  # 패딩 인덱스는 단어장 크기와 같게 설정 (맨 마지막)
vocab_size_with_pad = vocab_size + 1

# 0으로 채운 행렬 생성 후 기존 가중치 복사
extended_weights = np.zeros((vocab_size_with_pad, embed_dim), dtype=np.float32)
extended_weights[:vocab_size, :] = pretrained_weights
extended_weights[PAD_IDX, :] = 0.0  # 패딩 벡터는 0


# =============================================================================
# [STEP 5] PyTorch 데이터셋 & 로더 정의
# : 데이터를 배치 단위로 묶고 패딩을 적용하여 모델에 공급하는 역할
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
            # OOV(사전에 없는 단어)는 제외

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


# 데이터셋 및 로더 인스턴스 생성
train_dataset = PaddedTextDataset(X_train_tokens, y_train.values, word_index, PAD_IDX)
test_dataset = PaddedTextDataset(X_test_tokens, y_test.values, word_index, PAD_IDX)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_pad
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_pad
)


# =============================================================================
# [STEP 6] 모델 구조 정의 (Network Architecture)
# : 임베딩 -> 마스킹 -> 풀링 -> 분류기로 이어지는 신경망
# =============================================================================


class PaddedSentClassifier(nn.Module):
    def __init__(self, embedding: nn.Embedding, pad_idx: int):
        super().__init__()
        self.embedding = embedding
        self.pad_idx = pad_idx
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_batch):
        emb = self.embedding(idx_batch)

        # 패딩 마스크 생성 (패딩은 0, 실제 단어는 1)
        mask = (idx_batch != self.pad_idx).float()

        # 실제 길이 계산
        lenghs = mask.sum(dim=1, keepdim=True)

        # 마스킹 적용 (패딩 부분 값을 0으로)
        masked_emb = emb * mask.unsqueeze(dim=2)

        # 평균 풀링 (단어 벡터들의 평균 구하기)
        sum_emb = masked_emb.sum(dim=1)
        sent_vec = sum_emb / lenghs.clamp(min=1.0)

        # 결과 출력 (로짓)
        logits = self.fc(sent_vec).squeeze(1)
        return logits


# =============================================================================
# [STEP 7] 학습 및 평가 루프 정의
# : 에포크를 돌며 모델을 학습시키고 성능을 평가하는 함수
# =============================================================================


import copy  # 모델의 가중치를 복사하기 위해 필요


def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, patience=3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 조기 종료를 위한 변수 설정
    best_loss = float("inf")  # 초기 최적 오차는 무한대로 설정
    patience_counter = 0  # 성능이 향상되지 않은 횟수 카운트
    best_model_wts = copy.deepcopy(model.state_dict())  # 최고의 가중치를 저장할 변수

    for epoch in range(1, num_epochs + 1):
        # --- 훈련 모드 ---
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

        # --- 평가 모드 ---
        model.eval()
        test_loss = 0.0  # 테스트 오차도 계산해야 함 (조기 종료 기준)
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                # 오차 계산 (평가 기준)
                loss = criterion(logits, y)
                test_loss += loss.item()

                # 정확도 계산
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total * 100

        print(
            f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {accuracy:.2f}%"
        )

        # --- 조기 종료 로직 (Early Stopping) ---
        if avg_test_loss < best_loss:
            # 1. 오차가 줄어들어 성능이 좋아진 경우
            best_loss = avg_test_loss
            patience_counter = 0  # 카운트 초기화
            # 현재 최고의 모델 가중치를 백업해 둠
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"    -> 최고 성능 갱신! (Best Loss: {best_loss:.4f})")
        else:
            # 2. 오차가 줄어들지 않은 경우
            patience_counter += 1
            print(f"    -> 성능 개선 없음 ({patience_counter}/{patience})")

            # 참을성을 넘어서면 강제 종료
            if patience_counter >= patience:
                print("\n[조기 종료] 학습을 중단합니다.")
                break

    # 학습 종료 후, 가장 좋았던 상태로 모델을 되돌림
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
# [STEP 8] 실행 (Main Execution)
# : 준비된 부품들을 조립하고 실제 학습을 시작하는 단계
# =============================================================================

# 8-1. 임베딩 레이어 생성 및 가중치 이식
padded_embedding = nn.Embedding(vocab_size_with_pad, embed_dim)
with torch.no_grad():
    padded_embedding.weight.copy_(torch.from_numpy(extended_weights))
padded_embedding.weight.requires_grad = True  # 미세조정 허용

# 8-2. 모델 초기화
model_cls = PaddedSentClassifier(padded_embedding, pad_idx=PAD_IDX)

# 8-3. 학습 시작
print("\n=== 모델 학습 시작 ===")
train_model(model_cls, train_loader, test_loader, num_epochs=300, lr=0.0005, patience=5)


# =============================================================================
# [STEP 9] 새로운 문장 예측 (Inference)
# : 학습된 모델을 이용해 새로운 문장의 긍정/부정을 판단
# =============================================================================


def predict_sentiment(model, sentence, word_index, pad_idx):
    model.eval()
    tokens = str(sentence).split()
    idxs = []
    for w in tokens:
        if w in word_index:
            idxs.append(word_index[w])

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
