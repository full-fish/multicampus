import re  # 정규 표현식 모듈 임포트 (텍스트 정제용)
from collections import Counter  # 단어 빈도수 계산을 위한 Counter 임포트
import numpy as np  # numpy 임포트 (여기서는 직접 사용되지 않으나, 관례적으로 임포트)
import torch  # PyTorch 라이브러리 임포트 (텐서 및 모델 학습용)
from torch.utils.data import Dataset, DataLoader  # Dataset과 DataLoader 클래스 임포트
from dl_nlp.common.preprocess import get_encoded_data

# 동적 패딩을 위한 pad_sequence 함수 임포트
from torch.nn.utils.rnn import pad_sequence

# # 예시 데이터 (간단 한국어 문장 + 감성 라벨)
# sentences = [
#     "배송이 빠르고 포장이 깔끔해요",
#     "배송이 너무 느리고 제품이 마음에 안 들어요",
#     "가격이 저렴해서 만족스러워요",
#     "포장이 엉망이고 배송도 늦었어요",
# ]
# labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정 (감성 라벨)


# # --- 1. 텍스트 전처리 단계 ---
# def tokenize(text: str):
#     # 정제: 한글/숫자/공백(\s)을 제외한 모든 문자(특수문자 등)를 공백으로 대체
#     text = re.sub(r"[^가-힣0-9\s]", " ", text)
#     # 정제: 연속된 공백을 하나의 공백으로 압축하고 양쪽 끝 공백 제거
#     text = re.sub(r"\s+", " ", text).strip()
#     # 토큰화: 공백 기준으로 문장을 단어 리스트로 분리
#     return text.split()


# tokenized_sentences = [tokenize(s) for s in sentences]

# # 2. 단어 사전(Vocabulary) 생성
# counter = Counter()
# for tokens in tokenized_sentences:
#     counter.update(tokens)

# PAD_TOKEN = "<PAD>"
# UNK_TOKEN = "<UNK>"
# vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

# for word, _ in counter.most_common():
#     vocab[word] = len(vocab)


# # 3. 정수 인코딩 (Integer Encoding)
# def encode(tokens, vocab, unk_token=UNK_TOKEN):
#     unk_idx = vocab[unk_token]
#     return [vocab.get(t, unk_idx) for t in tokens]

# 특수 토큰 정의
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# encoded_sentences = [encode(tokens, vocab) for tokens in tokenized_sentences]
encoded_sentences, labels, vocab = get_encoded_data(
    "/Users/choimanseon/Documents/multicampus/example/nlp/movie_reviews",
    "csv",
    "document",
    "label",
    PAD_TOKEN,
    UNK_TOKEN,
)


# --- A. 정적 패딩 (Static Padding) 방식 ---
# 4. 패딩 및 마스크 생성 (PyTorch 사용)
def pad_sequences(encoded_list, max_len, pad_value=0):
    padded = []
    masks = []

    for seq in encoded_list:
        # 1. 길이 초과 처리: 문장 길이가 max_len을 넘으면 자르기 (Trimming)
        if len(seq) > max_len:
            seq = seq[:max_len]

        # 2. 패딩 처리
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len

        # 3. 마스크 생성
        mask = [1] * len(seq) + [0] * pad_len

        padded.append(padded_seq)
        masks.append(mask)

    return torch.tensor(padded), torch.tensor(masks)


# 최대 길이 설정 (정적 길이: 6)
max_len = 6

# 패딩 및 마스크 적용
padded_inputs, attention_masks = pad_sequences(
    encoded_sentences, max_len, pad_value=vocab[PAD_TOKEN]
)


# 5. PyTorch Dataset 클래스 정의 (정적 패딩 데이터용)
class ReviewDataSet(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 이미 패딩이 완료된 텐서를 반환
        return {
            "input_ids": self.inputs[index],
            "attention_mask": self.masks[index],
            "labels": self.labels[index],
        }


# 6. 정적 패딩 DataLoader 생성 및 확인
dataset = ReviewDataSet(padded_inputs, attention_masks, labels)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

batch = next(iter(loader))
print("--- 정적 패딩 DataLoader 확인 결과 ---")
print("input_ids batch shape:", batch["input_ids"].shape)
print("attention_mask batch shape:", batch["attention_mask"].shape)
print("labels batch:", batch["labels"])


# --- B. 동적 패딩 (Dynamic Padding) 방식 ---
# 7. Raw Dataset 클래스 정의 (패딩되지 않은 원본 데이터 보관)
class ReviewRawDataset(Dataset):
    # 인코딩된 시퀀스 리스트와 라벨을 받음
    def __init__(self, encoded_sequences, labels):
        super().__init__()
        # encoded_sequences를 torch.tensor 리스트로 변환하여 저장
        self.encoded_sequences = [
            torch.tensor(seq, dtype=torch.long) for seq in encoded_sequences
        ]
        # 라벨은 텐서로 변환
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 패딩되지 않은 원본 길이의 시퀀스와 라벨을 반환
        return self.encoded_sequences[index], self.labels[index]


# 8. collate_fn 정의 (배치 생성 시 동적 패딩 수행)
def collate_fn(batch, pad_value=0):
    # batch: list of (seq_tensor, label) -> zip(*batch)로 시퀀스와 라벨을 분리
    seqs, labels = zip(*batch)  # unzip: 시퀀스 튜플, 라벨 튜플로 분리

    # pad_sequence를 사용하여 현재 배치 내에서 가장 긴 길이에 맞춰 패딩
    padded_seqs = pad_sequence(
        seqs,
        batch_first=True,  # (batch, max_len) 형태로 출력
        padding_value=pad_value,  # 패딩 값은 0
    )

    # 마스크 생성: 패딩 값(0)이 아닌 위치(실제 토큰)는 1, 패딩은 0
    attention_mask = (padded_seqs != pad_value).long()

    # 라벨 튜플을 하나의 텐서로 쌓음
    labels = torch.stack(labels)

    return {
        "input_ids": padded_seqs,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# 9. 동적 패딩 DataLoader 생성 및 확인
raw_dataset = ReviewRawDataset(encoded_sentences, labels)

loader2 = DataLoader(
    raw_dataset,
    batch_size=2,
    shuffle=True,
    # collate_fn 인수에 lambda 함수를 사용하여 동적 패딩 함수 지정
    collate_fn=lambda batch: collate_fn(batch, pad_value=vocab[PAD_TOKEN]),
)

# 배치 하나 꺼내서 확인
batch2 = next(iter(loader2))

print("\n--- 동적 패딩 DataLoader 확인 결과 ---")
# 배치마다 길이가 달라질 수 있음 (최대 길이는 7이 될 수 있음)
print("동적 패딩 input_ids shape:", batch2["input_ids"].shape)

print(
    "\n\n-------------\n",
)
batch2_input_ids_list = batch2["input_ids"].tolist()
reverce_vocab = {index: token for token, index in vocab.items()}
get_batch2_vocab = []
for batch in batch2_input_ids_list:
    temp = []
    for num in batch:
        temp.append(reverce_vocab[num])
    get_batch2_vocab.append(temp)
print("\n\nget_batch2_vocab\n", get_batch2_vocab)

# print("동적 패딩 input_ids shape:", batch2["input_ids"])
print("동적 패딩 attention_mask:\n", batch2["attention_mask"])
