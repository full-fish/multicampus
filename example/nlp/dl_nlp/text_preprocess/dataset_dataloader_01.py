import re  # 정규 표현식 모듈 임포트 (텍스트 정제용)
from collections import Counter  # 단어 빈도수 계산을 위한 Counter 임포트
import numpy as np  # 패딩 및 마스크를 위해 numpy 임포트 (여기서는 직접 사용되지는 않으나, 관례적으로 임포트)
import torch  # PyTorch 라이브러리 임포트 (텐서 및 모델 학습용)
from torch.utils.data import Dataset, DataLoader  # Dataset과 DataLoader 클래스 임포트

# 예시 데이터 (간단 한국어 문장 + 감성 라벨)
sentences = [
    "배송이 빠르고 포장이 깔끔해요",
    "배송이 너무 느리고 제품이 마음에 안 들어요",
    "가격이 저렴해서 만족스러워요",
    "포장이 엉망이고 배송도 늦었어요",
]
labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정 (감성 라벨)


# 1. 텍스트 정제 및 토큰화
def tokenize(text: str):
    # 정제: 한글/숫자/공백(\s)을 제외한 모든 문자(특수문자 등)를 공백으로 대체
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    # 정제: 연속된 공백을 하나의 공백으로 압축하고 양쪽 끝 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    # 토큰화: 공백 기준으로 문장을 단어 리스트로 분리
    tokens = text.split()
    return tokens


# 모든 문장에 토큰화 함수를 적용
tokenized_sentences = [tokenize(s) for s in sentences]
print("토큰화 결과:", tokenized_sentences)

# 2. 단어 사전(Vocabulary) 생성
counter = Counter()
for tokens in tokenized_sentences:
    # 모든 토큰의 등장 빈도수를 계산
    counter.update(tokens)

# 특수 토큰 정의
PAD_TOKEN = "<PAD>"  # 패딩(길이 맞추기)을 위한 토큰 (인덱스 0, 가장 중요)
UNK_TOKEN = "<UNK>"  # 사전에 없는 단어(Out-Of-Vocabulary)를 위한 토큰 (인덱스 1)

# 정수 0과 1을 특수 토큰에 할당하여 Vocab 초기화
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

# 빈도수가 높은 단어부터 순서대로 Vocab에 정수 인덱스를 부여
for word, _ in counter.most_common():
    # 현재 Vocab의 크기를 새로운 단어의 인덱스로 사용
    vocab[word] = len(vocab)

print("Vocab:", vocab)


# 3. 정수 인코딩 (Integer Encoding)
def encode(tokens, vocab, unk_token=UNK_TOKEN):
    # UNK 토큰의 인덱스를 미리 저장 (사전에 없는 단어 처리용)
    unk_idx = vocab[unk_token]

    # 각 토큰에 대해 Vocab에서 해당 인덱스를 찾고, 없으면 UNK 인덱스를 사용
    return [vocab.get(t, unk_idx) for t in tokens]


# 모든 토큰화된 문장에 정수 인코딩을 적용
encoded_sentences = [encode(tokens, vocab) for tokens in tokenized_sentences]
print("정수 인코딩 결과:", encoded_sentences)


# 4. 패딩 및 마스크 생성 (PyTorch 사용)
def pad_sequences(encoded_list, max_len, pad_value=0):
    padded = []
    masks = []

    for seq in encoded_list:
        # 1. 길이 초과 처리: 문장 길이가 max_len을 넘으면 자르기 (Trimming)
        if len(seq) > max_len:
            seq = seq[:max_len]

        # 2. 패딩 처리: 최대 길이와 현재 길이의 차이만큼 패딩 값(0) 추가
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len

        # 3. 마스크 생성: 실제 단어 위치는 1, 패딩 위치는 0
        mask = [1] * len(seq) + [0] * pad_len

        padded.append(padded_seq)
        masks.append(mask)

    # 결과를 PyTorch 텐서로 변환하여 반환
    return torch.tensor(padded), torch.tensor(masks)


# 최대 길이 설정
max_len = 6

# 패딩 및 마스크 적용
padded_inputs, attention_masks = pad_sequences(
    encoded_sentences, max_len, pad_value=vocab[PAD_TOKEN]  # PAD_TOKEN의 인덱스 0 사용
)

print("Padded inputs:\n", padded_inputs)
print("Attention masks:\n", attention_masks)
print("Tensor shape:", padded_inputs.shape)  # (배치 크기, max_len)


# 5. PyTorch Dataset 클래스 정의 (데이터셋 구조화)
class ReviewDataSet(Dataset):
    # 데이터셋 초기화 메서드
    def __init__(self, inputs, masks, labels):
        # 패딩된 입력 텐서 (input_ids) 저장
        self.inputs = inputs
        # 어텐션 마스크 텐서 저장
        self.masks = masks
        # 라벨을 Long 타입의 텐서로 변환하여 저장 (분류 문제에서 라벨은 보통 Long 타입)
        self.labels = torch.tensor(labels, dtype=torch.long)

    # 전체 데이터셋의 크기를 반환하는 메서드
    def __len__(self):
        return len(self.labels)

    # 특정 인덱스의 항목을 반환하는 메서드
    def __getitem__(self, index):
        # 모델 입력에 필요한 세 요소를 딕셔너리 형태로 반환
        return {
            "input_ids": self.inputs[index],
            "attention_mask": self.masks[index],
            "labels": self.labels[index],
        }


# 6. Dataset 및 DataLoader 생성
# ReviewDataSet 인스턴스 생성
dataset = ReviewDataSet(padded_inputs, attention_masks, labels)

# DataLoader 생성 (데이터를 효율적으로 배치 단위로 로드하고 셔플링)
loader = DataLoader(
    dataset,
    batch_size=2,  # 한 번에 가져올 데이터의 수
    shuffle=True,  # 에포크마다 데이터를 섞을지 여부
)

# 배치 하나 꺼내서 확인
# iter(loader)로 이터레이터를 만들고 next()로 첫 번째 배치 데이터를 가져옴
batch = next(iter(loader))
print("\n--- DataLoader 확인 결과 ---")
print("input_ids batch shape:", batch["input_ids"].shape)  # (batch_size, max_len)
print(
    "attention_mask batch shape:", batch["attention_mask"].shape
)  # (batch_size, max_len)
print("labels batch:", batch["labels"])  # (batch_size)
