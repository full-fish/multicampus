import re
from collections import Counter
import numpy as np  # 패딩 및 마스크를 위해 numpy 임포트

# 예시 데이터 (간단 한국어 문장 + 감성 라벨)
sentences = [
    "배송이 빠르고 포장이 깔끔해요",
    "배송이 너무 느리고 제품이 마음에 안 들어요",
    "가격이 저렴해서 만족스러워요",
    "포장이 엉망이고 배송도 늦었어요",
]
labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정 (예시)


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
    counter.update(tokens)

# 특수 토큰 정의
PAD_TOKEN = "<PAD>"  # 패딩(길이 맞추기)을 위한 토큰 (인덱스 0)
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
encoded_list = [encode(tokens, vocab) for tokens in tokenized_sentences]
print("정수 인코딩 결과:", encoded_list)


# 4. 패딩(Padding) 및 마스크(Mask) 생성
PAD_INDEX = vocab[PAD_TOKEN]  # 패딩에 사용할 인덱스 (여기서는 0)

# 최대 길이 설정: 전체 문장 중 가장 긴 문장의 길이로 설정
max_len = max(len(seq) for seq in encoded_list)
print(f"\n최대 문장 길이 (Max Length): {max_len}")


def pad_sequences(encoded_list, max_len, padding_value=PAD_INDEX):
    # 빈 패딩 리스트 초기화
    padded = []
    # 마스크 리스트 초기화
    masks = []

    for sequence in encoded_list:
        # 패딩을 채워야 하는 길이
        pad_len = max_len - len(sequence)

        # 패딩: 문장 뒤에 0(PAD_INDEX)을 채워 길이를 maxlen으로 맞춤
        padded_seq = sequence + [padding_value] * pad_len
        padded.append(padded_seq)

        # 마스크: 원래 토큰 위치는 1, 패딩 위치는 0으로 채움
        # (시퀀스 길이만큼 1을 채우고, 패딩 길이만큼 0을 채움)
        mask = [1] * len(sequence) + [0] * pad_len
        masks.append(mask)

    return np.array(padded), np.array(masks)


# 패딩 및 마스크 생성 적용
# 'encoded_list'를 사용하여 패딩 함수를 호출합니다.
padded_sequences, attention_masks = pad_sequences(encoded_list, max_len)

print("패딩된 정수 시퀀스 (Padded Sequences):")
print(padded_sequences)
print("\n어텐션 마스크 (Attention Masks):")
print(attention_masks)
