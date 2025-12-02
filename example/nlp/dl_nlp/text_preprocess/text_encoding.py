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
encoded_sentences = [encode(tokens, vocab) for tokens in tokenized_sentences]
print("정수 인코딩 결과:", encoded_sentences)
