import re
import numpy as np
import pandas as pd
from collections import Counter  # 단어 빈도수 계산을 위한 Counter 임포트


def get_encoded_data(
    file_name: str,
    file_type: str,
    text_col: str,
    label_col: str,
    PAD_TOKEN: str,
    UNK_TOKEN: str,
) -> tuple:
    file_path = rf"{file_name}.{file_type}"

    if file_type == "csv":
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
        )
    elif file_type == "json":
        df = pd.read_json(
            file_path,
            encoding="utf-8",
        )
    else:
        raise ValueError(f"지원하지 않는 파일 형식: .{file_type}")

    df = df.dropna()

    sentences = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()

    def tokenize(text: str):
        # 정제: 한글/숫자/공백을 제외한 모든 문자(특수문자 등)를 공백으로 대체
        text = re.sub(r"[^가-힣0-9\s]", " ", text)
        # 정제: 연속된 공백을 하나의 공백으로 압축하고 양쪽 끝 공백 제거
        text = re.sub(r"\s+", " ", text).strip()
        # 토큰화: 공백 기준으로 문장을 단어 리스트로 분리
        return text.split()

    # 모든 문장에 토큰화 함수를 적용
    tokenized_sentences = [tokenize(s) for s in sentences]

    # --- 3. 단어 사전(Vocabulary) 생성 ---
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}  # 인덱스 0, 1 할당

    # 빈도수가 높은 단어부터 순서대로 Vocab에 정수 인덱스를 부여
    for word, _ in counter.most_common():
        vocab[word] = len(vocab)

    # --- 4. 정수 인코딩 (Integer Encoding) ---
    def encode(tokens, vocab, unk_token=UNK_TOKEN):
        unk_idx = vocab[unk_token]
        # 각 토큰에 대해 Vocab에서 해당 인덱스를 찾고, 없으면 UNK 인덱스를 사용
        return [vocab.get(t, unk_idx) for t in tokens]

    # 모든 토큰화된 문장에 정수 인코딩을 적용
    encoded_sentences = [encode(tokens, vocab) for tokens in tokenized_sentences]

    # 정수 인코딩된 시퀀스와 라벨을 반환
    return encoded_sentences, labels, vocab
