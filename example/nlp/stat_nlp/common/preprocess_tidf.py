import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


def get_vectored_value(file_name, file_type, sampling_num):
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
    _, df_sample = train_test_split(
        df, test_size=sampling_num, stratify=df["label"], shuffle=True, random_state=42
    )
    df_sample = df_sample.reset_index(drop=True)
    okt = Okt()
    with open(
        r"/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/stopwords-ko.txt",
        encoding="utf-8",
    ) as f:
        stopwords = set(w.strip() for w in f if w.strip())
    add_stopwords = set(
        [
            "그리고",
            "그래서",
            "하지만",
            "그러나",
            "너무",
            "정말",
            "진짜",
            "조금",
            "또한",
            "그냥",
            "매우",
            "에서",
            "에게",
            "미만",
            "이상",
            "같은",
            "하다",
            "되다",
        ]
    )
    stopwords.update(add_stopwords)

    def preprocess_text(text: str) -> list:
        """한글 리뷰를 정규화 + 형태소 분석해서 (명사/동사/형용사) 토큰만 반환"""
        text = text.lower()
        text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

        morphs = okt.pos(text, norm=True, stem=True)
        tokens = []
        for word, tag in morphs:
            if tag in ["Noun", "Verb", "Adjective"]:
                if word not in stopwords and len(word) > 1:
                    tokens.append(word)
        return tokens

    # 5. TF-IDF 객체 생성 및 벡터화 수행
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=3,
        # token_pattern=None, # tokenizer를 지정하면 token_pattern=None은 생략 가능
        tokenizer=preprocess_text,
    )

    # df_sample의 'document' 열을 사용하여 TF-IDF 학습 및 변환 수행
    TEXT_COL = "document"  # 리뷰 데이터의 열 이름
    X_tfidf = tfidf_vectorizer.fit_transform(df_sample[TEXT_COL])

    # 6. TruncatedSVD (LSA) 파이프라인은 반환하지 않으므로, 이 시점에서 사용하지 않습니다.
    # 만약 파이프라인도 반환해야 한다면 pipe_lsa를 추가하면 됩니다.

    # 7. 요청된 세 가지 값 반환
    return df_sample, X_tfidf, tfidf_vectorizer
