from gensim.models import FastText
import numpy as np  # 벡터 출력을 보기 좋게 하기 위해 numpy import

# 학습에 사용할 샘플 문장 (토큰화된 상태)
sentences = [
    ["이", "영화", "정말", "최고", "였다"],
    ["배우", "연기", "가", "최고", "이다"],
    ["스토리", "가", "지루하다"],
    ["내용", "이", "지루하고", "별로", "였다"],
    ["이", "브랜드", "디자인", "이", "세련되다"],
    ["이", "브랜드", "가격", "은", "비싸다"],
]


# FastText 모델 학습
ft_model = FastText(
    sentences,
    vector_size=100,  # 생성할 단어 벡터의 차원 수
    window=5,  # 주변 단어를 고려할 윈도우 크기
    min_count=1,  # 최소 출현 빈도를 1로 설정하여 모든 단어를 단어 집합(Vocabulary)에 포함 (이전 KeyError 방지)
    min_n=2,  # subword의 최소 길이 (2-gram)
    max_n=4,  # subword의 최대 길이 (4-gram)
    sg=1,  # Skip-gram 방식 사용 (sg=0이면 CBOW)
    workers=4,  # 학습에 사용할 CPU 코어 수
)

# 학습된 단어 "브랜드"의 벡터 크기 출력
# min_count=1 덕분에 "브랜드"는 정식 단어 벡터로 학습됨
print('단어 "브랜드" 벡터 크기 : ', ft_model.wv["브랜드"].shape)

# OOV(Out-Of-Vocabulary) 단어 정의
oov_word = "브랜드맛집"

# OOV 단어의 벡터 생성 및 출력
# FastText는 단어 집합에 없어도, 단어를 구성하는 subword들의 합으로 벡터를 생성할 수 있음
oov_vec = ft_model.wv[oov_word]
print("\n# OOV 단어 '브랜드맛집'의 벡터 (subword 합)\n", oov_vec)

# 정식 단어 "브랜드"의 벡터 출력
print('\n# 정식 단어 "브랜드"의 벡터\n', ft_model.wv["브랜드"])

# OOV 단어 "브랜드맛집"과 정식 단어 "브랜드" 사이의 유사도 계산 및 출력
# FastText의 강점: OOV 단어라도 유사한 벡터를 가질 수 있음
print("브랜드 vs 브랜드맛집 유사도 : ", ft_model.wv.similarity("브랜드", "브랜드맛집"))

# FastText가 학습한 subword(n-gram) 벡터들의 배열 크기 출력
# (gensim 기본값: 2,000,000개의 해시 버킷 크기, 100차원)
print(
    "\n# Subword 벡터 저장 공간 크기 (hash bucket, vector_size)\n",
    ft_model.wv.vectors_ngrams.shape,
)

# "지루하고" 단어의 인덱스 확인 (min_count=1 덕분에 이제 KeyError 없이 접근 가능)
word = "지루하고"
idx = ft_model.wv.key_to_index[word]
print("\n# 단어 '지루하고'의 단어 집합 인덱스 : ", idx)

# "지루하고" 단어를 구성하는 subword들이 저장된 해시 버킷 인덱스(list of hash indices)를 가져옴
bucket_indices = ft_model.wv.buckets_word[idx]
print("# '지루하고'를 구성하는 subword들이 매핑된 버킷 인덱스 : \n", bucket_indices)

# 위의 버킷 인덱스를 사용하여 실제 subword 벡터들을 가져옴
subword_vectors = ft_model.wv.vectors_ngrams[bucket_indices]

# "지루하고" 단어가 몇 개의 subword로 구성되었는지 확인
print("\n# '지루하고' 단어를 구성하는 subword의 개수 : ", subword_vectors.shape[0])
# 첫 번째 subword 벡터의 처음 5차원만 출력
print("# 첫 번째 subword 벡터 (처음 5차원) : ")
print(subword_vectors[0][:5])
