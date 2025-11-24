# konlpy: 한국어 형태소 분석 라이브러리
# sklearn: TF-IDF 벡터화 라이브러리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1️⃣ 형태소 분석기 객체 생성
okt = Okt()


# 2️⃣ 사용자 정의 토크나이저 함수 정의
def tokenize_korean(text):
    # 문장을 형태소 + 품사로 분석 (stem=True → 기본형으로 변환)
    morphs = okt.pos(text, stem=True)
    # 명사(Noun), 동사(Verb), 형용사(Adjective)만 추출
    tokens = [word for word, tag in morphs if tag in ["Noun", "Verb", "Adjective"]]
    return tokens  # 예: ['영화', '재밌다', '감동적이다', '보다', '싶다']


# 3️⃣ 분석할 문서(문장) 리스트
docs = [
    "이 영화 정말 재밌어요 최고예요",
    "별로 재미없고 지루했어요",
    "배우 연기도 좋고 감동적이었어요",
    "스토리가 엉망이고 시간 아까웠어요",
    "완전 최고 강추합니다",
    "돈 아깝고 다시는 보고 싶지 않아요",
]

# 4️⃣ TF-IDF 벡터라이저 생성
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_korean,  # 사용자 정의 형태소 분석 함수 지정
    token_pattern=None,  # 기본 영어용 정규식 비활성화 (한글 직접 처리)
    min_df=1,  # 최소 문서 등장 빈도(1 이상 단어만 포함)
    ngram_range=(1, 1),  # unigram 사용 (단어 하나씩)
)

# 5️⃣ 문서들을 TF-IDF 행렬로 변환 (fit: 단어 사전 생성 + transform: 벡터화)
X_tfidf = vectorizer.fit_transform(docs)

# 6️⃣ 생성된 단어 사전(vocabulary) 출력
print("단어집(vocabulary):")
print(vectorizer.get_feature_names_out())
# 예: ['감동적이다' '고프다' '보다' '싶다' '영화' '재밌다']

# 7️⃣ TF-IDF 행렬 크기 출력 (문서 수 × 단어 수)
print("\nTF-IDF 행렬 shape:", X_tfidf.shape)
# 예: (3, 6) → 3개의 문서, 6개의 고유 단어

# 8️⃣ TF-IDF 행렬을 배열로 변환하고 소수점 3자리까지 반올림해 출력
print(X_tfidf.toarray().round(3))
# 각 행 → 문서, 각 열 → 단어, 값 → TF-IDF 가중치
# 값이 높을수록 해당 문서에서 중요한 단어임
