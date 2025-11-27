import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# 한글
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # 리눅스 계열 (예: 구글코랩, 우분투)
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# ----------------------------------------


docs = [
    # 분석할 6개의 고객 리뷰(문서) 목록입니다.
    "이 브랜드 배송이 너무 빨라서 좋았어요",
    "품질은 괜찮은데 배송이 너무 느려요",
    "가격이 저렴해서 가성비가 좋아요",
    "브랜드 이미지가 세련되고 품질도 좋아요",
    "배송도 빠르고 포장도 깔끔해서 만족합니다",
    "가격은 비싼 편인데 품질이 좋아요",
]

# 1) TF-IDF 변환
# -----------------------------------------------------------
tfidf = TfidfVectorizer(
    # TfidfVectorizer 객체를 생성합니다.
    max_df=0.8,  # 문서의 80% 이상에 나타나는 단어는 너무 흔하다고 보고 무시합니다 (불용어 처리).
    min_df=1,  # 최소 1개 문서에 나타나는 단어만 사용합니다.
    token_pattern=r"(?u)\b\w+\b",  # 한글을 포함한 단어 토큰 패턴을 지정합니다.
)

X = tfidf.fit_transform(docs)
# docs를 TF-IDF 행렬로 변환합니다. X는 (문서 수, 단어 수) 크기의 행렬이 됩니다.

print("TF-IDF shape:", X.shape)  # X 행렬의 크기를 출력합니다. (6, 11)이 나올 것입니다.

# 2) LSA - TruncatedSVD 사용
# -----------------------------------------------------------
n_topics = 2  # 추출하고자 하는 주제(토픽)의 개수를 2개로 설정합니다.
# 이 값(k)이 LSA로 축소될 차원이 됩니다.

svd = TruncatedSVD(n_components=n_topics, random_state=42)
# TruncatedSVD 객체를 생성합니다. n_components=2로 설정했습니다.

X_lsa = svd.fit_transform(X)
# TF-IDF 행렬 X에 SVD를 적용하여 차원을 축소합니다.
# X_lsa는 (문서 수, 토픽 수), 즉 (6, 2) 크기의 행렬이 됩니다 (이전의 U_k와 관련됨).

print(
    "LSA shape:", X_lsa.shape
)  # X_lsa 행렬의 크기를 출력합니다. (6, 2)가 나올 것입니다.

# 3) 토픽별 상위 단어 보기
# -----------------------------------------------------------
terms = tfidf.get_feature_names_out()
# TF-IDF 행렬을 만들 때 사용된 모든 단어(피처 이름) 목록을 가져옵니다.

for topic_idx, comp in enumerate(svd.components_):
    # svd.components_는 (토픽 수, 단어 수) 크기의 행렬 (이전의 V_k^T와 관련됨)이며, 각 토픽을 구성하는 단어 가중치를 담고 있습니다.

    term_idx = comp.argsort()[::-1][:5]
    # 현재 토픽(comp)에서 가중치가 높은 단어 5개의 인덱스를 찾습니다. (argsort로 정렬 후 역순으로 5개 선택)

    print(f"\n[토픽 {topic_idx}]")
    # 현재 토픽의 인덱스를 출력합니다.

    for idx in term_idx:
        print(terms[idx], f"({comp[idx]:.4f})")
        # 단어 목록(terms)에서 해당 인덱스의 단어와 가중치를 출력합니다.

# 4) 각 문서의 토픽 좌표 확인
# -----------------------------------------------------------
lsa_df = pd.DataFrame(X_lsa, columns=[f"topic_{i}" for i in range(n_topics)])
# LSA 결과 행렬 X_lsa를 데이터프레임으로 변환하고, 열 이름은 'topic_0', 'topic_1'로 지정합니다.

lsa_df["text"] = docs
# 원본 리뷰 텍스트를 데이터프레임에 새로운 열로 추가합니다.

print("\n문서별 토픽 좌표:")
print(lsa_df)
# 각 문서가 2개의 토픽에 대해 얻은 점수(좌표)를 최종적으로 출력하여 문서의 주제를 확인합니다.
