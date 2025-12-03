"""
# 텍스트 분류 실습 과제 노트북

## 과제 목표

하나의 텍스트 데이터셋(예: 리뷰, 댓글, SNS 글 등)에 대해

1. 텍스트 데이터를 로드하고 간단한 EDA(탐색적 데이터 분석)를 수행한다.
2. TF-IDF 벡터화를 적용하고,
3. 세 가지 모델을 학습 및 비교한다.
   - 로지스틱 회귀 (`LogisticRegression`)
   - 나이브 베이즈 (`MultinomialNB`)
   - 선형 SVM (`LinearSVC`)
4. 평가지표(특히 macro-F1)를 기준으로 모델 성능을 비교·분석한다.
5. 간단한 보고서(요약 문장)를 마크다운으로 정리한다.

---

>
"""

import re
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
from konlpy.tag import Okt

okt = Okt()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

# plt.rcParams["font.family"] = "Malgun Gothic"
# plt.rcParams["axes.unicode_minus"] = False

print("라이브러리 임포트 완료")

"""
## 1. 데이터 불러오기 및 전처리

### 1-1. CSV 파일 불러오기


"""
df = pd.read_csv("stat_nlp/naive_bayes_svm/movie_reviews.csv")
print(df.head())

# 불용어
with open(
    "stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())

"""### 1-2. 간단 EDA

- 데이터 크기 확인 (`df.shape`)
- 레이블 분포 확인 (`value_counts()`)
- 결측값 여부 확인 (`isna().sum()`)
"""
print("데이터 크기:", df.shape)

print("\n레이블 분포:")
print(df["label"].value_counts())  # 각각 10만개

print("\n결측값 개수:")
print(df.isna().sum())
df.dropna(inplace=True)

print("\n결측값 개수:")
print(df.isna().sum())

print("\n레이블 분포:")
print(df["label"].value_counts())  # 각각 99996개


# 너무 오래 걸려서 층화추출
df_sample, _ = train_test_split(
    df,
    train_size=20000,
    stratify=df["label"],
    shuffle=True,
    random_state=42,
)
print(df_sample["label"].value_counts())  # 각각 10000개
"""
1-3. 데이터 전처리(함수로 구현)


*   정제 및 정규화(정규식사용), 어간/표제어처리, 불용어 제거
"""
okt = Okt()
possible_pos = ["Noun", "Verb", "Adjective"]


def preprocess(text):
    str_reg = re.sub(r"[^가-힝0-9a-zA-Z\s]", "", text).lower()
    pos = okt.pos(str_reg, norm=True, stem=True, join=True)
    pos = [word.split("/") for word in pos]
    filtered_pos = [
        word
        for word, tag in pos
        if word and word not in stopwords and tag in possible_pos
    ]
    return filtered_pos


"""
## 2. 학습/테스트 데이터 분리

- `train_test_split`으로 데이터를 분리
- 가능하면 `stratify=df["label"]` 옵션을 사용해 **레이블 비율을 유지**
"""
X = df_sample["document"]
y = df_sample["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    tokenizer=preprocess,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=2,
)

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

"""
## 3. 공통 함수: 모델 학습평가

- TF-IDF + 분류기를 하나의 `Pipeline`으로 묶어서 사용
- `classification_report`와 `macro-F1` 점수를 함께 출력
"""


def get_score(clf, name):
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=preprocess,
                    token_pattern=None,
                    ngram_range=(1, 2),
                ),
            ),
            ("clf", clf),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")

    return macro_f1


"""## 4. 모델별 학습 & 평가

세 가지 모델을 모두 학습해 보고 성능을 비교

1. 로지스틱 회귀 (`LogisticRegression`)
2. 나이브 베이즈 (`MultinomialNB`)
3. LinearSVC (`LinearSVC`)
"""
results = {}

log_clf = LogisticRegression(max_iter=1000)
log_f1 = get_score(log_clf, "LogisticRegression")
results["LogisticRegression"] = log_f1

nb_clf = MultinomialNB()
nb_f1 = get_score(nb_clf, "MultinomialNB")
results["MultinomialNB"] = nb_f1

svc_clf = LinearSVC()
svc_f1 = get_score(svc_clf, "LinearSVC")
results["LinearSVC"] = svc_f1
print(results)

"""## 5. 성능 비교 표 만들기

세 모델의 macro-F1 점수를 하나의 표로 정리
"""
f1_df = pd.DataFrame(
    {"model": list(results.keys()), "macro_f1": list(results.values())}
)
print("f---1_df---")

print(f1_df)
# MultinomialNB이 제일 좋게 나옴


"""
##5.1 최적 하이퍼파라미터 찾기
# """


def search_best(clf, param_grid, name):
    gs = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        verbose=1,
    )

    gs.fit(X_train_tfidf, y_train)

    print(f"\n===== {name} GridSearchCV 결과 =====")
    print("최적 하이퍼파라미터:", gs.best_params_)
    print("CV 기준 최고 Macro-F1:", gs.best_score_)

    best_clf = gs.best_estimator_
    y_pred = best_clf.predict(X_test_tfidf)

    print(f"\n===== {name} (best model) 테스트 성능 =====")
    print(classification_report(y_test, y_pred))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"테스트 세트 Macro-F1: {macro_f1:.4f}")

    return best_clf, macro_f1, gs.best_params_


best_results = {}
best_models = {}

log_param_grid = {
    "C": [0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}
log_best, log_f1, log_best_params = search_best(
    log_clf, log_param_grid, "LogisticRegression"
)
best_results["LogisticRegression"] = log_f1
best_models["LogisticRegression"] = {
    "model": log_best,
    "params": log_best_params,
}


nb_param_grid = {
    "alpha": [0.1, 0.5, 1.0],
    "fit_prior": [True, False],
}
nb_best, nb_f1, nb_best_params = search_best(nb_clf, nb_param_grid, "MultinomialNB")
best_results["MultinomialNB"] = nb_f1
best_models["MultinomialNB"] = {
    "model": nb_best,
    "params": nb_best_params,
}


svc_param_grid = {
    "C": [0.1, 1.0, 10.0],
}
svc_best, svc_f1, svc_best_params = search_best(svc_clf, svc_param_grid, "LinearSVC")
best_results["LinearSVC"] = svc_f1
best_models["LinearSVC"] = {
    "model": svc_best,
    "params": svc_best_params,
}

print("\n===== 최종 Macro-F1 요약 =====")
for name, score in best_results.items():
    print(f"{name}: {score:.4f}")

"""
### 6. 나이브 베이즈: 클래스별 대표 단어

- `MultinomialNB`의 `feature_log_prob_`를 이용해
- 각 클래스에서 중요한 단어 TOP-N을 뽑기
"""
tfidf_nb = vectorizer
nb_model = best_models["MultinomialNB"]["model"]

feature_names = np.array(tfidf_nb.get_feature_names_out())
top_n = 15

for i, class_label in enumerate(nb_model.classes_):
    class_log_prob = nb_model.feature_log_prob_[i]
    top_indices = class_log_prob.argsort()[-top_n:]
    print(
        f"\n클래스 {class_label} {'긍정' if class_label else '부정'} 대표 단어 TOP-{top_n}"
    )
    print(feature_names[top_indices])
"""
### 7. LinearSVC: 단어 가중치 분석

- `coef_`를 이용해 각 단어가 어떤 클래스로 기울게 만드는지 확인

"""
tfidf_svc = vectorizer
svc_model = best_models["LinearSVC"]["model"]

feature_names = np.array(tfidf_svc.get_feature_names_out())


coef = svc_model.coef_[0]

top_pos_idx = coef.argsort()[-top_n:]
top_neg_idx = coef.argsort()[:top_n]

print(f"\n[LinearSVC] 긍정(+) 방향으로 강하게 기여하는 단어 TOP-{top_n}")
print(feature_names[top_pos_idx])

print(f"\n[LinearSVC] 부정(-) 방향으로 강하게 기여하는 단어 TOP-{top_n}")
print(feature_names[top_neg_idx])
