import re
import pandas as pd
from konlpy.tag import Okt

okt = Okt()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# 불용어
with open(
    "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())

base_dir = "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/naive_bayes_svm/Sample/02.라벨링데이터"

df = []
i = 1
TARGET = 600
counts = {1: 0, 0: 0, -1: 0}
while True:
    path = base_dir + f"/쇼핑몰/01. 패션/1-1. 여성의류/1-1.여성의류({i}).json"
    try:
        print(i, "번 파일 읽음")
        tmp = pd.read_json(path)

        if all(value >= TARGET for value in counts.values()):
            print("모든 클래스 600개씩 수집 완료")
            break

        for _, row in tmp.iterrows():
            if pd.isna(row["GeneralPolarity"]):
                continue
            if counts[int(row["GeneralPolarity"])] < TARGET:
                counts[int(row["GeneralPolarity"])] += 1
                df.append(row.to_dict())

        i += 1
    except Exception as e:
        print(i, "번 파일에서 에러 발생")
        print("에러 내용:", e)
        break

df = pd.DataFrame(df)

print("총 행 개수:", len(df))
print(df.head())

df_filtered = df[["RawText", "GeneralPolarity"]].dropna()

X = df_filtered["RawText"].astype(str)
y = df_filtered["GeneralPolarity"].astype(int)

print("\ny 분포:")
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


def preprocess(text):
    str_reg = re.sub(r"[^가-힝0-9a-zA-Z\s]", "", text).lower()
    pos = okt.pos(str_reg, norm=True, stem=True, join=True)
    pos = [word.split("/") for word in pos]
    filtered_pos = [
        word
        for word, tag in pos
        if word and word not in stopwords and tag in ["Noun", "Verb", "Adjective"]
    ]
    return filtered_pos


def get_score(vectorizer_type, mode="basic"):
    if vectorizer_type == "count":
        vect = CountVectorizer(
            tokenizer=preprocess,
            token_pattern=None,
        )
        name = "count_MultinomialNB"
    elif vectorizer_type == "tfidf":
        vect = TfidfVectorizer(
            tokenizer=preprocess,
            token_pattern=None,
        )
        name = "tfidf_MultinomialNB"
    pipe = Pipeline([("vect", vect), ("clf", MultinomialNB())])

    if mode == "basic":
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

    elif mode == "best":
        param_grid = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__min_df": [1, 2],
            "clf__alpha": [0.5, 1.0],
        }
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
        )
        sample_w = compute_sample_weight(class_weight="balanced", y=y_train)
        grid.fit(x_train, y_train, clf__sample_weight=sample_w)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test)

    print(f"\n===== {name} =====")
    if mode == "best":
        print("최적 하이퍼파라미터:", grid.best_params_)
    print("\nclassification_report")
    print(classification_report(y_test, y_pred))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")
    return best_model if mode == "best" else pipe, macro_f1


results = {"basic": {}, "best": {}}
models = {"basic": {}, "best": {}}


def get_result(mode="basic"):
    count_model_pipe, count_f1 = get_score("count", mode)
    results[mode]["count"] = count_f1
    models[mode]["count"] = count_model_pipe

    tfidf_model_pipe, tfidf_f1 = get_score("tfidf", mode)
    results[mode]["tfidf"] = tfidf_f1
    models[mode]["tfidf"] = tfidf_model_pipe


get_result("basic")
get_result("best")


def show_df(dic, mode="basic"):

    df = pd.DataFrame(
        {"model": list(dic[mode].keys()), "macro_f1": list(dic[mode].values())}
    )

    print(f"\n=== {mode} 세팅 모델별 Macro-F1 비교 ===")
    print(df.sort_values("macro_f1", ascending=False))


show_df(results, "basic")
show_df(results, "best")


def get_top_words(vectorizer_type, mode, top_n=20, title=""):
    vect = models[mode][vectorizer_type].named_steps["vect"]
    nb = models[mode][vectorizer_type].named_steps["clf"]

    feature_names = np.array(vect.get_feature_names_out())

    print(f"\n====== {title}{(mode)} 클래스별 중요 단어 TOP {top_n} ======")

    for class_index, class_label in enumerate(nb.classes_):
        class_log_prob = nb.feature_log_prob_[class_index]

        top_indices = class_log_prob.argsort()[-top_n:]
        top_words = feature_names[top_indices]

        print(f"\n클래스 {class_label} 대표 단어 TOP {top_n}")
        print(top_words)


get_top_words("count", "basic", top_n=20, title="count_MultinomialNB")
get_top_words("tfidf", "basic", top_n=20, title="Tfid_MultinomialNB")
get_top_words("count", "best", top_n=20, title="count_MultinomialNB")
get_top_words("tfidf", "best", top_n=20, title="Tfid_MultinomialNB")

"""
------------------10번까지 찾을때--------------------
최적 하이퍼파라미터: {'clf__alpha': 0.5, 'vect__min_df': 2, 'vect__ngram_range': (1, 1)}

=== basic 세팅 모델별 Macro-F1 비교 ===
   model  macro_f1
0  count  0.485653
1  tfidf  0.369250

-----------------------------------------

=== best 세팅 모델별 Macro-F1 비교 ===
   model  macro_f1
0  count  0.564962
1  tfidf  0.459898

====== count_MultinomialNBbest 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['저렴하다' '없다' '재질' '좋다' '목' '따갑다' '디자인' '싸다' '그냥' '품질' '크다' '아니다' '가격' '않다'
 '반품' '이다' '사이즈' '옷' '입다' '하다']

클래스 0 대표 단어 TOP 20
['보풀' '세탁' '부분' '보다' '그냥' '품질' '구매' '편하다' '사이즈' '않다' '괜찮다' '옷' '자다' '저렴하다'
 '이다' '가격' '디자인' '좋다' '입다' '하다']

클래스 1 대표 단어 TOP 20
['따뜻하다' '괜찮다' '옷' '않다' '맘' '부드럽다' '이쁘다' '들다' '저렴하다' '편하다' '예쁘다' '자다' '색상'
 '사이즈' '이다' '디자인' '가격' '입다' '하다' '좋다']

====== Tfid_MultinomialNBbest 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['얇다' '가격' '보풀' '재질' '부분' '따갑다' '목' '별로' '그냥' '이다' '않다' '싸다' '품질' '크다'
 '아니다' '반품' '사이즈' '옷' '입다' '하다']

클래스 0 대표 단어 TOP 20
['보풀' '세탁' '부분' '구매' '싸다' '아쉽다' '않다' '사이즈' '그냥' '편하다' '품질' '괜찮다' '이다' '자다'
 '저렴하다' '가격' '좋다' '디자인' '입다' '하다']

클래스 1 대표 단어 TOP 20
['않다' '괜찮다' '대비' '따뜻하다' '맘' '이쁘다' '들다' '저렴하다' '부드럽다' '자다' '편하다' '색상' '사이즈'
 '이다' '예쁘다' '가격' '디자인' '입다' '하다' '좋다']

 

 
 --------------------196번까지 찾을때------------------------------
최적 하이퍼파라미터: {'clf__alpha': 0.5, 'vect__min_df': 2, 'vect__ngram_range': (1, 2)}

classification_report
              precision    recall  f1-score   support

          -1       0.79      0.68      0.73       860
           0       0.47      0.47      0.47       770
           1       0.84      0.88      0.86      2203

    accuracy                           0.75      3833
   macro avg       0.70      0.68      0.69      3833
weighted avg       0.75      0.75      0.75      3833

Macro F1: 0.6866
Fitting 3 folds for each of 8 candidates, totalling 24 fits

===== tfidf_MultinomialNB =====
최적 하이퍼파라미터: {'clf__alpha': 0.5, 'vect__min_df': 2, 'vect__ngram_range': (1, 1)}

classification_report
              precision    recall  f1-score   support

          -1       0.78      0.54      0.64       860
           0       0.43      0.14      0.21       770
           1       0.71      0.97      0.82      2203

    accuracy                           0.70      3833
   macro avg       0.64      0.55      0.56      3833
weighted avg       0.67      0.70      0.66      3833

Macro F1: 0.5565

=== basic 세팅 모델별 Macro-F1 비교 ===
   model  macro_f1
0  count  0.637054
1  tfidf  0.458186

=== best 세팅 모델별 Macro-F1 비교 ===
   model  macro_f1
0  count  0.686621
1  tfidf  0.556543

====== count_MultinomialNBbasic 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['불편하다' '핏' '품질' '허리' '그냥' '별로' '디자인' '아니다' '가격' '없다' '반품' '크다' '좋다' '옷'
 '않다' '이다' '바지' '사이즈' '입다' '하다']

클래스 0 대표 단어 TOP 20
['맞다' '얇다' '길이' '허리' '괜찮다' '자다' '색상' '핏' '옷' '않다' '크다' '가격' '디자인' '이다'
 '편하다' '바지' '사이즈' '입다' '좋다' '하다']

클래스 1 대표 단어 TOP 20
['구매' '따뜻하다' '가볍다' '맘' '이쁘다' '않다' '바지' '들다' '핏' '색상' '예쁘다' '자다' '디자인'
 '사이즈' '가격' '이다' '편하다' '입다' '하다' '좋다']

====== Tfid_MultinomialNBbasic 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['허리' '생각' '디자인' '작다' '가격' '품질' '그냥' '아니다' '불편하다' '없다' '옷' '반품' '이다' '별로'
 '크다' '않다' '바지' '사이즈' '입다' '하다']

클래스 0 대표 단어 TOP 20
['허리' '그냥' '색상' '길이' '아쉽다' '작다' '핏' '옷' '얇다' '괜찮다' '이다' '가격' '디자인' '크다'
 '편하다' '바지' '사이즈' '입다' '좋다' '하다']

클래스 1 대표 단어 TOP 20
['바지' '않다' '시원하다' '맘' '들다' '이쁘다' '따뜻하다' '핏' '가볍다' '색상' '예쁘다' '사이즈' '자다'
 '이다' '디자인' '가격' '편하다' '입다' '하다' '좋다']

====== count_MultinomialNBbest 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['불편하다' '핏' '품질' '허리' '그냥' '별로' '디자인' '아니다' '가격' '없다' '반품' '크다' '좋다' '옷'
 '않다' '이다' '바지' '사이즈' '입다' '하다']

클래스 0 대표 단어 TOP 20
['예쁘다' '얇다' '길이' '허리' '괜찮다' '자다' '색상' '옷' '핏' '않다' '크다' '가격' '디자인' '이다'
 '편하다' '바지' '사이즈' '입다' '좋다' '하다']

클래스 1 대표 단어 TOP 20
['구매' '따뜻하다' '가볍다' '맘' '이쁘다' '않다' '바지' '들다' '핏' '색상' '예쁘다' '자다' '디자인'
 '사이즈' '가격' '이다' '편하다' '입다' '하다' '좋다']

====== Tfid_MultinomialNBbest 클래스별 중요 단어 TOP 20 ======

클래스 -1 대표 단어 TOP 20
['허리' '생각' '디자인' '작다' '가격' '품질' '그냥' '아니다' '불편하다' '없다' '옷' '반품' '이다' '별로'
 '크다' '않다' '바지' '사이즈' '입다' '하다']

클래스 0 대표 단어 TOP 20
['허리' '색상' '그냥' '길이' '작다' '아쉽다' '핏' '얇다' '옷' '괜찮다' '이다' '가격' '디자인' '크다'
 '편하다' '바지' '사이즈' '입다' '좋다' '하다']

클래스 1 대표 단어 TOP 20
['바지' '않다' '시원하다' '맘' '들다' '이쁘다' '따뜻하다' '핏' '가볍다' '색상' '예쁘다' '사이즈' '자다'
 '이다' '디자인' '가격' '편하다' '입다' '하다' '좋다']
"""
