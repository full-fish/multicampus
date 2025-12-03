import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import seaborn as sns
import konlpy
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

okt = Okt()

df_list = []
for i in range(1, 11):
    df_temp = pd.read_json(
        rf"D:\multicompus\exam\nlp\stat_nlp\naive_bayes_linearsvc\1-1. 여성의류\1-1.여성의류{i}.json",
        encoding="utf-8",
    )
    df_list.append(df_temp)

df = pd.concat(df_list)
print(df.info())

df = df.dropna(subset=["RawText", "GeneralPolarity"])
df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)

X = df["RawText"]
y = df["GeneralPolarity"]

print("\n레이블 분포(클래스별 샘플 수)")
print(y.value_counts())

with open(
    r"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\stopwords-ko.txt",
    encoding="utf-8",
) as f:
    stopwords = set(w.strip() for w in f if w.strip())


def preprocess_text(text: str) -> list:
    text = text.lower()

    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

    morphs = okt.pos(text, norm=True, stem=True)
    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords and len(word) > 1:
                tokens.append(word)
    return tokens


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("학습데이터 개수 :", len(x_train))
print("테스트데이터 개수:", len(x_test))


def train_and_evaluate(model_name, vectorizer, param_grid=None):
    print("======================")
    print(model_name)

    pipe = Pipeline(
        steps=[
            ("vect", vectorizer),
            (
                "clf",
                MultinomialNB(alpha=1.0),
            ),  # class_prior=[1/3,1/3,1/3], fit_prior=False
        ]
    )

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="f1_macro", refit=True)

    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    gs.fit(x_train, y_train, clf__sample_weight=sample_w)

    print("Best Params : ", gs.best_params_)
    print("Best macro F1 : ", gs.best_score_)

    best_meodel = gs.best_estimator_

    y_pred = best_meodel.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"macro-F1 : {macro_f1:.3f}")

    return best_meodel, macro_f1


param_grid = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    # "vect__max_features" : [1000,2000, 3000],
    # "vect__min_df" : [1,2,3],
    # "clf__alpha" : [0.01, 0.1, 1.0, 10]
}

tfidf = TfidfVectorizer(token_pattern=None, tokenizer=preprocess_text)

tfidf_nb_pipe, tfidf_nb_f1 = train_and_evaluate("MultinomalNB", tfidf, param_grid)

count = CountVectorizer(token_pattern=None, tokenizer=preprocess_text)

cnt_nb_pipe, cnt_nb_f1 = train_and_evaluate("MultinomalNB", count, param_grid)

result = pd.DataFrame(
    {
        "vectorizer": ["TfidfVectorizer", "CountVectorizer"],
        "macro-F1": [tfidf_nb_f1, cnt_nb_f1],
    }
)

print("\nVertorizer 성능비교 : \n", result)


print("\nTfidfVectorizer + MultinomialNB : ")
tfidf_nb = tfidf_nb_pipe.named_steps["vect"]
nb = tfidf_nb_pipe.named_steps["clf"]

feature_names = np.array(tfidf_nb.get_feature_names_out())

for i, class_label in enumerate(nb.classes_):
    log_prob = nb.feature_log_prob_[i]
    top10_idx = log_prob.argsort()[-10:]
    print(f"===클래스 {class_label} 대표 단어(상위 10개)====")
    print(feature_names[top10_idx])


print("\nCountVectorizer + MultinomialNB : ")
cnt_nb = cnt_nb_pipe.named_steps["vect"]
cnb = cnt_nb_pipe.named_steps["clf"]

feature_names = np.array(cnt_nb.get_feature_names_out())

for i, class_label in enumerate(cnb.classes_):
    log_prob = cnb.feature_log_prob_[i]
    top10_idx = log_prob.argsort()[-10:]
    print(f"===클래스 {class_label} 대표 단어(상위 10개)====")
    print(feature_names[top10_idx])
