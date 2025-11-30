from sklearn.svm import LinearSVC  # 선형 SVM 분류기(확률 출력 X)
from sklearn.calibration import (
    CalibratedClassifierCV,
)  # SVM 결과를 확률로 보정해 주는 래퍼
from sklearn.pipeline import Pipeline  # 여러 전처리/모델을 순차적으로 묶는 도구
from sklearn.metrics import (
    classification_report,
)  # 분류 리포트(precision, recall, f1 등) 출력
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)  # 데이터 분할과 하이퍼파라미터 탐색
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # 텍스트를 TF-IDF 벡터로 변환
import pandas as pd  # 데이터프레임 처리용 라이브러리
import numpy as np  # 수치 연산용 라이브러리

# 영화 리뷰 데이터 CSV 파일 읽기
df = pd.read_csv(
    "/Users/choimanseon/Documents/multicampus/example/nlp/stat_nlp/text_preprocess/movie_reviews.csv",
    header=0,  # 첫 줄을 헤더로 간주
    names=["id", "review", "label"],  # 컬럼 이름 강제 지정
)

df = df.dropna()  # 결측값(NA) 포함된 행 제거

# label 컬럼을 정수형으로 강제 변환 (0, 1 레이블을 확실히 정수로 맞춰 줌)
df["label"] = df["label"].astype(int)

print(df.head())  # 데이터 상위 5개 행 출력
# print(df["label"].value_counts())                        # 레이블 개수 확인 (필요시만 사용)
print(df["label"].value_counts(normalize=True))  # 레이블 비율 확인(정규화=True)

# 2. 전체 데이터에서 층화추출(stratify)로 샘플 10000개 추출
_, df_sample = train_test_split(
    df,
    test_size=10000,  # 총 데이터 중 10000개를 샘플로 사용
    stratify=df["label"],  # 레이블 비율을 유지하면서 분할
    shuffle=True,  # 섞어서 추출 (기본값이긴 함)
    random_state=42,  # 재현 가능한 결과를 위한 시드 고정
)

print(len(df_sample))  # 샘플 데이터 크기(10000) 출력
print(len(_))  # 나머지 데이터 크기 출력

X = df_sample["review"]  # 입력 텍스트(리뷰)
y = df_sample["label"]  # 정답 레이블(0/1)

# 3. 샘플 10000개를 다시 train/test로 분할
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 전체 중 20퍼센트를 테스트 셋으로 사용
    stratify=y,  # 샘플 내부에서도 레이블 비율 유지
    shuffle=True,  # 섞어서 분할
    random_state=42,  # 시드 고정
)

# 기본 LinearSVC 모델 정의 (확률은 못 내지만 속도가 빠른 선형 SVM)
base_svm = LinearSVC()

# TF-IDF + LinearSVC를 하나의 파이프라인으로 묶기
svm_clf = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer()),  # 1단계: 문장을 TF-IDF 벡터로 변환
        ("svm", base_svm),  # 2단계: 선형 SVM으로 분류
    ]
)

# 기본 모델 학습
svm_clf.fit(x_train, y_train)

# 기본 모델로 테스트 데이터 예측
y_pred_svm = svm_clf.predict(x_test)

print("== 기본 LinearSVC 분류 리포트 ==")
print(classification_report(y_test, y_pred_svm))  # precision/recall/f1 출력

# GridSearch에서 탐색할 하이퍼파라미터 설정
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],  # 유니그램만 vs 유니그램+바이그램
    "tfidf__min_df": [2, 5],  # 최소 등장 문서 수 (희귀 단어 제거 기준)
    "svm__C": [0.1, 1.0, 10.0],  # SVM 규제 강도(C), 클수록 과적합 경향
}

# GridSearchCV로 하이퍼파라미터 탐색 (기본 LinearSVC 파이프라인 대상)
gs = GridSearchCV(
    svm_clf,  # 탐색 대상 모델(파이프라인)
    param_grid=param_grid,  # 탐색할 하이퍼파라미터 그리드
    scoring="f1_macro",  # 클래스 불균형 고려한 매크로 f1 사용
    cv=3,  # 3겹 교차검증
    refit=True,  # 최고 성능 모델로 다시 전체 학습
    n_jobs=-1,  # 가능한 모든 코어 사용
)

# 하이퍼파라미터 탐색 + 학습 수행
gs.fit(x_train, y_train)

print("Best params : ", gs.best_params_)  # 최적 하이퍼파라미터 출력
print("Best macro-f1 : ", gs.best_score_)  # 교차검증에서의 최고 f1-macro

# 최적 하이퍼파라미터로 다시 학습된 파이프라인
best_model = gs.best_estimator_

# 최적 모델로 테스트셋 예측
y_pred = best_model.predict(x_test)

print("== 하이퍼파라미터 튜닝 LinearSVC 분류 리포트 ==")
print(classification_report(y_test, y_pred, digits=3))  # 소수점 3자리까지 리포트

# TF-IDF 벡터라이저와 SVM 분류기 객체 꺼내기
tfidf = best_model.named_steps["tfidf"]  # 파이프라인에서 'tfidf' 단계
clf = best_model.named_steps["svm"]  # 파이프라인에서 'svm' 단계(LinearSVC)

# TF-IDF에서 단어(특징) 이름 배열 가져오기
feature_names = np.array(tfidf.get_feature_names_out())

# LinearSVC의 가중치(coef_) (이진 분류에서는 shape가 (1, n_features))
coef = clf.coef_

# 긍정 클래스(1)에 강하게 기여하는 상위 10개 단어 인덱스 (가중치 큰 순서)
top10_pos = coef[0].argsort()[-10:]

print("== 긍정에 강하게 기여하는 단어 ==")
print(feature_names[top10_pos])  # 인덱스를 단어로 변환해 출력

# 부정 클래스(0)에 강하게 기여하는 상위 10개 단어 인덱스 (가중치 작은 순서)
top10_neg = coef[0].argsort()[:10]

print("== 부정에 강하게 기여하는 단어 ==")
print(feature_names[top10_neg])  # 인덱스를 단어로 변환해 출력

# ==========================
# 여기부터: 확률 예측을 위한 보정된 SVM 파이프라인
# ==========================

# GridSearch에서 찾은 최적 C 값 가져오기
best_C = clf.C  # best_model의 LinearSVC가 가진 C

# 보정용 파이프라인: 같은 TF-IDF 설정 + CalibratedClassifierCV(LinearSVC)
calibrated_clf = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=gs.best_params_["tfidf__ngram_range"],
                min_df=gs.best_params_["tfidf__min_df"],
            ),
        ),
        (
            "svm",
            CalibratedClassifierCV(
                estimator=LinearSVC(C=best_C),  # base_estimator → estimator 로 변경
                cv=3,
                method="sigmoid",
            ),
        ),
    ]
)


# 보정된 파이프라인 학습 (train 데이터 전체 사용)
calibrated_clf.fit(x_train, y_train)

# 확률 예측: 각 샘플에 대해 [부정 확률, 긍정 확률] 반환
probas = calibrated_clf.predict_proba(
    ["지루하고 재미 없었다", "배우의 연기가 영화를 살렸다"]
)

print("== 확률 예측 결과 ==")
print(probas)  # 2행 2열 배열: 각 문장에 대한 클래스별 확률
