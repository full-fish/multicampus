import numpy as np  # 숫자 연산용 라이브러리
from sklearn.feature_extraction.text import CountVectorizer  # 단어 개수 벡터화
from sklearn.naive_bayes import MultinomialNB  # 다항 나이브베이즈
from sklearn.pipeline import Pipeline  # 벡터화 + 모델 묶기
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.metrics import classification_report  # 평가 지표 출력

texts = [  # 리뷰 문장들
    "서비스가 너무 느리고 불친절했어요",  # 부정 0
    "맛이 없고 양도 너무 적어요",  # 부정 0
    "직원들이 정말 친절하고 음식도 맛있어요",  # 긍정 1
    "가격도 적당하고 분위기가 좋아요",  # 긍정 1
    "다시는 오고 싶지 않아요",  # 부정 0
    "완전 만족스러운 식사였습니다",  # 긍정 1
]

labels = [0, 0, 1, 1, 0, 1]  # 각 문장의 레이블(0=부정, 1=긍정)

x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,  # 테스트 20%
    stratify=labels,  # 클래스 비율 유지
    random_state=42,  # 재현성
)

nb_clf = Pipeline(
    [
        ("vect", CountVectorizer()),  # 문장을 단어 count 벡터로 변환
        ("nb", MultinomialNB(alpha=1.0)),  # 다항 나이브베이즈 모델 + 스무딩(alpha)
    ]
)

nb_clf.fit(x_train, y_train)  # 벡터화 + 모델 학습

y_pred = nb_clf.predict(x_test)  # 테스트 데이터 예측

print(classification_report(y_test, y_pred, digits=3))  # 정밀도/재현율/F1 출력

probs = nb_clf.predict_proba(
    [
        "배송이 빠르고 친절해서 만족스럽다",  # 새 문장 1
        "제품이 고장나서 너무 화가 난다.",  # 새 문장 2
    ]
)
print(probs)  # 각 문장의 [P(0), P(1)] 확률 출력

vect = nb_clf.named_steps["vect"]  # Pipeline에서 CountVectorizer 꺼내기
nb = nb_clf.named_steps["nb"]  # Pipeline에서 MultinomialNB 꺼내기

feature_names = np.array(vect.get_feature_names_out())  # vectorizer가 만든 단어 사전
print("feature_names", feature_names)  # 단어 목록 출력

print("nb.classes_", nb.classes_)  # 모델이 가진 클래스 정보 [0,1]
print("nb.class_log_prior_", nb.class_log_prior_)  # 클래스별 log P(c)
print("np.exp(nb.class_log_prior_", np.exp(nb.class_log_prior_))  # 실제 P(c)로 복원
print(
    "np.exp(nb.feature_log_prob_)", np.exp(nb.feature_log_prob_)
)  # 각 단어의 P(w|c) (클래스별 단어 확률)

for i, class_label in enumerate(nb.classes_):  # 클래스 0/1 반복
    log_prob = nb.feature_log_prob_[i]  # 해당 클래스의 단어별 log P(w|c)
    top10_idx = log_prob.argsort()[-10:]  # 가장 확률 높은 단어 상위 10개
    print(f"=== 클래스 {class_label} 대표 단어 ===")
    print(feature_names[top10_idx])  # top 10 단어 출력
