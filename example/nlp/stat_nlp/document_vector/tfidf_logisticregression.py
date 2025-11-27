from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = [
    "이 영화 정말 재밌어요 최고예요",
    "별로 재미없고 지루했어요",
    "배우 연기도 좋고 감동적이었어요",
    "스토리가 엉망이고 시간 아까웠어요",
    "완전 최고 강추합니다",
    "돈 아깝고 다시는 보고 싶지 않아요",
]

labels = [1, 0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(x_train_vec, y_train)

y_pred = clf.predict(x_test_vec)
print("y_pred", y_pred)

print(classification_report(y_test, y_pred, digits=3))
