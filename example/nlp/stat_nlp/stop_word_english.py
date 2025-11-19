import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# NLTK 데이터 다운로드 (처음 한 번만 실행)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

# 예시 문장
text = "NLTK is a powerful Python library for natural language processing!"

# 1️⃣ 토큰화
tokens = word_tokenize(text)
print("토큰화 결과:", tokens)

# 2️⃣ 불용어 제거
stop_words = set(stopwords.words("english"))
filtered = [word for word in tokens if word.lower() not in stop_words]
print("불용어 제거 결과:", filtered)

# 3️⃣ 품사 태깅
tagged = pos_tag(filtered)
print("품사 태깅 결과:", tagged)
