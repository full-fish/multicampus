from konlpy.tag import Okt

okt = Okt()

with open("example/stat_nlp/stopwords-ko.txt", encoding="utf-8") as f:
    basic_stopwords = set(w.strip() for w in f if w.strip())


def korean_tokenize_okt(
    text: str,
    tagger=okt,
    stopwords: set | None = None,
    remove_pos=("Josa", "Eomi", "Punctuation", "Suffix"),
    min_len: int = 1,
):

    if stopwords is None:
        stopwords = basic_stopwords
    tokens = []
    for word, pos in tagger.pos(text, norm=True, stem=True):
        if pos in remove_pos:
            continue
        if word in stopwords:
            continue
        if len(word) < min_len:
            continue
        tokens.append(word)
    return tokens


if __name__ == "__main__":
    sentens = "오늘은 날씨가 너무 좋다"

print(korean_tokenize_okt(sentens))
