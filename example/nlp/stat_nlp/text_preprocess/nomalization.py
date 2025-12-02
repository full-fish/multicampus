import re

texts = [
    "요즘 딥-러닝 공부하는데 정말 재밌어요",
    "딥  --_  -_ - - 러닝은 어려워서 포기하고 싶다",
    "deep learning 수업이 너무 유익했어요",
    "머신-러닝 개념이 하나도 이해가 안 된다",
    "이 강의 덕분에 딥러닝에 자신감이 생겼다",
    "머신 러닝 과제가 너무 많아서 힘들다",
    "ai 프로젝트 하는 게 기대돼요",
    "이 수업은 지루하고 별로 도움이 안 된다",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)


NORMALIZE_RULES = [
    (r"딥[\s\-]*러닝", "딥러닝"),
    (r"deep[\s\-]*learning", "딥러닝"),
    (r"머신[\s\-]*러닝", "머신러닝"),
    (r"machine[\s\-]*learning", "머신러닝"),
    (r"\bai\b", "인공지능(AI)"),
]


def normalize_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()

    for pattern, repl in NORMALIZE_RULES:
        text = re.sub(pattern, repl, text)

    return text


for t in texts:
    txt = normalize_text(t)
    print(txt)
