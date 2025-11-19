from konlpy.tag import Okt

okt = Okt()

text = "이것도 되나욬ㅋㅋㅋㅋ 반갑습니당~~"

# print(okt.morphs(text, norm=False, stem=False))
# print(okt.morphs(text, norm=True, stem=False))
# print(okt.morphs(text, norm=False, stem=True))
# print(okt.morphs(text, norm=True, stem=True))


import re

text = "오늘은 2025-11-16, 가격은 30,000원입니다!!!"
# 1) 모두 소문자 (영문만 있는 경우)
text_lower = text.lower()
# 2) 숫자를 특수 토큰으로 치환 (예: "NUM")
text_num = re.sub(r"\d+", "NUM", text_lower)
# 3) 특수문자 제거 (한글/영문/숫자/공백만 남기기)
text_clean = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text_num)
print(text_clean)
# 예: "오늘은 NUM NUM NUM 가격은 NUM NUM원입니다 "
