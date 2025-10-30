from altair.utils.core import P
import pandas as pd
import numpy as np

name = [
    "홍창기",
    "신민재",
    "오스틴",
    "김현수",
    "문보경",
    "오지환",
    "박동원",
    "박해민",
    "구본혁",
]
subjects = ["C++", "Java", "Python", "JavaScript"]

scores = [np.random.randint(90, 100, 4).tolist() for _ in range(len(name))]
print(scores)

df = pd.DataFrame(scores, columns=subjects)
# df=['stu_nmae']=name
print(df)

df.insert(0, "stu_name", name)
print(df)

df.to_csv("score.csv", index=False)
