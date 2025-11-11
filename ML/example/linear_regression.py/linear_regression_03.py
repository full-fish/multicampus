# 자기 상관 확인 Durbin-Watson. 전차들이 독립적인지 (자기상관이 있는지) Durbin–Watson 통계량으로 판단
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
x = np.arange(1, 21)

# y = 3 * x + rng.normal(0, 3, size=20)  # 독립 잔차를 더함 2.23
y = 3 * x + np.cumsum(rng.normal(0, 3, size=20))  # 이전까지의 잡음을 누적합함 0.6

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
resid = model.resid
dw = durbin_watson(resid)
print("Durbin-Watson 통계량: ", dw)

# 0~1 양의 자기 상관 (잔차들이 비슷한 방향으로 움직인다)
# 1~3 독립적(이상적: 2근처)
# 3~4 음의; 자기 상관
