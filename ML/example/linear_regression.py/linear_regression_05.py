import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro

# 한글
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # 리눅스 계열 (예: 구글코랩, 우분투)
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# ----------------------------------------

np.random.seed(0)
x = np.linspace(1, 20, 50)
noise = np.random.chisquare(df=2, size=50)
y = 3 * x + noise

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
resid = model.resid

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(resid, kde=True, ax=axes[0])
axes[0].set_title("잔차의 분포")

sm.qqplot(resid, line="45", fit=True, ax=axes[1])
axes[1].set_title("Q-Q plot")
plt.tight_layout()
plt.show()

shpiro_test = shapiro(resid)
# 정규분포 아님 -> 대립가설이 맞음
print(f"Shapiro-Wilk 검정 : {shpiro_test.statistic:.3f}, {shpiro_test.pvalue:.5f}")
