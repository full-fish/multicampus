# 선형성 확인 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

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

rng = np.random.default_rng(42)
print(rng)

n = 120


def fit_and_plot_residual(x, y, title, save_prefix=None):
    X = x.reshape(
        -1, 1
    )  # -1은 자동으로 크기를 맞춰라  2차원으로 바꿔야해서. LinearRegression는 2차원만 받음
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    resid = y - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7)
    order = np.argsort(x)
    plt.plot(x[order], y_pred[order])
    plt.title(f"y vs X (+Linear Fit) - {title}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_scatter_fit.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals VS Fitted - {title}")
    plt.xlabel("Predicted (fitted)")
    plt.ylabel("Resiuals (y-y_pre)")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_residual.png")
    plt.show()

    return y_pred, resid, model


xA = rng.uniform(-2, 2, size=n)
yA = 2 + 3 * xA + rng.normal(0, 0.8, size=n)
print("xA", xA)

fit_and_plot_residual(xA, yA, "선형성 만족 그래프")

xB = rng.uniform(-2, 2, size=n)
yB = 2 + 3 * xB + 0.7 * (xB**2) + rng.normal(0, 0.8, size=n)
yB_pred, yB_resid, yB_model = fit_and_plot_residual(xB, yB, "부분 위배(선형모델)")
print(f"Case B - Linear model R^2 : ", r2_score(yB, yB_pred))

poly_model = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin", LinearRegression()),
    ]
)

XB = xB.reshape(-1, 1)
poly_model.fit(XB, yB)
yB_poly_pred = poly_model.predict(XB)
resid_poly = yB - yB_poly_pred

plt.figure(figsize=(6, 4))
plt.scatter(xB, yB, alpha=0.7)
order = np.argsort(xB)
plt.plot(xB[order], yB_poly_pred[order])
plt.title(f"y vs X (+Linear Fit)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(yB_poly_pred, resid_poly, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.title(f"Residuals VS Fitted")
plt.xlabel("Predicted (fitted)")
plt.ylabel("Residuals (y-y_pre)")
plt.tight_layout()
plt.show()
