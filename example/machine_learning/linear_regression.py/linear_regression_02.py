# 다중공정성 확인 VIF

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

n = 200
x1 = np.random.normal(size=n)
x2 = 0.9 * x1 + np.random.normal(scale=0.1, size=n)
x3 = np.random.normal(size=n)

df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
print(df.head())


def compute_vif(df_feature: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(df_feature)
    vif_rows = []
    for i in range(X.shape[1]):
        vif_val = variance_inflation_factor(X.values, i)
        vif_rows.append((X.columns[i], vif_val))
    return (
        pd.DataFrame(vif_rows, columns=["features", "VIF"])
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


vif_df = compute_vif(df)

print(vif_df)
"""
x1과 x2의 VIF 값이 매우 높음 (보통 10 이상이면 다중공선성 의심)

이는 x1과 x2가 거의 같은 정보를 담고 있어서
회귀 모델에서 계수 추정의 불안정성을 유발할 수 있음을 의미함.

반면 x3는 x1, x2와 독립적이라 VIF ≈ 1 정도로 정상적."""
