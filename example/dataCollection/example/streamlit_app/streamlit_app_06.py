import streamlit as st
import pandas as pd
import numpy as np

st.title("지도 시각화 예제")
df = pd.DataFrame(
    {
        "lat": np.random.uniform(37.4, 37.6, 100),
        "lon": np.random.uniform(126.8, 127.1, 100),
    }
)
st.map(df)
