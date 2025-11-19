import streamlit as st
import pandas as pd

st.header("데이터 출력")
df = pd.DataFrame({"이름": ["철수", "영희", "민수"], "점수": [80, 95, 88]})
st.dataframe(df)
st.table(df)
st.caption("출처: 예시 데이터셋")
