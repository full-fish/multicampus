import streamlit as st
import pandas as pd

st.title("기본 시각화 함수")
month = ["3월", "4월", "5월", "6월"]
df = pd.DataFrame({"영어": [80, 90, 100, 95], "수학": [60, 70, 85, 100]}, index=month)
st.line_chart(df)
st.bar_chart(df)
st.area_chart(df)
