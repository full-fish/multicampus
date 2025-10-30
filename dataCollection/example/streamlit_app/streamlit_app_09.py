import streamlit as st

st.sidebar.title("설정 메뉴")
option = st.sidebar.selectbox("과목 선택", ["영어", "수학", "과학"])
level = st.sidebar.slider("난이도", 1, 10, 5)
st.title("사이드바 실습")
st.write(f"선택한 과목: {option}")
st.write(f"난이도: {level}")
