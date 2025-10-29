import streamlit as st

st.title("Form 활용 예제")
with st.form("survey_form"):
    name = st.text_input("이름")
    gender = st.radio("성별", ["남성", "여성"])
    score = st.slider("만족도", 0, 10, 5)
    submitted = st.form_submit_button("제출")
if submitted:
    st.success(f"{name}({gender})님의 만족도: {score}/10")
