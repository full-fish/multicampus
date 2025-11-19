import streamlit as st
import pandas as pd

st.title("학생 점수 분석 대시보드")
uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader(" 데이터 미리보기")
    st.dataframe(df.head())

    subjects = df.columns[1:]
    subject = st.selectbox("과목 선택", subjects)
    st.line_chart(df[subject])
    min_score = st.slider("최소 점수", 0, 100, 50)
    filtered = df[df[subject] >= min_score]
    st.write(f"{subject} {min_score}점 이상 학생:")
    st.table(filtered)

    with st.form("feedback_form"):
        name = st.text_input("이름")
        feedback = st.text_area("피드백")
        submit = st.form_submit_button("저장")
        if submit:
            st.success(f"{name}님의 피드백이 저장되었습니다.")
else:
    st.info("CSV 파일을 업로드해주세요 (예: name, english, math, science)")
