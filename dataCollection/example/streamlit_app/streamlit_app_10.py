import streamlit as st

st.title(" Layout 예제")
col1, col2 = st.columns(2)
with col1:
    st.subheader("왼쪽")
    st.write("왼쪽 컬럼 내용")
with col2:
    st.subheader("오른쪽")
    st.write("오른쪽 컬럼 내용")
tab1, tab2 = st.tabs(["차트", "표"])
with tab1:
    st.line_chart({"A": [1, 2, 3], "B": [3, 2, 1]})
with tab2:
    st.table({"이름": ["철수", "영희"], "점수": [80, 90]})
