import streamlit as st

st.title(" 입력 위젯 예제")
name = st.text_input("이름을 입력하세요")
age = st.number_input("나이 입력", min_value=1, max_value=100)
color = st.color_picker("좋아하는 색상 선택")
agree = st.checkbox("개인정보 제공에 동의합니다")
if st.button("제출"):
    if agree:
        st.success(f"{name}님({age}세)의 좋아하는 색상은 {color}입니다.")
    else:
        st.warning("동의하지 않으면 제출할 수 없습니다.")
