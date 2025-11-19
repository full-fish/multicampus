import streamlit as st
import pandas as pd
st.title("Streamlit 기본 출력 예제")
st.header("텍스트 출력")
st.write("`st.write()`는 다양한 자료형을 자동으로 인식합니다.")
st.text("이건 단순 텍스트")
st.subheader("Markdown 예시")
st.markdown("""
- **굵게**
- _기울임_
- [공식문서](https://docs.streamlit.io)
""")