import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# * 한글
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")  # 윈도우: 맑은 고딕
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")  # 리눅스

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False
# * 한글

df = sns.load_dataset("penguins")

columns = df.columns.tolist()
columns.insert(0, "선택 안함")
num_columns = df.select_dtypes(include=["number"]).columns.tolist()
num_columns.insert(0, "선택 안함")
graph_kind_dummy = [
    "선택 안함",
    "scatterplot",
    "boxplot",
    "violinplot",
    "histplot",
    "barplot",
    "pairplot",
]
isOK = False

fig, ax = plt.subplots()

# 사이드바단
st.sidebar.title("설정 메뉴")
graph_kind = st.sidebar.radio("그래프 종류 선택", graph_kind_dummy)

if graph_kind != "선택 안함":
    # x축
    if graph_kind != "pairplot":
        axis_x = st.sidebar.selectbox("x축 선택", columns)
    # y축
    if graph_kind not in ["histplot", "pairplot"]:
        axis_y = st.sidebar.selectbox("y축 선택", num_columns)
    color_hue = st.sidebar.selectbox(
        "색상 구분 (선택 사항)", [None, "species", "island", "sex"]
    )
    # pairplot용
    if graph_kind == "pairplot":
        axis_x_list = st.sidebar.multiselect(
            "x축을 선택해주세요.", columns[1:], default=[]
        )
        axis_y_list = st.sidebar.multiselect(
            "y축을 선택해주세요.", columns[1:], default=[]
        )

    isOK = st.sidebar.button("확인")
    
# 메인 단
st.title("penguins 데이터 시각화 대시보드")
st.caption("Seaborn 내장 데이터셋을 이용한 인터랙티브 시각화 예제")

st.header("데이터 미리보기")
st.dataframe(df.head())
if isOK:
    st.header(f"📊 선택된 그래프: {graph_kind}")
    if graph_kind != "pairplot" and axis_x == "선택 안함":
        st.warning("x축을 선택해주세요.")
    elif graph_kind not in ["histplot", "pairplot"] and axis_y == "선택 안함":
        st.warning("y축을 선택해주세요.")
    else:
        if graph_kind == "scatterplot":
            ax.set_title(f"{axis_x} 와 {axis_y}")
            sns.scatterplot(data=df, x=axis_x, y=axis_y, hue=color_hue, ax=ax)
        elif graph_kind == "boxplot":
            ax.set_title(f"{axis_x} 와 {axis_y}")
            sns.boxplot(data=df, x=axis_x, y=axis_y, hue=color_hue, ax=ax)
        elif graph_kind == "violinplot":
            ax.set_title(f"{axis_x} 와 {axis_y}")
            sns.violinplot(data=df, x=axis_x, y=axis_y, hue=color_hue, ax=ax)
        elif graph_kind == "histplot":
            ax.set_title(f"{axis_x} 와 Count")
            sns.histplot(data=df, x=axis_x, hue=color_hue, ax=ax)
        elif graph_kind == "barplot":
            ax.set_title(f"{axis_x} 와 {axis_y}")
            sns.barplot(data=df, x=axis_x, y=axis_y, hue=color_hue, ax=ax)
        elif graph_kind == "pairplot":
            st.subheader("페어플롯 (변수 간 관계)")
            st.pyplot(
                sns.pairplot(
                    data=df, x_vars=axis_x_list, y_vars=axis_y_list, hue=color_hue
                )
            )
            st.stop()
        st.pyplot(fig)

# elif graph_kind == "선택 안함":
#     ax.set_title(f"{axis_x} 와 {axis_y}")
#     sns.barplot(data=df, x=axis_x, y=axis_y, hue=color_hue, ax=ax)
