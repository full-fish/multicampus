import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.title("Matplotlib & Seaborn 예제")
df = sns.load_dataset("tips")
fig, ax = plt.subplots()
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=df, ax=ax)
st.pyplot(fig)
