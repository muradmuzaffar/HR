import streamlit as st
import pandas as pd

st.set_page_config(page_title="Plotting ", page_icon="📈")


df = pd.read_csv('HR_no_duplicate.csv')



st.line_chart(df.last_evaluation)
