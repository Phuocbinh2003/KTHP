import streamlit as st
from side1 import sidebar_navigation
from utils.loader import load_data, load_model

from pages.eda import show_eda
from pages.recommend import show_recommend
from pages.evaluation import show_evaluation

# CONFIG
st.set_page_config(page_title="Food Recommendation", layout="wide")

# LOAD
df = load_data()
kmeans, scaler = load_model()

# SIDEBAR
page = sidebar_navigation()

# ROUTING
if page == "1. Giới thiệu & EDA":
    show_eda(df)

elif page == "2. Gợi ý thực đơn":
    show_recommend(df, kmeans, scaler)

elif page == "3. Đánh giá mô hình":
    show_evaluation(df, scaler, kmeans)
