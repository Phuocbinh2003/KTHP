import streamlit as st

def sidebar_navigation():
    st.sidebar.title("📌 Điều hướng")

    page = st.sidebar.radio(
        "Chọn trang:",
        [
            "1. Giới thiệu & EDA",
            "2. Gợi ý thực đơn",
            "3. Đánh giá mô hình"
        ]
    )

    return page
