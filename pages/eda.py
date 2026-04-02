
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.title("🍽️ Hệ thống gợi ý thực đơn")

    st.markdown("""
    **Sinh viên:** Nguyễn Phước Bình 
    **MSSV:** 21T1020117

    ### 🎯 Mục tiêu
    Gợi ý món ăn dựa trên BMI và bệnh lý nền.

    ### 💡 Giá trị thực tiễn
    Giúp người dùng ăn uống lành mạnh, phù hợp sức khỏe.
    """)

    st.subheader("📊 Dữ liệu thô")
    st.dataframe(df.head())

    st.subheader("📈 Phân phối Calories")

    fig, ax = plt.subplots()
    df['calories'].hist(ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Ma trận tương quan")

    fig, ax = plt.subplots()
    sns.heatmap(df[['calories','fat','protein','carbs']].corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### 📝 Nhận xét
    - Dữ liệu có sự chênh lệch giữa các giá trị dinh dưỡng  
    - Calories và fat có tương quan cao  
    - Cần chuẩn hóa trước khi clustering  
    """)
if __name__ == "__main__":
    show_eda()
