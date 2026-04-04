import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.title("🍽️ Hệ thống gợi ý thực đơn thông minh")

    # ======================
    # GIỚI THIỆU
    # ======================
    st.markdown("""
    **👨‍🎓 Sinh viên:** Nguyễn Phước Bình  .
    **🆔 MSSV:** 21T1020117  

    ---
    ### 🎯 Mục tiêu
    Xây dựng hệ thống gợi ý thực đơn dựa trên:
    - Chỉ số cơ thể (BMI).
    - Bệnh lý nền (Tiểu đường, Cao huyết áp)

    ### 💡 Giá trị thực tiễn
    - Hỗ trợ người dùng lựa chọn món ăn phù hợp sức khỏe  
    - Giảm nguy cơ bệnh lý liên quan đến dinh dưỡng  
    - Ứng dụng trong chăm sóc sức khỏe cá nhân hóa  
    """)

    # ======================
    # DỮ LIỆU
    # ======================
    st.subheader("📊 Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)

    col1.metric("Số món ăn", len(df))
    col2.metric("Số đặc trưng", df.shape[1])
    col3.metric("Thiếu dữ liệu", df.isnull().sum().sum())

    st.write("📌 Xem trước dữ liệu:")
    st.dataframe(df.head())

    # ======================
    # PHÂN PHỐI CALORIES
    # ======================
    st.subheader("📈 Phân phối Calories")

    fig, ax = plt.subplots()
    df['Calories'].hist(bins=50, ax=ax)
    ax.set_xlabel("Calories")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)

    st.markdown("""
    🔎 **Nhận xét:**
    - Phân phối lệch phải (right-skewed)
    - Phần lớn món ăn có mức calories trung bình
    - Có một số món rất cao calories (outliers)
    """)

    # ======================
    # BOXPLOT
    # ======================
    st.subheader("📦 So sánh các thành phần dinh dưỡng")

    fig, ax = plt.subplots()
    sns.boxplot(data=df[['Calories','FatContent','ProteinContent','CarbohydrateContent']], ax=ax)
    st.pyplot(fig)

    st.markdown("""
    🔎 **Nhận xét:**
    - Calories có độ biến thiên lớn nhất  
    - Fat và Carbs có nhiều outliers  
    - Protein phân bố ổn định hơn  
    """)

    # ======================
    # CORRELATION
    # ======================
    st.subheader("📊 Ma trận tương quan")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        df[['Calories','FatContent','ProteinContent','CarbohydrateContent','SugarContent','SodiumContent']].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("""
    🔎 **Nhận xét:**
    - Calories tương quan mạnh với Fat và Carbs  
    - Sugar liên quan chặt với Carbohydrate  
    - Sodium ít liên quan đến các thành phần khác  
    """)

    # ======================
    # TOP MÓN
    # ======================
    st.subheader("🔥 Top món ăn nhiều Calories")

    top_cal = df.sort_values(by='Calories', ascending=False).head(10)
    st.dataframe(top_cal[['Name','Calories']])

    # ======================
    # KẾT LUẬN
    # ======================
    st.markdown("""
    ## 🧠 Kết luận

    - Dữ liệu có sự phân bố không đồng đều giữa các đặc trưng  
    - Một số đặc trưng có tương quan cao → phù hợp cho Clustering  
    - Cần chuẩn hóa dữ liệu trước khi áp dụng KMeans  
    - Có thể sử dụng các đặc trưng dinh dưỡng để phân nhóm món ăn hiệu quả  

    ---
    🚀 **Hướng phát triển:**
    - Bổ sung dữ liệu người dùng (tuổi, giới tính)  
    - Áp dụng mô hình Hybrid Recommendation  
    - Tối ưu hóa trải nghiệm người dùng  
    """)
