import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):

    st.title("🍽️ APP gợi ý thực đơn ")

    # ======================
    # GIỚI THIỆU
    # ======================
    st.markdown("""
    **👨‍🎓 Sinh viên:** Nguyễn Phước Bình  
    **🆔 MSSV:** 21T1020117  

    ---
    ### 🎯 Mục tiêu bài toán

    Xây dựng một hệ thống **gợi ý thực đơn cá nhân hóa** dựa trên:

    - 🔢 Chỉ số cơ thể (**BMI / BMR**)  
    - 🧬 Thông tin cá nhân (**Giới tính, Tuổi, Mức vận động**)  
    - 🏥 Bệnh lý nền (**Tiểu đường, Cao huyết áp**)  

    👉 Hệ thống sử dụng:
    - **KMeans Clustering** để phân nhóm món ăn theo dinh dưỡng  
    - **Rule-based Filtering** để lọc theo bệnh lý  
    - **Distance Ranking** để chọn món phù hợp nhất  

    ---
    ### 💡 Giá trị thực tiễn

    - 🥗 Hỗ trợ lựa chọn thực đơn lành mạnh  
    - ❤️ Giảm nguy cơ bệnh lý liên quan dinh dưỡng  
    - 🧠 Cá nhân hóa chế độ ăn uống  
    - 📱 Có thể ứng dụng vào hệ thống chăm sóc sức khỏe thông minh  
    """)

    # ======================
    # DATA OVERVIEW
    # ======================
    st.subheader("📊 Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)

    col1.metric("🍽️ Số món ăn", len(df))
    col2.metric("📌 Số đặc trưng", df.shape[1])
    col3.metric("⚠️ Giá trị thiếu", df.isnull().sum().sum())

    st.markdown("""
    📌 Bộ dữ liệu bao gồm các thông tin dinh dưỡng của món ăn như:
    - Calories, Fat, Protein, Carbohydrate
    - Sugar, Sodium

    👉 Đây là các đặc trưng quan trọng để phân tích và phân cụm.
    """)

    st.write("🔍 Xem trước dữ liệu:")
    st.dataframe(df.head())

    # ======================
    # PHÂN PHỐI CALORIES
    # ======================
    st.subheader("📈 Phân phối Calories")

    fig, ax = plt.subplots(figsize=(5, 3))
    df['Calories'].hist(bins=50, ax=ax)
    ax.set_xlabel("Calories")
    ax.set_ylabel("Số lượng món ăn")
    
    st.pyplot(fig)

    st.markdown("""
    🔎 **Nhận xét:**

    - Phân phối **lệch phải (right-skewed)** → đa số món có calories trung bình  
    - Một số món có calories rất cao → **outliers**  
    - Điều này cho thấy dữ liệu không cân bằng → cần chuẩn hóa trước khi học máy  
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
    - Fat và Carbohydrate xuất hiện nhiều **outliers**  
    - Protein phân bố ổn định hơn  

    👉 Điều này ảnh hưởng trực tiếp đến KMeans → cần **Scaling (StandardScaler)**  
    """)

    # ======================
    # CORRELATION
    # ======================
    st.subheader("📊 Ma trận tương quan")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df[['Calories','FatContent','ProteinContent',
            'CarbohydrateContent','SugarContent','SodiumContent']].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("""
    🔎 **Nhận xét:**

    - Calories tương quan mạnh với **Fat và Carbohydrate**  
    - Sugar có mối liên hệ chặt với **Carbohydrate**  
    - Sodium gần như độc lập  

    👉 Đây là dấu hiệu tốt cho việc **phân cụm (clustering)**  
    """)

    # ======================
    # TOP MÓN
    # ======================
    st.subheader("🔥 Top món ăn nhiều Calories")

    top_cal = df.sort_values(by='Calories', ascending=False).head(10)

    st.dataframe(top_cal[['Name','Calories']])

    st.markdown("""
    🔎 **Nhận xét:**

    - Các món top thường là đồ chiên, nhiều dầu mỡ  
    - Không phù hợp cho người cần giảm cân hoặc bệnh lý  

    👉 Cần có cơ chế lọc theo sức khỏe (Rule-based)
    """)

    # ======================
    # BMR
    # ======================
    st.subheader("🧠 BMR là gì?")

    st.markdown("""
    **BMR (Basal Metabolic Rate)** là lượng năng lượng cơ thể tiêu hao khi nghỉ ngơi hoàn toàn.  

    👉 Ngay cả khi không vận động, cơ thể vẫn cần năng lượng để:
    - 🫁 Hô hấp  
    - ❤️ Tuần hoàn máu  
    - 🧠 Hoạt động não  
    - 🔥 Duy trì thân nhiệt  

    ---
    👉 BMR phụ thuộc vào:
    - Giới tính  
    - Cân nặng  
    - Chiều cao  
    - Độ tuổi  
    """)

    st.image("my_pages/images/bmr.jpg", caption="Công thức tính BMR", use_container_width=True)

    # ======================
    # CÔNG THỨC
    # ======================
    st.subheader("📐 Công thức tính BMR")

    st.latex(r"""
    \textbf{Nam: } BMR = 88.36 + (13.4 \times weight) + (4.8 \times height) - (5.7 \times age)
    """)

    st.latex(r"""
    \textbf{Nữ: } BMR = 447.6 + (9.2 \times weight) + (3.1 \times height) - (4.3 \times age)
    """)

    # ======================
    # TDEE
    # ======================
    st.subheader("⚡ TDEE là gì?")

    st.markdown("""
    **TDEE (Total Daily Energy Expenditure)** là tổng năng lượng cơ thể tiêu hao mỗi ngày.

    👉 Công thức:
    **TDEE = BMR × Hệ số hoạt động**

    | Mức độ | Hệ số |
    |--------|------|
    | Ít vận động | 1.2 |
    | Trung bình | 1.55 |
    | Nhiều | 1.9 |

    👉 TDEE giúp:
    - Xác định lượng calories cần thiết  
    - Hỗ trợ tăng / giảm / duy trì cân nặng  
    """)

    # ======================
    # KẾT LUẬN
    # ======================
    st.markdown("""
    ## 🧠 Kết luận

    - Dữ liệu có phân phối không đồng đều → cần chuẩn hóa  
    - Một số đặc trưng có tương quan cao → phù hợp cho clustering  
    - Có thể phân nhóm món ăn hiệu quả dựa trên dinh dưỡng  

    ---
    ## 🚀 Pipeline hệ thống

    1. 📊 EDA → hiểu dữ liệu  
    2. 🤖 Clustering → nhóm món ăn  
    3. 🧠 Rule-based → lọc theo bệnh  
    4. 📏 Distance → xếp hạng món ăn  

    👉 Đây là mô hình **Hybrid Recommendation System**

    ---
    ## 🚀 Hướng phát triển

    - Thêm dữ liệu người dùng 
    - Tối ưu KMeans (Elbow Method)  
    - Áp dụng Deep Learning / Recommender nâng cao  
    - Triển khai thành hệ thống thực tế  
    """)
