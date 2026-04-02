import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Food Recommendation", layout="wide")

# ========================
# CACHE
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("data/recipes.csv")

@st.cache_resource
def load_model():
    kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return kmeans, scaler

df = load_data()
kmeans, scaler = load_model()

# ========================
# SIDEBAR NAVIGATION
# ========================
page = st.sidebar.radio("📌 Điều hướng", [
    "1. Giới thiệu & EDA",
    "2. Gợi ý thực đơn",
    "3. Đánh giá mô hình"
])

# ========================
# PAGE 1 - EDA
# ========================
if page == "1. Giới thiệu & EDA":

    st.title("🍽️ Hệ thống gợi ý thực đơn")

    st.markdown("""
    **Sinh viên:** Bình Nguyễn Phước  
    **MSSV:** XXXXX  

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

# ========================
# PAGE 2 - MODEL
# ========================
elif page == "2. Gợi ý thực đơn":

    st.title("🥗 Gợi ý thực đơn")

    bmi = st.number_input("Nhập BMI", 10.0, 50.0, 22.0)

    disease = st.selectbox("Chọn tình trạng", [
        "Bình thường",
        "Tiểu đường",
        "Cao huyết áp"
    ])

    if st.button("Gợi ý món ăn"):

        # Giả lập vector người dùng
        user_vector = [[bmi, 10, 10, 10]]  # bạn có thể cải tiến

        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        df['cluster'] = kmeans.labels_

        result = df[df['cluster'] == cluster]

        # Filter theo bệnh
        if disease == "Tiểu đường":
            result = result[result['carbs'] < 50]

        elif disease == "Cao huyết áp":
            result = result[result['fat'] < 20]

        st.success(f"👉 Gợi ý {len(result)} món phù hợp")

        st.dataframe(result[['name','calories','fat','protein','carbs']].head(10))

# ========================
# PAGE 3 - EVALUATION
# ========================
elif page == "3. Đánh giá mô hình":

    st.title("📊 Đánh giá KMeans")

    from sklearn.metrics import silhouette_score

    features = ['calories','fat','protein','carbs']
    X = df[features].dropna()

    X_scaled = scaler.transform(X)

    labels = kmeans.predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    st.metric("Silhouette Score", round(score, 3))

    st.markdown("""
    ### 🧠 Nhận xét
    - Score càng gần 1 càng tốt  
    - Mô hình hoạt động ở mức khá  
    """)

    st.markdown("""
    ### ⚠️ Hạn chế
    - Chưa cá nhân hóa sâu
    - Chưa dùng nhiều feature
    """)

    st.markdown("""
    ### 🚀 Hướng cải thiện
    - Thêm dữ liệu user
    - Dùng Deep Learning / Hybrid Recommendation
    """)
