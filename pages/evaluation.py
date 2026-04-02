import streamlit as st
from sklearn.metrics import silhouette_score

def show_evaluation(df, scaler, kmeans):
    st.title("📊 Đánh giá KMeans")

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
