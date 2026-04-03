import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px


def show_evaluation(df, scaler, kmeans):
    st.title("📊 Đánh giá mô hình KMeans")

    # ======================
    # FEATURES (PHẢI ĐÚNG NHƯ LÚC TRAIN)
    # ======================
    features = [
        'Calories',
        'FatContent',
        'CarbohydrateContent',
        'ProteinContent',
        
        'SugarContent',
        'SodiumContent'
    ]

    # ======================
    # DATA
    # ======================
    X = df[features].dropna()

    # ❗ FIX lỗi feature name
    X_scaled = scaler.transform(X.values)

    # predict cluster
    labels = kmeans.predict(X_scaled)

    # ======================
    # METRIC
    # ======================
    score = silhouette_score(X_scaled, labels)

    st.metric("📈 Silhouette Score", round(score, 3))

    # ======================
    # PCA → 3D
    # ======================
    st.subheader("🌐 Visualization 3D (Interactive)")

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # ======================
    # PLOTLY 3D
    # ======================
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=labels.astype(str),
        hover_name=df.loc[X.index, 'Name'],  # hiển thị tên món
        title="KMeans Clustering (3D Interactive)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # EXPLAINED VARIANCE
    # ======================
    st.write("📊 Explained Variance (PCA):")
    st.write(pca.explained_variance_ratio_)

    # ======================
    # NHẬN XÉT
    # ======================
    st.markdown("""
    ### 🧠 Nhận xét x
    - Dữ liệu gốc có 6 chiều → sử dụng PCA để giảm xuống 3D
    - Các cụm được phân tách tương đối rõ ràng
    - Silhouette Score phản ánh chất lượng phân cụm

    ### ⚠️ Hạn chế
    - PCA làm mất một phần thông tin
    - KMeans phụ thuộc vào số cụm K
    - Chưa cá nhân hóa hoàn toàn theo user

    ### 🚀 Hướng cải thiện
    - Tuning số cụm K (Elbow Method)
    - Thử DBSCAN hoặc Hierarchical Clustering
    - Kết hợp Recommendation System nâng cao
    """)
