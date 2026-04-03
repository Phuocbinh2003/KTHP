from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_evaluation(df, scaler, kmeans):

    st.title("📊 Đánh giá KMeans")

    features = [
        'Calories',
        'FatContent',
        'ProteinContent',
        'CarbohydrateContent',
        'SugarContent',
        'SodiumContent'
    ]

    X = df[features].dropna()

    # scale
    X_scaled = scaler.transform(X)

    # predict cluster
    labels = kmeans.predict(X_scaled)

    # ======================
    # SILHOUETTE
    # ======================
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, labels)

    st.metric("Silhouette Score", round(score, 3))

    # ======================
    # PCA 3D
    # ======================
    st.subheader("🌐 Visualization 3D (PCA)")

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=labels
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("KMeans Clustering (3D PCA)")

    st.pyplot(fig)

    # ======================
    # GIẢI THÍCH
    # ======================
    st.markdown("""
    ### 🧠 Nhận xét
    - Dữ liệu gốc có 6 chiều nên sử dụng PCA để giảm về 3D
    - Các cụm có sự tách biệt tương đối rõ
    """)

    st.markdown("""
    ### ⚠️ Hạn chế
    - PCA làm mất một phần thông tin
    - KMeans phụ thuộc số cụm K
    """)

    st.markdown("""
    ### 🚀 Hướng cải thiện
    - Tuning số cụm K
    - Dùng DBSCAN / Hierarchical Clustering
    """)
