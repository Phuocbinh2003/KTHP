import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px


def show_evaluation(df, scaler, kmeans):

    st.title("📊 Đánh giá mô hình KMeans Clustering")

    st.markdown("""
    Trang này nhằm đánh giá hiệu quả của mô hình **KMeans Clustering** 
    trong việc phân nhóm các món ăn dựa trên đặc trưng dinh dưỡng.

    👉 Mục tiêu:
    - Kiểm tra chất lượng phân cụm
    - Trực quan hóa dữ liệu
    - Phân tích điểm mạnh và hạn chế của mô hình
    """)

    # ======================
    # FEATURES
    # ======================
    st.subheader("🔍 Đặc trưng sử dụng")

    features = [
        'Calories',
        'FatContent',
        'CarbohydrateContent',
        'ProteinContent',
        'SugarContent',
        'SodiumContent'
    ]

    st.write("Các đặc trưng dinh dưỡng được sử dụng để phân cụm:")
    st.write(features)

    # ======================
    # DATA
    # ======================
    st.subheader("📦 Tiền xử lý dữ liệu")

    X = df[features].dropna()
    
    st.write(f"📊 Số lượng mẫu sau khi loại bỏ giá trị thiếu: {X.shape[0]}")
    
    st.markdown("""
    ### 🔧 Chuẩn hóa dữ liệu
    
    Trong bài toán này, các đặc trưng dinh dưỡng như **Calories, Fat, Protein, Carbohydrate, Sugar, Sodium**  
    có đơn vị và thang đo rất khác nhau.
    
    👉 Ví dụ:
    - Calories có thể lên đến hàng nghìn  
    - Fat/Protein chỉ vài chục gram  
    
    ❗ Nếu không chuẩn hóa:
    - Các đặc trưng lớn (như Calories) sẽ **chi phối mô hình KMeans**
    - Làm sai lệch kết quả phân cụm
    
    ---
    
    ### ⚙️ Phương pháp sử dụng
    
    Để giải quyết vấn đề này, em sử dụng **StandardScaler** từ thư viện `sklearn.preprocessing`.
    
  
    
    """)
    
    st.markdown("""
    ### ⚙️ Chuẩn hóa dữ liệu
    
    👉 Công thức chuẩn hóa:
    """)
    
    st.latex(r"X_{scaled} = \frac{X - \mu}{\sigma}")
    
    st.markdown("""
    Trong đó:
    """)
    st.latex(r"  \mu")
    - **\( \mu \)**: Giá trị trung bình của đặc trưng  
    - **\( \sigma \)**: Độ lệch chuẩn  
    
    ---
    st.markdown("""
    ### 🎯 Ý nghĩa
    
    - Đưa các đặc trưng về cùng thang đo (**mean = 0, std = 1**)  
    - Giúp mô hình **KMeans** hoạt động chính xác hơn  
    - Tránh việc một đặc trưng "áp đảo" các đặc trưng khác  
    
    👉 Đây là bước **rất quan trọng** trong các bài toán *clustering*.
    """)

    # scale (QUAN TRỌNG)
    X_scaled = scaler.transform(X.values)

    # predict cluster
    labels = kmeans.predict(X_scaled)

    # ======================
    # METRIC
    # ======================
    st.subheader("📈 Đánh giá bằng Silhouette Score")

    score = silhouette_score(X_scaled, labels)

    st.metric("Silhouette Score", round(score, 3))

    st.markdown("""
    **Giải thích:**
    - Giá trị nằm trong khoảng [-1, 1]
    - Gần 1 → cụm tách biệt tốt
    - Gần 0 → các cụm chồng lấn
    - Âm → phân cụm sai

    👉 Đây là thước đo nội tại (unsupervised), không phản ánh hoàn toàn chất lượng gợi ý thực tế.
    """)

    # ======================
    # PCA 3D
    # ======================
    st.subheader("🌐 Trực quan hóa không gian 3D")

    st.markdown("""
    Do dữ liệu ban đầu có **6 chiều**, ta sử dụng **PCA (Principal Component Analysis)** 
    để giảm chiều xuống 3D nhằm trực quan hóa.

    👉 Lưu ý:
    - PCA giúp giữ lại phần lớn thông tin quan trọng
    - Nhưng vẫn có thể làm mất một phần dữ liệu
    """)

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
        hover_name=df.loc[X.index, 'Name'],
        title="KMeans Clustering Visualization (3D)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # EXPLAINED VARIANCE
    # ======================
    #st.subheader("📊 Mức độ giữ lại thông tin (PCA)")

    #variance = pca.explained_variance_ratio_

    #st.write("Tỷ lệ phương sai giữ lại của từng thành phần:")
    #st.write(variance)

    #st.write(f"Tổng phương sai giữ lại: {round(sum(variance), 3)}")

    # ======================
    # NHẬN XÉT CHUYÊN SÂU
    # ======================
    st.subheader("🧠 Phân tích & Nhận xét")

    st.markdown(f"""
    ### ✅ Điểm mạnh
    - Mô hình có khả năng **phân nhóm món ăn theo dinh dưỡng**
    - Silhouette Score = **{round(score,3)}** cho thấy mức độ phân tách ở mức **khá**
    - PCA cho thấy các cụm có xu hướng tách biệt trong không gian 3D
    - Có thể áp dụng tốt cho hệ thống gợi ý thực đơn

    ### ⚠️ Hạn chế
    - KMeans yêu cầu xác định trước số cụm K
    - Nhạy với outliers và phân phối dữ liệu
    - Chưa phản ánh hoàn toàn nhu cầu cá nhân (BMR, bệnh lý...)

    ### ❗ Vấn đề thực tế
    - Silhouette Score không phản ánh trực tiếp "món ăn có phù hợp người dùng không"
    - Bài toán này cần kết hợp thêm **Rule-based**

    ### 🚀 Hướng cải thiện
    - Thử các thuật toán khác:
        - DBSCAN (tự động cụm)
        - Hierarchical Clustering
    - Kết hợp:
        - Clustering + Rule-based 
        - Clustering + Recommendation System
    
    """)

    # ======================
    # KẾT LUẬN
    # ======================
    st.subheader("🎯 Kết luận")

    st.markdown("""
    Mô hình KMeans đóng vai trò **phân nhóm món ăn**, không phải dự đoán trực tiếp.

    👉 Hệ thống hoàn chỉnh gồm:
    1. Clustering → nhóm món ăn
    2. Rule-based → lọc theo bệnh lý
    3. Distance ranking → chọn món phù hợp nhất

    → Đây là cách tiếp cận **Hybrid Recommendation System**
    """)
