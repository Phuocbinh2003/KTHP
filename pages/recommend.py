import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances

def show_recommend(df, kmeans, scaler):
    st.title("🥗 Gợi ý thực đơn")

    # ======================
    # INPUT USER
    # ======================
    weight = st.number_input("Nhập cân nặng (kg)", 30.0, 150.0, 60.0)
    height = st.number_input("Nhập chiều cao (cm)", 100.0, 220.0, 170.0)

    disease = st.selectbox("Chọn tình trạng", [
        "Bình thường",
        "Tiểu đường",
        "Cao huyết áp"
    ])

    # ======================
    # TÍNH BMI
    # ======================
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    st.info(f"📊 BMI của bạn: {bmi:.2f}")

    # ======================
    # GỢI Ý
    # ======================
    if st.button("Gợi ý món ăn"):

        # Mapping BMI → vector dinh dưỡng (6 features)
        if bmi > 25:
            user_vector = [[1500, 20, 80, 150, 20, 400]]
        elif bmi < 18.5:
            user_vector = [[2500, 60, 100, 300, 40, 800]]
        else:
            user_vector = [[2000, 40, 90, 250, 30, 600]]

        # Scale + predict cluster
        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        df['cluster'] = kmeans.labels_
        result = df[df['cluster'] == cluster].copy()

        # ======================
        # FILTER BỆNH
        # ======================
        if disease == "Tiểu đường":
            result = result[result['CarbohydrateContent'] < 50]
            result = result[result['SugarContent'] < 20]

        elif disease == "Cao huyết áp":
            result = result[result['FatContent'] < 20]
            result = result[result['SodiumContent'] < 500]

        # ======================
        # SORT THEO DISTANCE (QUAN TRỌNG)
        # ======================
        features = [
            'Calories',
            'FatContent',
            'ProteinContent',
            'CarbohydrateContent',
            'SugarContent',
            'SodiumContent'
        ]

        X = result[features]
        X_scaled = scaler.transform(X)

        distances = euclidean_distances(X_scaled, user_scaled)

        result['distance'] = distances

        # sort từ gần → xa
        result = result.sort_values(by='distance')

        # ======================
        # OUTPUT
        # ======================
        st.success(f"👉 Gợi ý {len(result)} món phù hợp (đã xếp hạng)")

        st.dataframe(
            result[['Name','Calories','FatContent','ProteinContent',
                    'CarbohydrateContent','SugarContent','SodiumContent']]
            .head(10)
        )
