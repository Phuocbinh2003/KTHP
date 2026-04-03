import streamlit as st

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

        # Mapping BMI → vector dinh dưỡng (quan trọng)
        if bmi > 25:
            user_vector = [[1500, 20, 80, 150, 20, 400]]
        elif bmi < 18.5:
            user_vector = [[2500, 60, 100, 300, 40, 800]]
        else:
            user_vector = [[2000, 40, 90, 250, 30, 600]]
                # Predict cluster
        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        df['cluster'] = kmeans.labels_
        result = df[df['cluster'] == cluster]

        # ======================
        # FILTER BỆNH
        # ======================
        if disease == "Tiểu đường":
            result = result[result['carbs'] < 50]
            result = result[result['sugar'] < 20]

        elif disease == "Cao huyết áp":
            result = result[result['fat'] < 20]
            result = result[result['sodium'] < 500]

        st.success(f"👉 Gợi ý {len(result)} món phù hợp")

        st.dataframe(
            result[['name','calories','fat','protein','carbs']].head(10)
        )
        
