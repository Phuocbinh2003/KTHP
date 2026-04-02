import streamlit as st

def show_recommend(df, kmeans, scaler):
    st.title("🥗 Gợi ý thực đơn")

    bmi = st.number_input("Nhập BMI", 10.0, 50.0, 22.0)

    disease = st.selectbox("Chọn tình trạng", [
        "Bình thường",
        "Tiểu đường",
        "Cao huyết áp"
    ])

    if st.button("Gợi ý món ăn"):

        user_vector = [[bmi, 10, 10, 10]]

        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        df['cluster'] = kmeans.labels_

        result = df[df['cluster'] == cluster]

        if disease == "Tiểu đường":
            result = result[result['carbs'] < 50]

        elif disease == "Cao huyết áp":
            result = result[result['fat'] < 20]

        st.success(f"👉 Gợi ý {len(result)} món phù hợp")

        st.dataframe(result[['name','calories','fat','protein','carbs']].head(10))
