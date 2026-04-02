import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("recipes.csv")

st.title("🥗 Smart Diet App")

height = st.number_input("Chiều cao (cm)", value=170)
weight = st.number_input("Cân nặng (kg)", value=65)

if st.button("Phân tích"):
    bmi = weight / (height/100)**2
    st.success(f"BMI: {bmi:.2f}")

    # 🎯 Xác định mục tiêu
    if bmi < 18.5:
        st.write("👉 Gầy - cần tăng cân")
        filtered = df[df["Calories"] > 300]

    elif bmi < 25:
        st.write("👉 Bình thường")
        filtered = df[(df["Calories"] >= 200) & (df["Calories"] <= 400)]

    else:
        st.write("👉 Thừa cân - cần giảm cân")
        filtered = df[df["Calories"] < 300]

    # 🔥 Gợi ý món
    st.subheader("🍽️ Gợi ý món ăn:")

    if len(filtered) > 0:
        st.dataframe(filtered.head(5))
    else:
        st.write("Không tìm thấy món phù hợp")
