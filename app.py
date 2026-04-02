import streamlit as st

st.title("🥗 Smart Diet App")

height = st.number_input("Chiều cao (cm)", value=170)
weight = st.number_input("Cân nặng (kg)", value=65)

if st.button("Tính BMI"):
    bmi = weight / (height/100)**2
    st.success(f"BMI: {bmi:.2f}")

    if bmi < 18.5:
        st.write("👉 Gầy - nên tăng cân")
    elif bmi < 25:
        st.write("👉 Bình thường")
    else:
        st.write("👉 Thừa cân - nên giảm cân")
