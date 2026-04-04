import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import re
import random


def extract_image_url(img_field):
    if pd.isna(img_field):
        return None
    urls = re.findall(r'https?://[^"]+', str(img_field))

    if not urls:
        return None

    return random.choice(urls)  # 🎯 random ảnh
def show_recommend(df, kmeans, scaler):
    
    st.title("🥗 Gợi ý thực đơn")

    # ======================
    # INPUT USER
    # ======================
    weight = st.number_input("Nhập cân nặng (kg)", 30.0, 150.0, 60.0)
    height = st.number_input("Nhập chiều cao (cm)", 100.0, 220.0, 170.0)
    age = st.number_input("Tuổi", 10, 80, 22)

    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    
    activity = st.selectbox("Mức vận động", [
        "Ít vận động",
        "Trung bình",
        "Nhiều"
    ])
    
    goal = st.selectbox("Mục tiêu", [
        "Giảm cân",
        "Giữ dáng",
        "Tăng cân"
    ])
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

        # ======================
        # BMR
        # ======================
        # ======================
        # BMR
        # ======================
        if gender == "Nam":
            bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
        else:
            bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)
        
        # ======================
        # Activity
        # ======================
        activity_map = {
            "Ít vận động": 1.2,
            "Trung bình": 1.55,
            "Nhiều": 1.9
        }
        
        tdee = bmr * activity_map[activity]
        
        # ======================
        # Goal (QUAN TRỌNG)
        # ======================
        if goal == "Giảm cân":
            calories = tdee - 500
        elif goal == "Tăng cân":
            calories = tdee + 500
        else:
            calories = tdee
        
        # ======================
        # Clamp (giới hạn hợp lý)
        # ======================
        calories = max(1200, min(3500, calories))
        
        # ======================
        # Macro
        # ======================
        fat = calories * 0.25 / 9
        protein = calories * 0.2 / 4
        carbs = calories * 0.5 / 4
        
        sugar = carbs * 0.2
        sodium = max(200, 500 + (bmi - 22) * 20)
        
        user_vector = [[calories, fat, protein, carbs, sugar, sodium]]
    
        # ======================
        # MODEL
        # ======================
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
        features = ['Calories','FatContent','CarbohydrateContent','ProteinContent','SugarContent','SodiumContent']

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

        import matplotlib.pyplot as plt
        import pandas as pd

        top_n = result.head(5)  # lấy 5 món đẹp hơn
        
        for i, row in top_n.iterrows():
        
            st.markdown("---")
        
            # 📝 TÊN MÓN
            st.subheader(f"🍽️ {row['Name']}")
        
            col1, col2 = st.columns([1, 1])
        
            # ======================
            # 🥧 BIỂU ĐỒ TRÒN
            # ======================
            with col1:
                labels = [
                    'Calories', 'Fat', 'Protein',
                    'Carbs', 'Sugar', 'Sodium'
                ]
        
                values = [
                    row['Calories'],
                    row['FatContent'],
                    row['ProteinContent'],
                    row['CarbohydrateContent'],
                    row['SugarContent'],
                    row['SodiumContent']
                ]
        
                fig, ax = plt.subplots()
                # ax.pie(values, labels=labels, autopct='%1.1f%%')
                ax.pie(
                    values,
                    labels=labels,
                    autopct=autopct_func,
                    startangle=90
                )
                ax.axis('equal')  # hình tròn đẹp
                #ax.set_title("Tỷ lệ dinh dưỡng")
        
                st.pyplot(fig)
        
            # ======================
            # 🖼️ HÌNH ẢNH
            # ======================
           
            with col2:
                img_url = extract_image_url(row['Images'])

                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.write("Không có ảnh")
        
            # ======================
            # 📊 INFO NHANH
            # ======================
            st.write(f"🔥 Calories: {row['Calories']}")
            st.write(f"🥩 Protein: {row['ProteinContent']}")
