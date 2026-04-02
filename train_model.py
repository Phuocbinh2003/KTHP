import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv("data/recipes.csv")

# Chọn feature dinh dưỡng (có thể chỉnh lại)
features = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent','SugarContent','SodiumContent']
  
df = df.dropna(subset=features)

X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save
pickle.dump(kmeans, open("models/kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Done training!")
