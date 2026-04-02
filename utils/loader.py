import pandas as pd
import pickle
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("data/recipes.csv")

@st.cache_resource
def load_model():
    kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return kmeans, scaler
