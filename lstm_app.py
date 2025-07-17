import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# --- Function to load model from .pkl ---
@st.cache_resource
def load_model_from_pickle(pkl_path='lstm_model.pkl'):
    with open(pkl_path, 'rb') as f:
        model_bytes = pickle.load(f)
    temp_h5_path = 'temp_model.h5'
    with open(temp_h5_path, 'wb') as f:
        f.write(model_bytes)
    model = load_model(temp_h5_path, custom_objects={'mse': MeanSquaredError()})
    os.remove(temp_h5_path)
    return model

# --- Streamlit UI ---
st.set_page_config(page_title="LSTM Predictor", page_icon="🔮")

st.title("🔮 LSTM Sequence Predictor")
st.markdown("Enter a 3-number sequence to predict the next number using an LSTM model.")

# Upload model
uploaded_file = st.file_uploader("📤 Upload LSTM Model (.pkl)", type=["pkl"])

if uploaded_file is not None:
    with open("lstm_model.pkl", "wb") as f:
        f.write(uploaded_file.read())
    model = load_model_from_pickle("lstm_model.pkl")
    st.success("✅ Model loaded successfully!")

    # Input 3 numbers
    n1 = st.number_input("Number 1", value=1.0)
    n2 = st.number_input("Number 2", value=2.0)
    n3 = st.number_input("Number 3", value=3.0)

    if st.button("🧠 Predict"):
        input_seq = np.array([[n1, n2, n3]]).reshape((1, 3, 1))
        prediction = model.predict(input_seq)
        st.success(f"🔢 Predicted Next Number: **{prediction[0][0]:.2f}**")

        # Optional: Show sequence plot
        st.line_chart(np.append([n1, n2, n3], prediction[0][0]))
else:
    st.info("Please upload your trained LSTM model in `.pkl` format.")
