import streamlit as st
import numpy as np
import tensorflow as tf

# ==============================
# Load models (h5 format)
# ==============================
model_stress = tf.keras.models.load_model("model/model_stress.h5")
model_mood = tf.keras.models.load_model("model/model_mood.h5")

# ==============================
# App Title & Description
# ==============================
st.set_page_config(page_title="Stress & Mood Prediction", page_icon="🧠", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; color:#1E90FF;'>🧠 Stress & Mood Prediction App</h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Enter your daily activity details below to see your predicted stress and mood levels.
    </p>
    """,
    unsafe_allow_html=True
)

# ==============================
# Input Fields
# ==============================
st.markdown("### Input Your Daily Data")
screen_time_hours = st.number_input("📱 Screen Time Hours (per day)", min_value=0.0, step=0.5)
social_media_code = st.number_input("🌐 Social Media Platform (numeric code)", min_value=0, step=1)
hours_on_tiktok = st.number_input("🎵 Hours on TikTok (per day)", min_value=0.0, step=0.5)
sleep_hours = st.number_input("💤 Sleep Hours (per day)", min_value=0.0, step=0.5)

# ==============================
# Prediction Logic
# ==============================
if st.button("🔮 Predict"):
    # ---- Step 1: Predict Stress ----
    stress_features = np.array([
        screen_time_hours,
        social_media_code,
        hours_on_tiktok,
        sleep_hours
    ]).reshape(1, -1)

    predicted_stress_probs = model_stress.predict(stress_features)
    predicted_stress_class = int(np.argmax(predicted_stress_probs, axis=1)[0])

    stress_labels = {
        0: "Low Stress 😌",
        1: "Medium Stress 😐",
        2: "High Stress 😫"
    }

    st.markdown(
        f"""
        <div style='background-color:#87CEEB; padding:20px; border-radius:12px; 
                    text-align:center; margin-top:15px;'>
            <h3 style='color:white;'> Predicted Stress Level</h3>
            <h2 style='color:white;'>{stress_labels[predicted_stress_class]}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Step 2: Prepare Mood Features using predicted stress ----
    # Use predicted stress class + one of the stress probabilities to make 5 features
    mood_features = np.array([
        screen_time_hours,
        hours_on_tiktok,
        sleep_hours,
        predicted_stress_class,             # replaces manual stress input
        float(predicted_stress_probs[0][predicted_stress_class])  # probability of predicted class
    ]).reshape(1, -1)

    # ---- Step 3: Predict Mood ----
    predicted_mood_probs = model_mood.predict(mood_features)
    predicted_mood_class = int(np.argmax(predicted_mood_probs, axis=1)[0])

    mood_labels = {
        0: "Low Mood 😞 (Very Bad)",
        1: "Medium Mood 😐 (Okay-ish)",
        2: "High Mood 😄 (Very Good)"
    }

    st.markdown(
        f"""
        <div style='background-color:#87CEEB; padding:20px; border-radius:12px; 
                    text-align:center; margin-top:15px;'>
            <h3 style='color:white;'> Predicted Mood</h3>
            <h2 style='color:white;'>{mood_labels[predicted_mood_class]}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
