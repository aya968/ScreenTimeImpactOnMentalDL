📌 Stress & Mood Classification Project
Brief Description:
Classifying users’ stress and mood levels based on their daily behaviors (screen time, sleep, TikTok usage, etc.).

🚀 Overview
This project aims to analyze the relationship between daily factors and stress/mood levels, and build predictive classification models:
Stress Class: Low / Medium / High
Mood Class: Low / Medium / High

🧠 Problem Statement
Stress and mood affect productivity and mental health.
It is hard to predict them accurately using traditional methods.
Large-scale data (100k users) requires AI models for precise classification.

💡 Solution
We used users’ daily data (screen time, TikTok usage, sleep, etc.) to build a classification model.
Neural Networks were used to classify:
Stress Class: 3 categories
Mood Class: 3 categories
The data was split into Train/Test (80/20) with One-Hot Encoding for categorical labels.

🏗️ System Architecture
Input: screen_time_hours, hours_on_TikTok, sleep_hours
↓
Preprocessing: feature selection, class encoding
↓
Neural Network Model
↓
Output: stress_class / mood_class

Stress Model: 7 Dense Layers + ReLU + Softmax Output
Mood Model: 7 Dense Layers + ReLU + Softmax Output

🛠️ Technologies Used
Python
TensorFlow/Keras
Pandas, NumPy
Seaborn, Matplotlib
Scikit-Learn

📊 Models / Algorithms
Neural Network Classifier for both Stress and Mood
Activation: ReLU in hidden layers, Softmax in the output layer
Optimizer: Adam
Loss Function: Categorical Crossentropy
Performance:
Stress Model Accuracy: 76%
Mood Model Accuracy: 75%

project-name/

│── notebooks/        # exploratory analysis

│── models/           # saved models

│── README.md

│── requirements.txt

📄 License
This project is for educational purposes.
