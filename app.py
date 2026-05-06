import cv2
import numpy as np
import dtcwt
import joblib
from PIL import Image
import streamlit as st
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state to store "Actual" and "Predicted" labels
if 'history' not in st.session_state:
    st.session_state.history = [] 

# 1. Load the saved model and scaler
# Ensure these files are uploaded to your GitHub repository
model = joblib.load('deepfake_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
transform = dtcwt.Transform2d()

# 2. DT-CWT Extraction Function (Verbatim from Notebook)
def extract_dtcwt_features(image_channel, levels=4):
    coeffs = transform.forward(image_channel, nlevels=levels)
    features = []
    for level in coeffs.highpasses:
        for direction in range(6):
            mag = np.abs(level[:,:, direction])
            # Basic stats and energy features
            features.extend([np.mean(mag), np.var(mag), np.std(mag), np.max(mag), np.min(mag), np.median(mag), np.sum(mag**2)])
            # Entropy calculation
            hist, _ = np.histogram(mag, bins=20)
            prob = hist / (np.sum(hist) + 1e-8)
            entropy = -np.sum(prob * np.log2(prob + 1e-8))
            features.append(entropy)
    return features

# 3. Streamlit UI Setup
st.set_page_config(page_title="Deepfake Verifier", layout="wide")
st.title("🛡️ Deepfake Image Verifier")
st.write("Analysis based on Chrominance DT-CWT features.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # --- IMAGE PROCESSING ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_display, caption="Uploaded Image", width=400)
    
    # Preprocess (YCbCr conversion & Resizing to 256x256)[cite: 1]
    img_resized = cv2.resize(img, (256, 256))
    img_ycc = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)[cite: 1]
    channels = [c/255.0 for c in cv2.split(img_ycc)]
    
    # --- FEATURE EXTRACTION & PREDICTION ---
    all_features = []
    for ch in channels:
        all_features.extend(extract_dtcwt_features(ch))
    
    # Scale and get decision score[cite: 1]
    scaled = scaler.transform(np.array(all_features).reshape(1, -1))
    score = model.decision_function(scaled)[0]

    # LOGIC UPDATE: > 0 is Real (Class 0), < 0 is Fake (Class 1)
    if score > 0:
        prediction = 0
        st.success(f"**PREDICTION: REAL** (Decision Score: {score:.2f})")
    else:
        prediction = 1
        st.error(f"**PREDICTION: DEEPFAKE** (Decision Score: {score:.2f})")

    # --- USER FEEDBACK (For Confusion Matrix) ---
    st.subheader("Manual Verification")
    st.info("Please tell us the ground truth to update the analytics below:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("This image is actually REAL"):
            st.session_state.history.append({"actual": 0, "predicted": prediction})
            st.rerun()
    
    with col2:
        if st.button("This image is actually FAKE"):
            st.session_state.history.append({"actual": 1, "predicted": prediction})
            st.rerun()

# --- ANALYTICS DASHBOARD ---
st.divider()
st.header("📊 Performance Analytics")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    
    # Calculate Metrics
    total = len(df)
    correct = (df['actual'] == df['predicted']).sum()
    acc = (correct / total) * 100
    fakes_caught = len(df[(df['actual'] == 1) & (df['predicted'] == 1)])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Verified", total)
    m2.metric("System Accuracy", f"{acc:.1f}%")
    m3.metric("Fakes Identified", fakes_caught)

    # Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['actual'], df['predicted'], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Real', 'Pred Fake'], 
                yticklabels=['Actual Real', 'Actual Fake'], ax=ax)
    plt.ylabel('Ground Truth (User)')
    plt.xlabel('Model Prediction')
    st.pyplot(fig)
    
    if st.button("Reset Analytics"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No analytics data available yet. Please verify an image prediction above.")
