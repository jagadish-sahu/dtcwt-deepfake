import cv2
import numpy as np
import dtcwt
import joblib
from PIL import Image

import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state to store "Actual" and "Predicted" labels
if 'history' not in st.session_state:
    st.session_state.history = [] # Will store dictionaries of {actual, predicted}

# 1. Load the saved model and scaler
model = joblib.load('deepfake_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
transform = dtcwt.Transform2d()

# 2. Define the extraction function (Copy-paste exactly from your Step 5)
def extract_dtcwt_features(image_channel, levels=4):
    coeffs = transform.forward(image_channel, nlevels=levels)
    features = []
    for level in coeffs.highpasses:
        for direction in range(6):
            mag = np.abs(level[:,:, direction])
            features.extend([np.mean(mag), np.var(mag), np.std(mag), np.max(mag), np.min(mag), np.median(mag), np.sum(mag**2)])
            hist, _ = np.histogram(mag, bins=20)
            prob = hist / (np.sum(hist) + 1e-8)
            entropy = -np.sum(prob * np.log2(prob + 1e-8))
            features.append(entropy)
    return features

# 3. Streamlit UI
st.set_page_config(page_title="Deepfake Verifier")
st.title("Deepfake Image Verifier")
st.write("Using Chrominance DT-CWT Analysis")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_display, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess (YCbCr conversion)
    img_resized = cv2.resize(img, (256, 256))
    img_ycc = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
    channels = [c/255.0 for c in cv2.split(img_ycc)]
    
    # Extract & Predict
    all_features = []
    for ch in channels:
        all_features.extend(extract_dtcwt_features(ch))
    
    scaled = scaler.transform(np.array(all_features).reshape(1, -1))
    prediction = model.predict(scaled)[0]
    score = model.decision_function(scaled)[0]



    # Display prediction to user
    st.write(f"Model Prediction: {'DEEPFAKE' if prediction == 1 else 'REAL'}")
    
    # User Verification Section
    st.subheader("Was the model right?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("It was actually REAL"):
            st.session_state.history.append({"actual": 0, "predicted": prediction})
            st.success("Result saved!")
    
    with col2:
        if st.button("It was actually FAKE"):
            st.session_state.history.append({"actual": 1, "predicted": prediction})
            st.success("Result saved!")

    # Show Results
    if prediction == 0:
        st.success(f"REAL (Confidence Score: {abs(score):.2f})")
    else:
        st.error(f"DEEPFAKE (Confidence Score: {abs(score):.2f})")




st.divider()
st.header("Performance Analytics")

if len(st.session_state.history) > 0:
    # Convert history to a DataFrame for easy analysis
    df = pd.DataFrame(st.session_state.history)
    
    # Calculate Counts
    total = len(df)
    correct = (df['actual'] == df['predicted']).sum()
    
    # Display Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analyzed", total)
    m2.metric("Accuracy", f"{(correct/total)*100:.1f}%")
    m3.metric("Fakes Caught", len(df[(df['actual'] == 1) & (df['predicted'] == 1)]))

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['actual'], df['predicted'], labels=[0, 1])
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Real', 'Pred Fake'], 
                yticklabels=['Actual Real', 'Actual Fake'], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # Button to clear history
    if st.button("Clear Analytics Data"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No data yet. Upload an image and verify the result to see analytics.")
