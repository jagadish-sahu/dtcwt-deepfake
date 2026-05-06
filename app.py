import streamlit as st
import cv2
import numpy as np
import dtcwt
import joblib
from PIL import Image

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

    # Show Results
    if prediction == 0:
        st.success(f"REAL (Confidence Score: {abs(score):.2f})")
    else:
        st.error(f"DEEPFAKE (Confidence Score: {abs(score):.2f})")
