import streamlit as st
import cv2
import numpy as np
import dtcwt
import joblib # To load trained model/scaler
from PIL import Image

# --- FUNCTIONS FROM YOUR NOTEBOOK ---
def preprocess_for_web(uploaded_file):
    # Convert Streamlit upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycc)
    return [y/255.0, cr/255.0, cb/255.0]

# (Include your extract_dtcwt_features function here exactly as it is in Step 5)

# --- STREAMLIT UI ---
st.title("Deepfake Detector")
st.write("Upload an image to check if it's Real or a Deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing...")
    
    # 1. Preprocess
    channels = preprocess_for_web(uploaded_file)
    
    # 2. Extract (Same logic as Step 11 in your notebook)[cite: 1]
    features = []
    for ch in channels:
        feats = extract_dtcwt_features(ch) # You'll paste the function code here
        features.extend(feats)
    
    # 3. Predict (Requires your saved model and scaler)
    # Note: You should save your model using joblib.dump(model, 'model.pkl') in Colab first
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array) # scaler must be loaded/defined
    prediction = model.predict(scaled_features)[0]
    
    # 4. Show Result
    if prediction == 0:
        st.success("PREDICTION: REAL ")
    else:
        st.error("PREDICTION: DEEPFAKE ")
