import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Load trained model
MODEL_PATH = 'mnist_cnn.h5'  # Update this if you rename the file

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model and save as 'mnist_cnn.h5'")
    st.stop()

model = load_model(MODEL_PATH)

# Streamlit UI
st.title('🖌️ Handwritten Digit Recognizer')
st.markdown('Draw a digit (0-9) below and click Predict!')

# Canvas settings
SIZE = 192
mode = st.checkbox("Enable Draw Mode?", True)

canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=20,
    stroke_color="#FFFFFF",  # White ink
    background_color="#000000",
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Resize to 28x28 and convert to grayscale
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0  # Normalize
    st.subheader("Input Preview (rescaled to 28x28)")
    st.image(cv2.resize(gray, (SIZE, SIZE)), width=SIZE)

    if st.button("Predict"):
        input_tensor = gray.reshape(1, 28, 28, 1)
        prediction = model.predict(input_tensor)
        predicted_digit = np.argmax(prediction)

        st.success(f"🧠 Predicted Digit: **{predicted_digit}**")
        st.subheader("Confidence Levels")
        st.bar_chart(prediction[0])
