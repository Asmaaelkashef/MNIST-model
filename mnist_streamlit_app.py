

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier")

st.title("✍️ MNIST Digit Recognition")
st.write("ارفع صورة رقم مكتوب بخط اليد (28x28) بالأبيض والأسود")

# تحميل الموديل المدرب
model = tf.keras.models.load_model('mnist_model.keras')

# تحميل الصورة
uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    st.image(image, caption="الصورة المعالجة", width=150)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader("🎯 الرقم المتوقع:")
    st.write(f"**{predicted_digit}**")
    st.bar_chart(prediction[0])
