

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Fashion MNIST Classifier")

st.title("🧥 Fashion MNIST Classification")
st.write("ارفع صورة رمادية 28x28 لأحد أنواع الملابس")

# تحميل الموديل
model = tf.keras.models.load_model('fashion_mnist_model.keras')

# لابيلز الداتا
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  # عكس الألوان
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    st.image(image, caption="الصورة المعالجة", width=150)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("🎯 النوع المتوقع:")
    st.write(f"**{predicted_class}**")
    st.bar_chart(prediction[0])
