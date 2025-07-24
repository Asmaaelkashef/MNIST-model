# ✍️ Handwritten Digit & Fashion Item Classifier using Keras (MNIST & Fashion MNIST)

This project includes two deep learning models trained using Keras and TensorFlow:

1. **MNIST Digit Classifier** – Classifies handwritten digits (0–9).
2. **Fashion MNIST Classifier** – Classifies fashion items (e.g., shirts, sneakers, bags).

## 📊 Datasets Used

- **MNIST**
  - 70,000 grayscale images (28x28 pixels)
  - 10 classes (digits 0–9)

- **Fashion MNIST**
  - 70,000 grayscale images (28x28 pixels)
  - 10 fashion categories:
    - T-shirt/top
    - Trouser
    - Pullover
    - Dress
    - Coat
    - Sandal
    - Shirt
    - Sneaker
    - Bag
    - Ankle boot

## 🏗️ Model Architecture

- Input Layer: Flatten 28x28 image to 784
- Hidden Layers: Dense + ReLU + Dropout
- Output Layer: Dense with Softmax activation

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
