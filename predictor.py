import numpy as np
from model import DigitClassifier
from tensorflow.keras.datasets import mnist

def test_prediction():
    _, _, x_test, y_test = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    classifier = DigitClassifier()
    classifier.load()
    index = 42
    prediction = classifier.predict(x_test[index])
    print(f"🔢 Przewidziana cyfra: {prediction}")