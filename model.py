from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "mnist_model.h5"

class DigitClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128)
        self.model.save(MODEL_PATH)

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError("Model not found. Train it first.")

    def predict(self, image):
        result = self.model.predict(image.reshape(1, 28, 28, 1))
        return result.argmax()