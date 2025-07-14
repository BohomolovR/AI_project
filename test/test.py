import unittest
from model import DigitClassifier
import numpy as np

class TestDigitClassifier(unittest.TestCase):
    def test_prediction_output(self):
        classifier = DigitClassifier()
        classifier.load()
        dummy_image = np.zeros((28, 28), dtype=np.float32)
        prediction = classifier.predict(dummy_image)
        self.assertIsInstance(prediction, int)
        self.assertTrue(0 <= prediction <= 9)

if __name__ == '__main__':
    unittest.main()