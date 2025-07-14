from data_loader import load_data
from model import DigitClassifier
from utils import handle_error
from predictor import test_prediction

def main():
    print("🧠 Rozpoznawanie cyfr MNIST z użyciem sieci neuronowej")

    try:
        x_train, y_train, x_test, y_test = load_data()
        classifier = DigitClassifier()
        classifier.train(x_train, y_train, x_test, y_test)
        print("✅ Trening zakończony sukcesem.")
        test_prediction()
    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    main()