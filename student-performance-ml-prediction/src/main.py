
from preprocessing import load_and_preprocess
from train_models import train_and_evaluate

if __name__ == "__main__":
    X, y, _ = load_and_preprocess("../data/student_performance.csv")
    train_and_evaluate(X, y)
