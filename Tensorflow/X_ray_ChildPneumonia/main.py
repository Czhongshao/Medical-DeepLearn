from src.model_evaluator import evaluate_model
from src.model_predictor import predict_model
import warnings
warnings.filterwarnings("ignore")


def main():
    model, history = evaluate_model()
    model.save('models/final/CNN_ChildPneumonia_based_on_Xception.keras')
    predict_model()


if __name__ == "__main__":
    main()

