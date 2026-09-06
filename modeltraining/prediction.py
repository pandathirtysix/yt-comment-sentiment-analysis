import tensorflow as tf
from tensorflow.keras.models import load_model
import os 


MODEL_PATH = os.path.join(os.getcwd(), "models", "stacked_bilstm.keras")

model = load_model(MODEL_PATH)


def model_predict(data):
    try:
        prediction = model.predict(data)

        return prediction 

    except Exception as e:
        print("model wasn't available", e)

