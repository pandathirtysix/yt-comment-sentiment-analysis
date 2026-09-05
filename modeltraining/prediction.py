import tensorflow as tf
from tensorflow.keras.models import load_model
import os 


model = load_model("models\stacked_bilstm.keras")

def model_predict(data):
    try:
        prediction = model.predict(data)

        return prediction 

    except Exception as e:
        print("model wasn't available", e)

