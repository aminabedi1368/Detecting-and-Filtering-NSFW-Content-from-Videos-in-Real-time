# app/model.py
import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_frame(model, frame):
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    prediction = model.predict(frame)
    return prediction
