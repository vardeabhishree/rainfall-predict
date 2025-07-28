
import gradio as gr
import pickle
import numpy as np

# Load model
with open("rainfall_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_rainfall(input1, input2, input3):
    features = np.array([[input1, input2, input3]])  # update based on your features
    prediction = model.predict(features)
    return f"Predicted Rainfall: {prediction[0]} mm"

# Interface
inputs = [
    gr.Number(label="Feature 1"),
    gr.Number(label="Feature 2"),
    gr.Number(label="Feature 3")
]

output = gr.Textbox(label="Prediction")

gr.Interface(fn=predict_rainfall, inputs=inputs, outputs=output, title="Rainfall Predictor").launch()
