from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to load the Keras model
model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')

# Load the Keras model
model = load_model(model_file_path)

@app.route('/')
def home ():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        img = request.files['image'].read()
        npimg = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (50, 50))

        # Preprocess the image
        input_data = np.array([img]) / 255.0

        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0][0] > 0.5:
            result = "True"
        else:
            result = "False"

        return render_template('result.html', result=result, probability=prediction[0][0])

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=0000, debug=True)
