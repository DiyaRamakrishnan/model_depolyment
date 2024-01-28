from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from grad_cam import GradCAMModel, get_grad_cam  # Import GradCAMModel and get_grad_cam

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to load the Keras model
model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')

# Load the Keras model
model = load_model(model_file_path)

# Load GradCAM model (ensure you have the GradCAMModel and get_grad_cam functions available)
grad_cam_model = GradCAMModel(model, layer_name="conv2d_173")

# Define img_length and img_width as global variables
img_length = 50
img_width = 50

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global img_length, img_width  # Add this line to use the global variables

    if request.method == 'POST':
        img = request.files['image'].read()
        npimg = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_length, img_width))

        # Preprocess the image
        input_data = np.array([img], dtype=np.float32) / 255.0  # Cast to float32

        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0][0] > 0.5:
            result = "True"
        else:
            result = "False"

        # Initialize grad_cam_path before the condition
        grad_cam_path = None

        # Generate Grad-CAM image
        grad_cam_result = get_grad_cam(grad_cam_model, img, class_index=1, img_length=img_length, img_width=img_width)

        # Print the grad_cam_result for debugging
        print(grad_cam_result)

        # Save the Grad-CAM image if it's not None
        if grad_cam_result is not None:
            # Save the Grad-CAM image
            grad_cam_path = os.path.join(script_dir, 'static', 'grad_cam_result.jpg')
            cv2.imwrite(grad_cam_path, grad_cam_result)
        else:
            print("Grad-CAM image is None. Check the image generation process.")

        return render_template('result.html', result=result, probability=prediction[0][0], grad_cam_path=grad_cam_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=45811, debug=True)
