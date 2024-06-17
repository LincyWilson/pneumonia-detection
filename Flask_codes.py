# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:09:53 2023

@author: CC
"""

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model


app = Flask(__name__)

# Set the template folder path
app = Flask(__name__, template_folder=r"C:/Users/CC/Downloads/templates")

# Load the trained CNN model
model = load_model(r"C:\Users\CC\Downloads\Pneumonia_model3.h5")
 
def preprocess_image(image):

    #Resize the image
    resized_image = image.resize((256, 256))
    
     #Convert the image to RGB
    image = resized_image.convert("RGB")
    
    i = tf.keras.preprocessing.image.img_to_array(image)/255
    
    preprocessed_image = np.array([i])
 
    return preprocessed_image
    
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    try:
        image = Image.open(file)
        preprocessed_image = preprocess_image(image)  # Preprocess the image
        # ...
        return "Image uploaded and processed successfully"
    except:
        return "Error processing image"


@app.route('/')
def home():
    return render_template('index.html')

from flask import render_template

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
     
    file = request.files['file']
    
    image = Image.open(file)
    
    preprocessed_image = preprocess_image(image)
    pred = model.predict(preprocessed_image)
    pred = np.argmax(pred)

    if pred == 0:
        processed_prediction = "The is Normal"
    else:
        processed_prediction = "This is Pneumonia"

    # Render the template with the processed prediction result
    return render_template('prediction_result.html', prediction=processed_prediction)#processed_prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
