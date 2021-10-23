from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from . import db
from flask import Flask, redirect, url_for
from werkzeug.utils import secure_filename
import json
from flask.helpers import flash

# import modules for Prediction
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
import numpy as np


views = Blueprint('views', __name__)

model = tf.keras.models.load_model('SampleModel.h5', compile=False)


def predict_model(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    predictions = model.predict(x)
    return predictions


@views.route('/', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html', user=current_user)


@views.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    result = 0
    if request.method == 'POST':
        # Getting the file from post request
        f = request.files['plant-img']

        if f.filename == "":
            flash("Please select a proper image!", category="error")
            pass


        # Saving the file to ./img-uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'img-uploads', secure_filename(f.filename))
        f.save(file_path)

        # Making prediction by method model_predict
        predictions = predict_model(file_path, model)
        print(predictions[0])

        CLASS = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = predictions[0]
        index = np.argmax(a)
        print('Prediction:', CLASS[index])
        result = CLASS[index]
    return render_template('predict.html', prediction_result=result,user=current_user)
