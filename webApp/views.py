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
from tensorflow.keras.models import load_model

MODEL_PATH = 'GoogleNet.h5'

views = Blueprint('views', __name__)

model = load_model(MODEL_PATH)


def predict_model(img_path, model,img_file):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Pepper Bell Bacterial Spot"
    elif preds == 1:
        preds = "Pepper Bell Healthy"
    elif preds == 2:
        preds = "Potato Early Blight"
    elif preds == 3:
        preds = "Potato Healthy"
    elif preds == 4:
        preds = "Tomato Tomato Mosaic Virus"
    else:
        preds = "Tomato Healthy"

    return preds
    full_filename = path.join(app.config['UPLOAD_FOLDER'], img_file)
    print(full_filename)
    return render_template("predict.html", user_image = full_filename)


@views.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', user=current_user)


@views.route('/predict', methods=['GET', 'POST'])
@login_required
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
            basepath, 'static\img-uploads', secure_filename(f.filename))
        f.save(file_path)

        # Making prediction by method model_predict
        preds = predict_model(file_path,model,secure_filename(f.filename))
        print(preds)
        result = preds
    return render_template('predict.html', prediction_result=result, user=current_user)
