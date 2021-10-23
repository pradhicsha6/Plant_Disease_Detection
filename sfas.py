from flask.helpers import flash
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Defining the flask app
app = Flask(__name__)

# Model saved with Keras model.save()

model = tf.keras.models.load_model('SampleModel.h5', compile=False)


def predict_model(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    predictions = model.predict(x)
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == "POST":
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        conf_pwd = request.form.get('confirm-password')

        if len(email) < 4:
            flash('Email must be greater than 4 characters ', category="error")
        elif password != conf_pwd:
            flash('Two Passwords doesn\'t match', category="error")
        elif len(password) < 8:
            flash("Password must be atleast 8 Characters", category="error")
        else:
            flash("Account Created successfully!", category='success')

    return render_template('register.html')


@app.route('/predict')
def upload():
    return render_template('predict.html')


@app.route('/prediction-result', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        # Getting the file from post request
        f = request.files['plant-img']

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
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
