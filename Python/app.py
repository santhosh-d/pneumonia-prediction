from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras import backend
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import tensorflow as tf
from skimage.transform import resize
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model(
    r"C:\Users\reach\Jupyter Notebooks\Intern Project\Python\Final_1_Final.h5")


@app.route("/", methods=["POST", "GET"])
def upload():
    try:
        if(request.method == "POST"):
            f = request.files['image']

            basePath = os.path.dirname(__file__)
            filePath = os.path.join(
                basePath, 'uploads', secure_filename(f.filename))
            f.save(filePath)
            img = image.load_img(filePath, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if(np.max(x) > 1):
                x = x/255.0
            predics = model.predict_classes(x)
            predics = int(predics)

            text = 'none'
            if(predics == 0):
                color = 'green'
                text = "The Person is a Healthy Person!"
            elif(predics == 1):
                color = 'red'
                text = "The Person has Pneumonia! Please consult with Doctor as soon as Possible!"
            back = '{window.history.back();}'
            return f'''
            <html>
    <head>
        <meta charset="utf-8" />
        <title>Pneumonia Prediction System</title>
        <link
        href="//fonts.googleapis.com/css?family=Raleway:400,300,600"
        rel="stylesheet"
        type="text/css"
        />
        <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/skeleton-framework/1.1.1/skeleton.min.css"
        integrity="sha512-zzq8ZrbhP7DGbwClbqSYuo+di3FvAlMNykCKWE6pHuqmCgGjJSISDTzy7QkdI2ajLlgw0nx6hxIu8os2hYuN8w=="
        crossorigin="anonymous"
        />
    </head>
    <body>
        <div class="container">
        <div class="results" style="margin-top: 5%;">
            <h1>Results</h1>
            <strong>
            <h3 style="color: {color};">{text}</h3>
            </strong>
        </div>
        <button
            class="button button-primary"
            style="font-size: large;"
            onclick="goBack()"
        >
            Go Back to Prediction
        </button>

        <script>
            function goBack()
            {back}
        </script>
        </div>
    </body>
    </html>

            '''
        else:
            return render_template('upload.html')
    except:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
