import os, sys
from flask import Flask, render_template, request, redirect, flash, redirect, url_for
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename

from app import app
import pymongo
import secrets as sec
import StyleYourArt

from PIL import Image
import numpy as np

BASE_MODEL, TOP_MODEL = StyleYourArt.models.load_my_models(StyleYourArt.models.MODEL_VERSION, StyleYourArt.models.MODEL_BASE_TAG)


## create flask app
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=16,
    DROPZONE_MAX_FILES=1,
    DROPZONE_DEFAULT_MESSAGE="<img src=\"/static/cloud.png\" width=\"30\"> Drop a file, or Browse.",
    DROPZONE_MAX_FILE_EXCEED="Your can't upload any more files.",
    DROPZONE_REDIRECT_VIEW='predict' 
)

dropzone = Dropzone(app)

## valid extensions checked before upload
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

UPLOAD_FOLDER = 'app/static/uploads/'
STYLE_FOLDER = 'app/static/styles'
PREDICTION_FOLDER = 'app/static/prediction'

app.secret_key = "secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

## check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## home page
@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            ## clear cache if needed
            files = os.listdir(app.config['UPLOAD_FOLDER'])
            for f in files:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

            ## get uploaded file
            filename = secure_filename(file.filename)

            ## save uploaded file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('home.html')
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('home.html')

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) >0:
            filename = os.listdir(app.config['UPLOAD_FOLDER'])[0]
        else:
            filename = None

        if filename:
             ## remove existing predictions
            files = os.listdir(app.config['PREDICTION_FOLDER'])
            for f in files:
                os.remove(os.path.join(app.config['PREDICTION_FOLDER'], f))

            ## make predictions
            image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predicted_class = StyleYourArt.models.make_prediction(BASE_MODEL, TOP_MODEL, image, filename[:-4])

            ## display text
            style_name = predicted_class
            style_text_file = style_name.replace(" ", "_").replace("-", "_")
            style_text_file += ".txt"
            with open(os.path.join(app.config['STYLE_FOLDER'], style_text_file), 'r') as file:
                style_description = file.read()
            plot = 'prediction_{}.png'.format(filename[:-4])
            return render_template('home.html', filename=filename, style_name=style_name, style_description=style_description, plot=plot)
        else:
            return render_template('home.html')
    
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/plot/<filename>')
def display_plot(filename):
    print('plot filename: ' + filename)
    return redirect(url_for('static', filename='prediction/' + filename), code=301)

@app.route('/about')
def about():
    return render_template('about.html')    

@app.route('/styles')
def styles():
    return render_template('styles.html')  

@app.route('/model')
def model():
    return render_template('model.html')  

@app.route('/chronology')
def chronology():
    return render_template('chronology.html')  