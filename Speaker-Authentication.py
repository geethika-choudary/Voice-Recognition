import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
from Feature_Extraction import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
import sklearn.mixture.gaussian_mixture
from flask import Flask,redirect,url_for,jsonify,flash,request
from werkzeug import secure_filename
from Model_Test import test_sample

app=Flask(__name__)
UPLOAD_FOLDER = 'testsamples/'
ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/authentication-upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.getcwd()+"/"+app.config['UPLOAD_FOLDER'], filename))
            detection_result=test_sample(filename)
            return detection_result
    return '''
<!doctype html>
<title>Upload test File</title>
<h1>Upload Test File</h1>
<form action="" method=post enctype=multipart/form-data>
<p><input type=file name=file>
<input type=submit value=Upload>
</form>
'''

if __name__ == "__main__":
    app.run(debug=True)