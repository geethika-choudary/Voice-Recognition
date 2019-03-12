import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
from sklearn import mixture
from Feature_Extraction import extract_features
import warnings         
warnings.filterwarnings("ignore")
from flask import Flask,redirect,url_for,jsonify,flash,request,render_template,send_from_directory
from werkzeug import secure_filename
from Model_Train import model_train


app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/Varshini-002/'
app.config['ALLOWED_EXTENSIONS'] = set(['wav', 'mp3', 'mp4'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    count=0
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
            count=count+1 
            if count==15:
                training_result=model_train()
                return training_result   
    return render_template('upload.html', filenames=filenames)

if __name__ == '__main__':
    app.run(debug=True)