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


def test_sample(path):
    #Path to audio files to be tested
    source   = "testsamples/"   

    #Path where trained models are be saved
    modelpath = "Speakers_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

    #Load the Gaussian gender Models
    models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

    #Reading the test audio file & extracting features
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    #Checking with each model one by one
    for i in range(len(models)):
        gmm    = models[i]  
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[winner])
    time.sleep(1.0)
    return "Sample Detected as %s" % speakers[winner]