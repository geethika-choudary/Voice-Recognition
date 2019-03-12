import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
from sklearn import mixture
from Feature_Extraction import extract_features
import warnings         
warnings.filterwarnings("ignore")


def model_train():
    source   = "./uploads/"
    dest = "Speakers_models/"
    train_file = "trainingDataPath.txt"        
    file_paths = open(train_file,'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()   
        print (path)
    
        # Read the audio
        sr,audio = read(source + path)
    
        # Extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        # When features of 15 files of speaker are concatenated, then do model training
        if count == 15:    
            gmm = GaussianMixture(n_components = 16, covariance_type='diag',n_init = 3)
            gmm.fit(features)
            # Dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            cPickle.dump(gmm,open(dest + picklefile,'wb'))
            print ('Modeling completed for speaker:',picklefile," with data point = ",features.shape  )  
            features = np.asarray(())
            count = 0
        count = count + 1
    return "Modelling completed"