"An example of predicting a music genre from a custom audio file"
import librosa
import logging
import sys
import numpy as np
import glob
import os
import csv
# import pandas as pd
from keras.models import model_from_json
import warnings
warnings.filterwarnings('ignore')

from New_GenreFeatureData import (
    New_GenreFeatureData,
)  


# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    

def load_model(model_path, weights_path):
    # print("Load the trained model...")
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="mse",
     optimizer="adam", metrics=["accuracy"]
    )
    return trained_model


def extract_audio_features(file):
    
    # print("Extracting features...")

    timeseries_length = 128 
    
    features = np.zeros((1, timeseries_length, 40), dtype=np.float64) 

    y, sr = librosa.load(file)
  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512) 

    features[0, :, 0:20] = mfcc.T[0:timeseries_length, :]
    features[0, :, 20:21] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 21:33] = chroma.T[0:timeseries_length, :]
    features[0, :, 33:40] = spectral_contrast.T[0:timeseries_length, :] 

    return features


def get_genre(model, music_path):
    
    prediction = model.predict(extract_audio_features(music_path))
    predict_genre = prediction
    predict_genre=float(predict_genre)
 
    return predict_genre
  
 
def predict():
    #Create a csv file to save the results
    csv_file='results.csv'

    with open(csv_file, 'w', newline="") as csvfile:
        fieldnames = ['File_name', 'Predicted_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        count=0
        
        #Give path of the folder containing the .wav files
        for file_path in glob.glob('./f1/_test/*.wav'):
            
            count=count +1
            print(file_path)
            print("file no:",count)

            #Load the model
            MODEL = load_model("./model1_final_mfccall_2l_f2.json","./model1_final_mfccall_2l_f2.h5")  
             
            GENRE = get_genre(MODEL, file_path)   

            if GENRE==None:
                print("Prediction couldn't be made for:",file_path)
                continue

            
            predicted_class= GENRE
            print("predicted class:",predicted_class)
    
            writer.writerow({'File_name': file_path, 'Predicted_score': predicted_class})



def predict_single():

   #Give the path to the .wav file to be predicted
   PATH = sys.argv[1] if len(sys.argv) == 2 else "./f1/_test/1.08_audio.interview_22_1.wav" 

   #Load the trained model files
   MODEL = load_model("./model1_final_mfccall.json","./model1_final_mfccall.h5") 

   GENRE = get_genre(MODEL, PATH)
   print("Model predict: {}".format(GENRE))


if __name__ == "__main__":
      

    #For single file prediction
    predict_single()

    #For multiple file prediction. Load the audios  
    # predict()


