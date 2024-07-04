# app.py
import pickle 
import numpy as np
import soundfile
import librosa
import os
import mysql_db

from flask import Flask, render_template, request
from flask import Flask
from flask_mysqldb import MySQL
from mysql_db import *

app = Flask(__name__)

# Create a global database connection and cursor
db_conn = connect_to_db()
db_cursor = create_cursor(db_conn)

model=pickle.load(open('D:/Project/Models/svm_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def process_audio():
    file = request.files['file']
    file_path = os.path.join('Files', file.filename)
    file.save(file_path)
    
    # Check if file path already exists
    db_cursor.execute('SELECT `prediction` FROM uploaded_files WHERE files = %s', (file_path,))
    result = db_cursor.fetchone()

    emoji = ""
    if result:
        prediction = result[0]
        if prediction == "Hungry":
            emoji = "🍼"
        elif prediction == "Tired":
            emoji = "😴"
        elif prediction == "Discomfort":
            emoji = "😓"
        elif prediction == "Belly_pain":
            emoji = "😖"
        elif prediction == "Burping":
            emoji = "😊"
        return render_template('already_uploaded.html',prediction=prediction,emoji=emoji)
    
    features = extract_features(file_path)
    result = model.predict(features.reshape(1,-1))[0]
    
    emoji = ""
    if result == "Hungry":
        emoji = "🍼"
    elif result == "Tired":
        emoji = "😴"
    elif result == "Discomfort":
        emoji = "😓"
    elif result == "Belly_pain":
        emoji = "😖"
    elif result == "Burping":
        emoji = "😊"

    # storing to our database
    db_cursor.execute('INSERT INTO uploaded_files (files, prediction) VALUES (%s, %s)', (file_path, result))
    commit_to_db(db_conn)  # Commit the changes
    
    return render_template('result.html', prediction=result,emoji=emoji)

def extract_features(filename):
    with soundfile.SoundFile(filename) as soundFile:
        x = soundFile.read(dtype="float32")
        sampleRate = soundFile.samplerate
        all = np.array([])

        # MFCC -> 40 features
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sampleRate, n_mfcc=40).T, axis=0)
        all = np.hstack((all, mfccs))

        # Mel Spectrogram -> 128 features
        melSpec = np.mean(librosa.feature.melspectrogram(y=x, sr=sampleRate).T, axis=0)
        all = np.hstack((all, melSpec))

        # Spectral contrast -> 7 features
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sampleRate, fmin=20, n_bands=7).T, axis=0)
        all = np.hstack((all, spectral_contrast))

        # Chroma -> 12 features
        chroma = np.mean(librosa.feature.chroma_stft(y=x, sr=sampleRate, n_chroma=12).T, axis=0)
        all = np.hstack((all, chroma))

        # Tonnetz -> 6 features
        tonnetz = np.mean(librosa.feature.tonnetz(y=x, sr=sampleRate * 2, n_chroma=6).T, axis=0)
        all = np.hstack((all, tonnetz))

    return all

if __name__ == '__main__':
    app.run(debug=True)
