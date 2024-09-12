# app.py
import pickle 
import numpy as np
import soundfile
import librosa
import os
from flask import Flask
from flask_mysqldb import MySQL
import mysql_db

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from flask import Flask, render_template, request

app = Flask(__name__)
mysql = MySQL(app)
model=pickle.load(open('D:/Project/Models/rf_model.pkl','rb'))

app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'infantcryclassification'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def process_audio():
    file = request.files['file']
    file_path = os.path.join('Files', file.filename)
    file.save(file_path)

    # Check if file path already exists
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT `prediction` FROM uploaded_files WHERE files = %s', (file_path,))
    result = cursor.fetchone()
    cursor.close()

    if result:
        prediction = result[0]
        print(str(result[0]))
        return render_template('already_uploaded.html',prediction=prediction)

    features = extract_features(file_path)
    result = model.predict(np.array(features).reshape(1,-1))[0]

    # storing to our database
    cursor = mysql.connection.cursor()
    cursor.execute('INSERT INTO uploaded_files (files, prediction) VALUES (%s, %s)', (file_path, result))
    mysql.connection.commit()
    cursor.close()

    return render_template('result.html', prediction=result)

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