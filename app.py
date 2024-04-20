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
model=pickle.load(open('D:/Project/Models/svm_model.pkl','rb'))

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

    emoji = ""
    if result:
        prediction = result[0]
        if prediction == "Hungry":
            emoji = "🍼"
        elif prediction == "Tired":
            emoji = "😴"
        elif prediction == "Discomfort":
            emoji = "😟"
        elif prediction == "Belly_pain":
            emoji = "🤢"
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
        emoji = "😟"
    elif result == "Belly_pain":
        emoji = "🤢"
    elif result == "Burping":
        emoji = "😊"

    # storing to our database
    cursor = mysql.connection.cursor()
    cursor.execute('INSERT INTO uploaded_files (files, prediction) VALUES (%s, %s)', (file_path, result))
    mysql.connection.commit()
    cursor.close()
    
    return render_template('result.html', prediction=result,emoji=emoji)

def extract_features(filename):
    with soundfile.SoundFile(filename) as soundFile:
        x = soundFile.read(dtype="float32")
        sampleRate = soundFile.samplerate
        res = np.array([])

        # Mfcc  
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sampleRate, n_mfcc=13).T, axis=0)
        res = np.hstack((res, mfccs))


        # Mel Spectrogram
        melSpec = np.mean(librosa.feature.melspectrogram(y=x, sr=sampleRate).T, axis=0)
        res = np.hstack((res, melSpec))
    return res

if __name__ == '__main__':
    app.run(debug=True)
