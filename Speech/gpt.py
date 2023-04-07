import numpy as np
import librosa
from tensorflow.keras.models import  model_from_json
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# define the path to the audio file
path = "E:/Specch Emotion Detection/silent.mp3"

# define the function to extract features from the audio file
def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load the audio file
data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

# extract features from the audio file
features = extract_features(data, sample_rate)

# prepare the features for prediction
X = np.expand_dims(features, axis=0)
X = np.expand_dims(X, axis=2)
print("hey",X.shape)
# standardize the features

scaler = StandardScaler()
X = scaler.fit_transform(X)

# make predictions
pred = loaded_model.predict(X)

# convert the predictions to emotion labels
encoder = OneHotEncoder()
encoder.fit([[0], [1], [2], [3], [4], [5], [6], [7]])
emotion_label = encoder.inverse_transform(pred)

# print the predicted emotion label
print("Predicted emotion:", emotion_label[0][0])
