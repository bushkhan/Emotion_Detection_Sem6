import numpy as np
import librosa
from tensorflow.keras.models import  model_from_json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
  


def run(path):

    data, sample_rate = librosa.load(path)
    def noise(data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data
    
                                                                                                                                                
    def stretch(data):
        return librosa.effects.time_stretch(data, rate=0.8)
    
    def shift(data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)
    
    def pitch(data, sampling_rate ):
        return librosa.effects.pitch_shift(data, sr= sample_rate,n_steps=4)
    def extract_features(data):
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
    
    def get_features(path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = extract_features(data)
        result = np.array(res1)
        
        # data with noise
        noise_data = noise(data)
        res2 = extract_features(noise_data)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = stretch(data)
        data_stretch_pitch = pitch(new_data, sample_rate)
        res3 = extract_features(data_stretch_pitch)
        result = np.vstack((result, res3)) # stacking vertically
        
        return result
    
    X = []
    feature = get_features(path)
    print("1st feature",feature.shape)
    for ele in feature:
        X.append(ele)
    
    Features = pd.DataFrame(X)
    # X = Features.iloc[: ,:-1].values
    X = np.array(X)
    print("1st X",X.shape)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = scaler.transform(X)
    
    X = np.expand_dims(X, axis=2)
    
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    feature = scaler.transform(feature)
    feature = np.expand_dims(feature, axis=2)
    
    
    # load json and create model
    json_file = open('E:/Emotion_Detection_Sem6/speech/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("E:/Emotion_Detection_Sem6/speech/model.h5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    loaded_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    pred_test = loaded_model.predict(feature)
    
    
    encoder = OneHotEncoder()
    encoder.fit([[0], [1], [2], [3], [4], [5], [6], [7] ])
    
    emotion_label = encoder.inverse_transform(pred_test)
    print(emotion_label)
    
    emotions = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
    total  =0
    for i in emotion_label:
        total += sum(i)
    total = total // 3
    output = emotions.get(total, emotions[1])
    print("Emotion predicted is: ",emotions.get(total, emotions[1]))


    return output

 
    
