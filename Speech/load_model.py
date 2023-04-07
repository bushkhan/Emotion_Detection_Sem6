# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from tensorflow.keras.models import  model_from_json
import numpy
import os
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# X, Y = [], []
# for path, emotion in zip(data_path.Path, data_path.Emotions):
#     feature = get_features(path)
#     for ele in feature:
#         print(ele)
#         X.append(ele)
#         # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
#         Y.append(emotion)


# len(X), len(Y), data_path.Path.shape

# # evaluate loaded model on test data
# loaded_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))