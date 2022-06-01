# Example script how to load a trained model contained in this SI.zip and predict utilizing the input features.
# To utilize this script adjust the model pkl file path and the feature csv file path.
# The trained ML model requires scaled input features, below is shown how to use the input features to define the scaler.


import pickle 
#pickle version 4.0
import sklearn
#sklearn version 1.0
import numpy as np

#load model pkl
model_pkl = open('models/model_implicit_OROP_B3LYP-D3_KRR_1.pkl', 'rb')
training_indices, model = pickle.load(model_pkl)


#load feature csv
features=np.loadtxt('features/feature_input_implicit_OROP_B3LYP-D3.csv', skiprows=2, delimiter=',')

#scale input features
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(features[training_indices,:])
features_scaled= scaler.transform(features[training_indices,:])

#predict for training set
prediction=model.predict(features_scaled)
