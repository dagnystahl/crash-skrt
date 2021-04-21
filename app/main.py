'''***************'''
'''  API Imports  '''
'''***************'''
from flask import Flask, jsonify, request
import json

'''**************'''
'''  ML Imports  '''
'''**************'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

'''********************************'''
'''  Data Cleaning/Model Training  '''
'''********************************'''
app = Flask(__name__)

df_targets = pd.read_csv('labels.csv')
df_targets = df_targets.drop(columns=["Unnamed: 0"])

df_features = pd.read_csv('features.csv')
df_features = df_features.drop(columns=["Unnamed: 0"])

X_train, X_test, y_train, y_test = train_test_split(df_features, df_targets, test_size=0.2, random_state=0)

#create new a knn model
knn_model = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn_model, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train, y_train)
n_neighbors = knn_gscv.best_params_['n_neighbors']

model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

def predict_casualties(features):
  pd.Series(features).to_frame()
  inst_df = pd.DataFrame.from_dict(features)
  inst_df["BICYCLE_IND"] = inst_df["BICYCLE_IND"] / 4.0
  inst_df["PEDESTRIAN_IND"] = inst_df["PEDESTRIAN_IND"] / 4.0
  predictions = model.predict(inst_df)
  return predictions

'''***************'''
'''  API Request  '''
'''***************'''
@app.route("/", methods=['POST'])
def new_request():
        #JSON that is sent
        features = request.json
        predictions_float = predict_casualties(features)
        predictions_bool = [False if p == 0 else True for p in predictions_float]
        return predictions_bool