# IMPORT LIBRARIES

# flask
from flask import Flask, request
from flask_cors import CORS

# major libs
import numpy as np
import pandas as pd

# serialiser
import pickle

###

# LOAD MODEL & SCALER

churn_model_sklearn_pkl = pickle.load(open('churn_model_sklearn.pkl', 'rb'))

scaler_pkl = pickle.load(open('scaler.pkl', 'rb'))

# DEFINE APP AND APP ROUTE

app = Flask(__name__)
CORS(app)

@app.route('/api_predict', methods=['POST', 'GET'])

def api_predict():
    if request.method == 'GET':
        
        return "Please send POST request (Churn Model 2.3, classif thres 0.3)"
    
    elif request.method == 'POST':

        # parse data into pandas dataframe

        print('Hello' + str(request.get_json()))
        data = request.get_json()

        CreditScore = data["CreditScore"]
        Geography = data["Geography"]
        Age = data["Age"]
        Tenure = data["Tenure"]
        Balance = data["Balance"]
        NumOfProducts = data["NumOfProducts"]
        HasCrCard = data["HasCrCard"] 
        IsActiveMember = data["IsActiveMember"]
        EstimatedSalary = data["EstimatedSalary"]

        data = np.array([[CreditScore, Geography, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])

        dataframe = pd.DataFrame(data, columns=["CreditScore", "Geography", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"])

        # apply column transformer to data 

        data_scaled = scaler_pkl.transform(dataframe)

        # apply predict_proba() and classification threshold and get pred
     
        proba_pred = churn_model_sklearn_pkl.predict_proba(data_scaled)

        threshold = 0.3

        if proba_pred[0][1] >= threshold:
            prediction = 1
        elif proba_pred[0][1] < threshold and proba_pred[0][1] > proba_pred[0][0]:
            prediction = 1   
        elif proba_pred[0][1] < threshold and proba_pred[0][1] < proba_pred[0][0]:
            prediction = 0   

        # get confidence score of prediction; remember that output of predict_proba is [[ num, num ]]

        if prediction == 1:
            confidence = proba_pred[0][1]
        elif prediction == 0:
            confidence = proba_pred[0][0]

        # prepare dict for json return

        dict_format = {"prediction" : 0, "confidence" : 0}

        values_list = [prediction, confidence]

        json_return = dict(zip(dict_format, values_list))

        return json_return

if __name__ == "__main__":
    app.run()





