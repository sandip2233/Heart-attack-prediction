from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('rfc_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        age = int(request.form['age'])
        trestbps=int(request.form['trestbps'])
        chol=int(request.form['chol'])
        thalach=int(request.form['thalach'])

        sex = request.form['sex']
        cp = request.form['cp']
        fbs = request.form['fbs']
        exang = request.form['exang']
        ca = request.form['ca']
        thal=request.form['thal']

        prediction=model.predict([[age,sex,cp,trestbps,chol,fbs,thalach,exang,ca,thal]])
        output=prediction[0]
        if output==1:
            return render_template('index.html',prediction_text="You are Unlikely to have a heart attack")
        if output==0:
            return render_template('index.html',prediction_text="You are likely to have a heart attack ")

if __name__=="__main__":
    app.run(debug=True)
        

            
