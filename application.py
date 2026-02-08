from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# import ridge regressor adn standard scaler pickle
ridge_model=pickle.load(open('MODELS/ridge_reg.pkl','rb'))
standard_scaler=pickle.load(open('MODELS/scaler.pkl','rb'))

application=Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoints():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Region_Code=float(request.form.get("Region_Code"))
        Class_Code=float(request.form.get("Classes_Coded"))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Region_Code,Class_Code]])
        result=ridge_model.predict(new_data_scaled)

        return render_template("home.html",result=result[0])

    else:
        return  render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")