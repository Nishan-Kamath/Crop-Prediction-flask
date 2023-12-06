from flask import Flask, render_template, request ,redirect
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    rainfall = request.form.get("rainfall")
    temperature = request.form.get("temperature")
    ph = request.form.get("ph")
    humidity = request.form.get("humidity")

    prediction_data = [float(temperature),float(humidity),float(ph),float(rainfall)]

    df = pd.read_csv('cropdata.csv')

    new_df = df.copy()
    new_df.drop(columns=['N','P','K'],axis=1,inplace=True)
    x = new_df.drop(['label'], axis=1)
    y = new_df['label']

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

    rf_model = RandomForestClassifier()
    rf_model.fit(x_train,y_train)

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(x_train,y_train)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train,y_train)

    prediction_data = [temperature,humidity,ph,rainfall]
    for i in range(len(prediction_data)):
        prediction_data[i] = float(prediction_data[i])

    rf_pred = rf_model.predict([prediction_data])
    lr_pred = lr_model.predict([prediction_data])
    dt_pred = dt_model.predict([prediction_data])

    for i in rf_pred:
        rf_pred = str(i)

    for i in lr_pred:
        lr_pred = str(i)

    for i in dt_pred:
        dt_pred = str(i)

    final_pred = 0

    if(rf_pred == lr_pred or rf_pred == dt_pred):
        final_pred = rf_pred
    else:
        final_pred = lr_pred

    data = final_pred
    return render_template('prediction.html',data=final_pred)


app.run(debug=True)