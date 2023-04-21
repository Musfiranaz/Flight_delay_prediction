from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

df = pd.read_csv('flightdata.csv')
dl = {}
dl['FL_NUM'] = sorted(df.FL_NUM.unique())


oe = pickle.load(open('oe.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
rc = pickle.load(open('random_forest_classifier.pkl','rb'))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/prediction',methods = ['post'])
def prediction():
    if request.method == 'POST':
        results = request.form
        response = {}
        dic = {}

        for key,value in results.items():
            dic[key] = [value]

        delay = int(dic['DEP_TIME'][0]) - int(dic['CRS_DEP_TIME'][0])
        dic['DEP_DELAY'] = [delay]
        dic['DEP_DEL15'] = [float(delay > 15)]

        df = pd.DataFrame(dic)

        df.FL_NUM = df.FL_NUM.astype('int')
        df.MONTH = df.MONTH.astype('int')
        df.DAY_OF_MONTH = df.DAY_OF_MONTH.astype('int')
        df.DAY_OF_WEEK = df.DAY_OF_WEEK.astype('int')
        df.CRS_ARR_TIME = df.CRS_ARR_TIME.astype('int')
        df.DEP_DELAY = df.DEP_DELAY.astype('float')
        df.DEP_DEL15 = df.DEP_DEL15.astype('float')

        if(dl['FL_NUM'].count(df['FL_NUM'][0]) == 0):
            response['result'] = "No flight have this number"
            return render_template('prediction.html',response=response)

        x = df[['FL_NUM','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','ORIGIN','DEST','CRS_ARR_TIME','DEP_DEL15','DEP_DELAY']]

        x = pd.DataFrame(oe.transform(x),columns=oe.get_feature_names_out())
        x = pd.DataFrame(sc.transform(x),columns=sc.get_feature_names_out())

        y = rc.predict(x)
        if(y):
            response['result'] = "Flight may be delayed"
            response['result_type'] = 'negative'
        else:
            response['result'] = "Flight will be on-time"
            response['result_type'] = 'positive'

        return render_template('prediction.html',response=response)


if __name__ == '__main__':
    app.run(debug=True)