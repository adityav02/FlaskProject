from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import Premiumprice_MLR_Deployment as p

#import the prediction file:

app = Flask(__name__)

@app.route('/')
def login():
    #return ("Welcome")
    return render_template("login.html")

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        uname = request.form['username']
        predict = "prediction"
        #print (type(uname))
        #Save it in excel file
        #df = pd.DataFrame([uname.values()], columns=uname.keys())
        #df.to_csv("username.csv",header=uname.keys())    
        #return ("Done")
        return render_template("prediction.html")
        #return redirect(url_for(predict))

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        
        Age = request.form['Age']
        Diabetes = request.form['Diabetes']
        BloodPressureProblems = request.form['BloodPressureProblems']
        #print (BloodPressureProblems)
        AnyTransplants = request.form['AnyTransplants']
        AnyChronicDiseases = request.form['AnyChronicDiseases']
        Height = request.form['Height']
        Weight = request.form['Weight']
        KnownAllergies = request.form['KnownAllergies']
        HistoryOfCancerInFamily = request.form['HistoryOfCancerInFamily']
        NumberOfMajorSurgeries = request.form['NumberOfMajorSurgeries']

        testlist = [[Age,Diabetes,BloodPressureProblems, AnyTransplants,
       AnyChronicDiseases,Height,Weight,KnownAllergies,
       HistoryOfCancerInFamily, NumberOfMajorSurgeries]]

        #print(p.x_train[0][0])

        scaled_testlist = p.scale.transform(testlist)

        newPredictedValue = p.model.predict(scaled_testlist)
        newPredictedValue = p.np.array_str(newPredictedValue)

        #return newPredictedValue
        return render_template("prediction.html",value=newPredictedValue)

if __name__ == '__main__':
    app.run(debug=True)