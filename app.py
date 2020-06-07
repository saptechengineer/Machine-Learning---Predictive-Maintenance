# Import required Libraries
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

# Flask web server (app)
app = Flask(__name__)

# Read the custom model (pickle file) and load into the application for further use
model=pickle.load(open('model.pkl','rb'))


# Default Route for web UI application
@app.route('/')
def hello_world():
    return render_template("machine_break.html")


# Predict Route which will be called on the form submission from the UI
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('machine_break.html',pred='Machine | Motor MC1 is in Danger!\nProbability of failure/downtime is {}'.format(output))
    else:
        return render_template('machine_break.html',pred='Machine | Motor MC1 is safe.\n Probability of failure/downtime is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
