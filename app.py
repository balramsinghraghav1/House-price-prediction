from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('House_price_pridiction.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['CRIM']),
        float(request.form['ZN']),
        float(request.form['INDUS']),
        float(request.form['CHAS']),
        float(request.form['NOX']),
        float(request.form['RM']),
        float(request.form['AGE']),
        float(request.form['DIS']),
        float(request.form['RAD']),
        float(request.form['TAX']),
        float(request.form['PTRATIO']),
        float(request.form['B']),
        float(request.form['LSTAT']),
    ]

    features_array = np.array([features])
    prediction = model.predict(features_array)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f"Predicted Price: ${output}k")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
