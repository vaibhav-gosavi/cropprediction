# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
xgb = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    data = request.get_json()
    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']
    
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = xgb.predict(input_data)
    crop = label_encoder.inverse_transform(prediction)
    return jsonify({"recommended_crop": crop[0]})

if __name__ == '__main__':
    app.run(debug=True)
