from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
model_path = os.path.join('models', 'xgboost_model.pkl')
encoder_path = os.path.join('models', 'label_encoder.pkl')
xgb = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    data = request.get_json()
    N = float(data['N'])
    P = float(data['P'])
    K = float(data['K'])
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])
    
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float32)
    prediction = xgb.predict(input_data)
    crop = label_encoder.inverse_transform(prediction)
    return jsonify({"recommended_crop": crop[0]})

if __name__ == '__main__':
    app.run(debug=True)
