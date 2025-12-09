import cv2
import os
import mahotas
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
import io
import joblib
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import json

app = Flask(__name__)
CORS(app)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_path = os.path.join("model_v2.keras")
scaler_path = os.path.join("scaler_v2.save")

model = None
scaler = None

def get_model():
    global model
    if model is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        import tensorflow as tf
        
        # Limit memory usage
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        
        import numpy as np
        dummy_input = np.zeros((1, 532))
        model.predict(dummy_input, verbose=0)
    return model

def get_scaler():
    global scaler
    if scaler is None:
        scaler = joblib.load(scaler_path)
    return scaler

# Extract Features
fixed_size = (300, 300)
bins = 8

class_names = [
    "Bacterial Blight (Xanthomonas oryzae pv. oryzae)",
    "Brown Spot (Bipolaris oryzae)",
    "Healthy (No Pathogen Detected)",
    "Leaf Blast (Magnaporthe oryzae)"
]

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(0)
    return haralick

def fd_histogram(image, bins=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def pre_process(image_file):
    img = Image.open(io.BytesIO(image_file.read()))
    
    img = np.array(img)
    
    img = cv2.resize(img, fixed_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fv_hu = fd_hu_moments(img)
    fv_haralick = fd_haralick(img)
    fv_hist = fd_histogram(img, bins=bins)

    global_features = np.hstack([fv_hist, fv_haralick, fv_hu])
    global_features = global_features.reshape(1, -1)

    scaler = get_scaler()
    featured_scaled = scaler.transform(global_features)

    model = get_model()
    prediction_probs = model.predict(featured_scaled, verbose=0)
    prediction_index = np.argmax(prediction_probs)
    prediction_label = class_names[prediction_index]
    confidence = float(prediction_probs[0][prediction_index])
    
    del img, global_features, featured_scaled, prediction_probs

    return prediction_label, confidence

def ai_gemini(disease_name):
    prompt = f"""
    Kamu adalah ahli patologi tanaman padi (Rice Plant Pathologist).
    Penyakit terdeteksi: "{disease_name}".
    
    Berikan informasi singkat dan padat dalam Format JSON (tanpa markdown ```json) dengan key berikut:
    1. "symptoms": Jelaskan ciri-ciri fisik pada daun/tanaman.
    2. "favorable_conditions": Lingkungan seperti apa yang memicu penyakit ini (suhu, kelembaban, dll).
    3. "management": Cara penanganan, pencegahan, atau pengobatan (kimiawi/organik).
    
    Jawab kurang lebih dalam 2-5 kalimat saja.

    Jawab dalam Bahasa Inggris !.
    """

    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    response = model_ai.generate_content(prompt)

    text_response = response.text.replace("```json", "").replace("```", "").strip()

    data = json.loads(text_response)

    return data


@app.route("/health", methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=['POST'])
def predict():
    print(request.files)
    image_file = request.files['file']

    prediction_label, confidence = pre_process(image_file)

    ai_info = ai_gemini(prediction_label)

    return jsonify({
        'prediction' : prediction_label,
        'confidence' : confidence,
        'ai_analysis' : ai_info
    })

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True) 
