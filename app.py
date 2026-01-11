from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

MODEL_PATH = 'oct_diagnosis_model.keras'
IMG_SIZE = (128, 128)  

LABEL_MAP = {
    'DRUSEN': 0,
    'CNV': 1,
    'NORMAL': 2,
    'DME': 3
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

CLASS_DESCRIPTIONS = {
    'DRUSEN': 'Yellow deposits under the retina — may indicate age-related changes.',
    'CNV': 'Choroidal neovascularization — presence of abnormal blood vessels and fluid.',
    'NORMAL': 'No signs of retinal disease detected.',
    'DME': 'Diabetic macular edema — fluid accumulation in the macula associated with diabetes.'
}

CLASS_RECOMMENDATIONS = {
    'DRUSEN': 'Monitor retinal changes closely. Schedule regular eye examinations every 3-6 months. Consider antioxidant supplements after consulting with your ophthalmologist.',
    'CNV': 'Seek immediate ophthalmologic attention. Anti-VEGF injections or laser therapy may be recommended. Arrange follow-up imaging within 1-2 weeks.',
    'NORMAL': 'Continue routine eye care. Maintain a healthy lifestyle with regular exercise and balanced diet. Schedule annual eye examinations.',
    'DME': 'Consult an endocrinologist to optimize blood sugar control. Ophthalmologic treatment with injections or laser therapy may be necessary. Follow up within 1-2 weeks.'
}

loaded_model = None

def load_model():
    """Load the pre-trained model"""
    global loaded_model
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from: {MODEL_PATH}")
            return True
        else:
            print(f"Model file not found at: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image the same way as in your Colab notebook"""
    try:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None, "Failed to read image"
        
        img = cv2.resize(img, IMG_SIZE)
        
        img = img / 255.0
        
        img = np.expand_dims(img, axis=-1)
        
        img = np.expand_dims(img, axis=0)
        
        return img, None
    except Exception as e:
        return None, str(e)

def predict_image(image_path):
    """Make prediction on image using loaded model"""
    try:
        if loaded_model is None:
            return None, "Model not loaded"
        
        processed_img, error = preprocess_image(image_path)
        if error:
            return None, error
        
        predictions = loaded_model.predict(processed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        
        predicted_label_name = IDX_TO_LABEL.get(predicted_class_index, "Unknown")
        
        confidence = float(predictions[0][predicted_class_index])
        
        all_predictions = {
            IDX_TO_LABEL.get(i, "Unknown"): float(predictions[0][i])
            for i in range(len(predictions[0]))
        }
        
        return {
            'predicted_class': predicted_label_name,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'description': CLASS_DESCRIPTIONS.get(predicted_label_name, ""),
            'recommendation': CLASS_RECOMMENDATIONS.get(predicted_label_name, ""),
            'error': None
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result, error = predict_image(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'all_predictions': result['all_predictions'],
            'filename': filename
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check if model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded_model is not None,
        'model_path': MODEL_PATH,
        'available_classes': list(IDX_TO_LABEL.values())
    }), 200

@app.before_request
def before_request():
    """Load model before first request"""
    global loaded_model
    if loaded_model is None:
        load_model()

if __name__ == '__main__':

    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
