"""
Flask Web Application for Heart Disease Predictor.
Modular UI component that can be integrated into fullstack applications.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import HeartDiseasePredictor, predict_from_dict

app = Flask(__name__)
CORS(app)  # Enable CORS for fullstack integration

# Global predictor instance (loaded once on startup)
predictor = None

def init_predictor():
    """Initialize the predictor on startup."""
    global predictor
    try:
        predictor = HeartDiseasePredictor()
        print(f"✓ Predictor initialized: {predictor.model_name}")
        return True
    except FileNotFoundError as e:
        print(f"✗ Error initializing predictor: {e}")
        print("  Please train a model first: python main.py train")
        return False

@app.route('/')
def index():
    """Render the main prediction form."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy' if predictor else 'error',
        'model_loaded': predictor is not None,
        'model_name': predictor.model_name if predictor else None
    }
    return jsonify(status)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions.
    Can be called from frontend or external applications.
    
    Request body (JSON):
    {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    Response (JSON):
    {
        "success": true,
        "prediction": 1,
        "prediction_label": "Heart Disease",
        "probability": 0.85,
        "probabilities": {
            "No Heart Disease": 0.15,
            "Heart Disease": 0.85
        }
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train a model first.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Make prediction
        result = predict_from_dict(data)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    """
    Batch prediction endpoint.
    Accepts array of patient data.
    
    Request body (JSON):
    {
        "patients": [
            {"age": 63, "sex": 1, ...},
            {"age": 45, "sex": 0, ...}
        ]
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train a model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request. Expected {"patients": [...]}'
            }), 400
        
        patients = data['patients']
        results = []
        
        for i, patient_data in enumerate(patients):
            try:
                result = predict_from_dict(patient_data)
                results.append({
                    'patient_id': i + 1,
                    **result
                })
            except Exception as e:
                results.append({
                    'patient_id': i + 1,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def api_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'model_name': predictor.model_name,
        'feature_count': len(predictor.feature_columns),
        'features': predictor.feature_columns
    })

if __name__ == '__main__':
    print("="*60)
    print("Heart Disease Predictor - Web Interface")
    print("="*60)
    
    # Initialize predictor
    if init_predictor():
        print("\nStarting Flask server...")
        print("Access the UI at: http://127.0.0.1:5000")
        print("API endpoint: http://127.0.0.1:5000/api/predict")
        print("\nPress Ctrl+C to stop the server")
        print("="*60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server: Model not found")
        print("Please train a model first:")
        print("  python main.py train")
        sys.exit(1)

