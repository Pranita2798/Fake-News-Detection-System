#!/usr/bin/env python3
"""
Flask API for Fake News Detection System
Provides RESTful API endpoints for fake news detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging
import traceback
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = []

def load_model(model_path='models/best_fake_news_model.pkl'):
    """Load the trained model."""
    global model, preprocessor, feature_names
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news article is fake or real."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model is properly loaded'
            }), 500
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field',
                'message': 'Please provide text field in JSON payload'
            }), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({
                'error': 'Empty text',
                'message': 'Text cannot be empty'
            }), 400
        
        # Process text and make prediction
        features = preprocessor.process_single_text(text)
        prediction = model.predict([features])[0]
        prediction_proba = model.predict_proba([features])[0]
        
        # Extract additional features for response
        linguistic_features = preprocessor.extract_linguistic_features(text)
        
        # Prepare response
        result = {
            'prediction': {
                'label': 'Real' if prediction == 1 else 'Fake',
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    'fake': float(prediction_proba[0]),
                    'real': float(prediction_proba[1])
                }
            },
            'analysis': {
                'word_count': int(linguistic_features['word_count']),
                'sentence_count': int(linguistic_features['sentence_count']),
                'avg_word_length': round(linguistic_features['avg_word_length'], 2),
                'sentiment_polarity': round(linguistic_features['sentiment_polarity'], 3),
                'sentiment_subjectivity': round(linguistic_features['sentiment_subjectivity'], 3),
                'exclamation_count': int(linguistic_features['exclamation_count']),
                'question_count': int(linguistic_features['question_count']),
                'capital_ratio': round(linguistic_features['capital_ratio'], 3),
                'punctuation_ratio': round(linguistic_features['punctuation_ratio'], 3)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple news articles."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model is properly loaded'
            }), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing texts field',
                'message': 'Please provide texts field with list of articles'
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Invalid texts format',
                'message': 'texts must be a list of strings'
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            try:
                features = preprocessor.process_single_text(text)
                prediction = model.predict([features])[0]
                prediction_proba = model.predict_proba([features])[0]
                
                result = {
                    'index': i,
                    'prediction': {
                        'label': 'Real' if prediction == 1 else 'Fake',
                        'confidence': float(max(prediction_proba)),
                        'probabilities': {
                            'fake': float(prediction_proba[0]),
                            'real': float(prediction_proba[1])
                        }
                    }
                }
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(texts),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': len(feature_names),
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An internal server error occurred'
    }), 500

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("Model not loaded. Some endpoints may not work.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)