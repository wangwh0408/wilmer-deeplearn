import os
import sys
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import traceback
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_loader import ModelLoader


app = Flask(__name__)
app.config.from_object(Config)

if Config.CORS_ENABLED:
    CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})

model_loader = ModelLoader()


def make_error_response(message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None) -> Response:
    response = {
        'success': False,
        'error': message,
        'timestamp': datetime.utcnow().isoformat()
    }
    if details:
        response['details'] = details
    return jsonify(response), status_code


def make_success_response(data: Optional[Dict[str, Any]] = None, message: str = 'Success') -> Response:
    response = {
        'success': True,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }
    if data:
        response['data'] = data
    return jsonify(response), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    model_loaded = model_loader.is_loaded()
    health_status = {
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'host': Config.HOST,
            'port': Config.PORT,
            'device': Config.DEVICE,
            'debug': Config.DEBUG,
        }
    }
    
    if model_loaded:
        health_status['model_info'] = model_loader.get_model_info()
    
    return jsonify(health_status), 200


@app.route('/api/info', methods=['GET'])
def get_info():
    model_info = model_loader.get_model_info() if model_loader.is_loaded() else None
    
    info = {
        'service': 'FNO Model Service',
        'version': '1.0.0',
        'description': 'Web service for FNO (Fourier Neural Operator) model inference',
        'model_loaded': model_loader.is_loaded(),
        'model_info': model_info,
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/info': 'Get service information',
            'POST /api/model/load': 'Load model',
            'POST /api/model/unload': 'Unload model',
            'GET /api/model/info': 'Get model information',
            'POST /api/predict': 'Run prediction',
            'POST /api/predict/batch': 'Run batch prediction',
            'GET /api/test/sample': 'Get sample input data',
            'POST /api/test/predict': 'Run prediction with sample data',
        }
    }
    
    return make_success_response(info)


@app.route('/api/model/load', methods=['POST'])
def load_model():
    try:
        data = request.get_json(silent=True) or {}
        model_path = data.get('model_path')
        
        success, message = model_loader.load_model(model_path)
        
        if success:
            return make_success_response({
                'model_info': model_loader.get_model_info()
            }, 'Model loaded successfully')
        else:
            return make_error_response(message, 500)
            
    except Exception as e:
        return make_error_response(f'Failed to load model: {str(e)}', 500, {
            'traceback': traceback.format_exc()
        })


@app.route('/api/model/unload', methods=['POST'])
def unload_model():
    try:
        model_loader.unload_model()
        return make_success_response(message='Model unloaded successfully')
    except Exception as e:
        return make_error_response(f'Failed to unload model: {str(e)}', 500)


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    if not model_loader.is_loaded():
        return make_error_response('Model not loaded', 404)
    
    return make_success_response(model_loader.get_model_info())


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not model_loader.is_loaded():
            return make_error_response('Model not loaded', 404)
        
        data = request.get_json(silent=True)
        if data is None:
            return make_error_response('Invalid JSON payload', 400)
        
        source = data.get('source')
        if source is None:
            return make_error_response('Missing required field: source', 400)
        
        source_array = np.array(source, dtype=np.float32)
        
        if source_array.size == 0:
            return make_error_response('Empty input data', 400)
        
        max_res = Config.MAX_RESOLUTION
        if source_array.ndim >= 2 and (source_array.shape[-2] > max_res or source_array.shape[-1] > max_res):
            return make_error_response(
                f'Input resolution too large. Max: {max_res}x{max_res}', 
                400
            )
        
        result = model_loader.predict(source_array, return_numpy=True)
        
        response_data = {
            'input_shape': result['input_shape'],
            'output_shape': result['output_shape'],
            'output': result['output'].tolist(),
        }
        
        if data.get('include_input', False):
            response_data['input'] = result['input'].tolist()
        
        return make_success_response(response_data, 'Prediction completed successfully')
        
    except Exception as e:
        return make_error_response(f'Prediction failed: {str(e)}', 500, {
            'traceback': traceback.format_exc()
        })


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if not model_loader.is_loaded():
            return make_error_response('Model not loaded', 404)
        
        data = request.get_json(silent=True)
        if data is None:
            return make_error_response('Invalid JSON payload', 400)
        
        sources = data.get('sources')
        if sources is None or not isinstance(sources, list):
            return make_error_response('Missing or invalid field: sources (should be a list)', 400)
        
        max_batch = Config.MAX_BATCH_SIZE
        if len(sources) > max_batch:
            return make_error_response(
                f'Batch size too large. Max: {max_batch}', 
                400
            )
        
        source_arrays = [np.array(s, dtype=np.float32) for s in sources]
        
        results = model_loader.predict_batch(source_arrays, return_numpy=True)
        
        response_results = []
        for result in results:
            item = {
                'input_shape': result['input_shape'],
                'output_shape': result['output_shape'],
                'output': result['output'].tolist(),
            }
            if data.get('include_input', False):
                item['input'] = result['input'].tolist()
            response_results.append(item)
        
        return make_success_response({
            'predictions': response_results,
            'count': len(response_results)
        }, 'Batch prediction completed successfully')
        
    except Exception as e:
        return make_error_response(f'Batch prediction failed: {str(e)}', 500, {
            'traceback': traceback.format_exc()
        })


@app.route('/api/test/sample', methods=['GET'])
def get_sample_input():
    try:
        resolution = request.args.get('resolution', 64, type=int)
        num_sources = request.args.get('num_sources', 3, type=int)
        
        resolution = min(resolution, Config.MAX_RESOLUTION)
        
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        f = np.zeros((resolution, resolution))
        np.random.seed(42)
        for _ in range(num_sources):
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(1.0, 5.0)
            f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        
        sample = {
            'resolution': resolution,
            'num_sources': num_sources,
            'source': f.tolist(),
            'shape': [resolution, resolution],
        }
        
        return make_success_response(sample, 'Sample input generated successfully')
        
    except Exception as e:
        return make_error_response(f'Failed to generate sample: {str(e)}', 500)


@app.route('/api/test/predict', methods=['POST'])
def test_predict():
    try:
        if not model_loader.is_loaded():
            success, message = model_loader.load_model()
            if not success:
                return make_error_response(f'Failed to load model: {message}', 500)
        
        data = request.get_json(silent=True) or {}
        resolution = data.get('resolution', 64)
        num_sources = data.get('num_sources', 3)
        
        resolution = min(resolution, Config.MAX_RESOLUTION)
        
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        f = np.zeros((resolution, resolution))
        np.random.seed(42)
        for _ in range(num_sources):
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(1.0, 5.0)
            f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        
        result = model_loader.predict(f, return_numpy=True)
        
        response_data = {
            'input_shape': result['input_shape'],
            'output_shape': result['output_shape'],
            'input': result['input'].tolist(),
            'output': result['output'].tolist(),
            'resolution': resolution,
            'model_info': model_loader.get_model_info(),
        }
        
        return make_success_response(response_data, 'Test prediction completed successfully')
        
    except Exception as e:
        return make_error_response(f'Test prediction failed: {str(e)}', 500, {
            'traceback': traceback.format_exc()
        })


@app.before_request
def before_request():
    if request.method != 'OPTIONS':
        app.logger.info(f'{request.method} {request.path} from {request.remote_addr}')


@app.errorhandler(404)
def not_found(error):
    return make_error_response('Endpoint not found', 404)


@app.errorhandler(500)
def internal_error(error):
    return make_error_response('Internal server error', 500, {
        'error': str(error)
    })


def initialize_service():
    if os.path.exists(Config.MODEL_PATH):
        logger = app.logger
        logger.info(f"Auto-loading model from: {Config.MODEL_PATH}")
        success, message = model_loader.load_model()
        if success:
            logger.info("Model loaded successfully during startup")
        else:
            logger.warning(f"Failed to auto-load model: {message}")
    else:
        app.logger.info(f"Model file not found at: {Config.MODEL_PATH}. Use POST /api/model/load to load a model.")


if __name__ == '__main__':
    initialize_service()
    app.logger.info(f"Starting FNO Model Service on {Config.HOST}:{Config.PORT}")
    app.logger.info(f"Debug mode: {Config.DEBUG}")
    app.logger.info(f"Device: {Config.DEVICE}")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )
