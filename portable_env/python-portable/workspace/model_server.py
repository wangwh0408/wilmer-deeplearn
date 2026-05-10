"""
FNO2 Model Web Service (Flask)

This is a Flask web service that:
1. Loads trained FNO2 models (PyTorch or PaddlePaddle)
2. Provides REST API for velocity field prediction
3. Supports health check and model information
4. Supports dynamic model reloading

Usage:
    python model_server.py --model_path fno2_full_model.pth --port 5000
    python model_server.py --model_path fno2_paddle_model.pth --framework paddle --port 5001

API Endpoints:
    GET  /health           - Health check
    GET  /info             - Get model information
    POST /predict          - Predict velocity field from input
    POST /reload           - Reload model (with optional new path)
    GET  /shutdown         - Gracefully shutdown the server
"""

import os
import sys
import time
import json
import argparse
import threading
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from werkzeug.serving import make_server

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_LOADED = False
MODEL_PATH = ""
MODEL_FRAMEWORK = ""
MODEL_INFO = {}
NORMALIZATION_PARAMS = {}
DEVICE = "cpu"

model_instance = None
server_instance = None


def get_default_normalization_params() -> Dict[str, float]:
    """Get default normalization parameters from training data"""
    return {
        'u_mean': 0.129165,
        'u_std': 1.810130,
        'v_mean': -0.000730,
        'v_std': 1.386699,
    }


def load_model_pytorch(model_path: str) -> tuple:
    """Load PyTorch FNO2 model"""
    import torch
    import torch.nn as nn
    
    try:
        sys.path.insert(0, os.path.dirname(model_path))
    except:
        pass
    
    from fno2_model import FNO2d
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {})
            modes = config.get('modes', 12)
            width = config.get('width', 32)
            in_channels = config.get('in_channels', 4)
            out_channels = config.get('out_channels', 2)
            
            model = FNO2d(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            norm_params = checkpoint.get('normalization_params', None)
            if norm_params:
                global NORMALIZATION_PARAMS
                NORMALIZATION_PARAMS = norm_params
                
            model_info = {
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'framework': 'pytorch',
                'config': config
            }
        else:
            modes = 12
            width = 32
            in_channels = 4
            out_channels = 2
            
            model = FNO2d(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            )
            model.load_state_dict(checkpoint)
            
            model_info = {
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'framework': 'pytorch'
            }
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    model.to(device)
    model.eval()
    
    return model, model_info, device


def load_model_paddle(model_path: str) -> tuple:
    """Load PaddlePaddle FNO2 model"""
    import paddle
    
    try:
        sys.path.insert(0, os.path.dirname(model_path))
    except:
        pass
    
    from fno2_model_paddle import FNO2dPaddle
    
    device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
    paddle.device.set_device(device)
    logger.info(f"Using device: {device}")
    
    checkpoint = paddle.load(model_path)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {})
            modes = config.get('modes', 12)
            width = config.get('width', 32)
            in_channels = config.get('in_channels', 4)
            out_channels = config.get('out_channels', 2)
            
            model = Fno2dPaddle(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            )
            model.set_state_dict(checkpoint['model_state_dict'])
            
            norm_params = checkpoint.get('normalization_params', None)
            if norm_params:
                global NORMALIZATION_PARAMS
                NORMALIZATION_PARAMS = norm_params
                
            model_info = {
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'framework': 'paddlepaddle',
                'config': config
            }
        else:
            modes = 12
            width = 32
            in_channels = 4
            out_channels = 2
            
            model = FNO2dPaddle(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            )
            model.set_state_dict(checkpoint)
            
            model_info = {
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'framework': 'paddlepaddle'
            }
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    model.eval()
    
    return model, model_info, device


def load_model(model_path: str, framework: str = "auto") -> tuple:
    """Load model with automatic framework detection"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    if framework == "auto":
        if 'paddle' in model_path.lower():
            framework = "paddle"
        else:
            framework = "pytorch"
    
    logger.info(f"Detected framework: {framework}")
    
    if framework == "pytorch":
        return load_model_pytorch(model_path)
    elif framework == "paddle":
        return load_model_paddle(model_path)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def predict_pytorch(model, input_data: np.ndarray, device) -> np.ndarray:
    """Run prediction with PyTorch model"""
    import torch
    
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.cpu().numpy()


def predict_paddle(model, input_data: np.ndarray, device) -> np.ndarray:
    """Run prediction with PaddlePaddle model"""
    import paddle
    
    input_tensor = paddle.to_tensor(input_data, dtype='float32')
    
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    with paddle.no_grad():
        output = model(input_tensor)
    
    return output.numpy()


def normalize_input(u_data: np.ndarray, v_data: np.ndarray) -> np.ndarray:
    """Normalize input data"""
    u_normalized = (u_data - NORMALIZATION_PARAMS['u_mean']) / NORMALIZATION_PARAMS['u_std']
    v_normalized = (v_data - NORMALIZATION_PARAMS['v_mean']) / NORMALIZATION_PARAMS['v_std']
    
    return np.stack([u_normalized, v_normalized], axis=-1)


def denormalize_output(output: np.ndarray) -> Dict[str, np.ndarray]:
    """Denormalize model output"""
    u_output = output[..., 0] * NORMALIZATION_PARAMS['u_std'] + NORMALIZATION_PARAMS['u_mean']
    v_output = output[..., 1] * NORMALIZATION_PARAMS['v_std'] + NORMALIZATION_PARAMS['v_mean']
    
    return {
        'u': u_output,
        'v': v_output
    }


def run_prediction(input_data: np.ndarray) -> Dict[str, Any]:
    """Run prediction and return results"""
    global model_instance, MODEL_FRAMEWORK, DEVICE, MODEL_INFO
    
    if not MODEL_LOADED or model_instance is None:
        raise RuntimeError("Model not loaded")
    
    try:
        start_time = time.time()
        
        # Get expected input channels from model configuration
        expected_in_channels = MODEL_INFO.get('in_channels', 4)
        
        # The model adds 2 coordinate channels internally, so we expect
        # (expected_in_channels - 2) data channels from the input
        expected_data_channels = expected_in_channels - 2
        actual_data_channels = input_data.shape[-1]
        
        logger.debug(f"Input shape: {input_data.shape}, expected data channels: {expected_data_channels}")
        
        # If we have more channels than expected, we'll truncate to the expected number
        # This allows backward compatibility while supporting potential future models
        if actual_data_channels > expected_data_channels:
            logger.warning(f"Input has {actual_data_channels} channels, but model expects {expected_data_channels}. "
                        f"The extra {actual_data_channels - expected_data_channels} channels will be ignored.")
            input_data = input_data[..., :expected_data_channels]
        elif actual_data_channels < expected_data_channels:
            logger.warning(f"Input has {actual_data_channels} channels, but model expects {expected_data_channels}. "
                        f"Padding with zeros.")
            padding = np.zeros(input_data.shape[:-1] + (expected_data_channels - actual_data_channels,), 
                            dtype=np.float32)
            input_data = np.concatenate([input_data, padding], axis=-1)
        
        if MODEL_FRAMEWORK == "pytorch":
            output = predict_pytorch(model_instance, input_data, DEVICE)
        elif MODEL_FRAMEWORK in ["paddle", "paddlepaddle"]:
            output = predict_paddle(model_instance, input_data, DEVICE)
        else:
            raise ValueError(f"Unknown framework: {MODEL_FRAMEWORK}")
        
        inference_time = (time.time() - start_time) * 1000
        
        if output.ndim == 4:
            output = output[0]
        
        return {
            'output': output.tolist(),
            'inference_time_ms': round(inference_time, 2),
            'output_shape': list(output.shape)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'model_loaded': False
        }), 503
    
    return jsonify({
        'model_loaded': True,
        'model_path': MODEL_PATH,
        'framework': MODEL_FRAMEWORK,
        'device': DEVICE,
        'model_info': MODEL_INFO,
        'normalization_params': NORMALIZATION_PARAMS,
        'timestamp': datetime.now().isoformat()
    })


def create_condition_channels(
    height: int, width: int,
    height_param: Optional[float] = None,
    width_param: Optional[float] = None,
    vel_top: Optional[float] = None,
    density: Optional[float] = None,
    viscosity: Optional[float] = None,
    dx: Optional[float] = None,
    dy: Optional[float] = None
) -> np.ndarray:
    """
    Create condition parameter channels for the input tensor.
    """
    channels = []
    condition_params = [height_param, width_param, vel_top, density, viscosity, dx, dy]
    
    for param in condition_params:
        if param is not None:
            channel = np.full((height, width), param, dtype=np.float32)
            channels.append(channel)
    
    if channels:
        return np.stack(channels, axis=-1)
    else:
        return np.zeros((height, width, 0), dtype=np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict velocity field from input
    
    Request body (JSON):
    {
        "input": [[u_data], [v_data]]  // Shape: (64, 64, 2) or (N, 64, 64, 2)
        "u": [[...]]                     // U velocity data (shape: 64x64)
        "v": [[...]]                     // V velocity data (shape: 64x64)
        
        // CFD condition parameters (optional)
        "height": 1.0                    // Cavity height (m)
        "width": 1.0                     // Cavity width (m)
        "vel_top": 1.0                   // Top wall velocity (m/s)
        "density": 1000.0                // Fluid density (kg/m³)
        "viscosity": 0.001               // Fluid dynamic viscosity (Pa·s)
        "dx": 0.015625                   // Grid spacing in x-direction (m)
        "dy": 0.015625                   // Grid spacing in y-direction (m)
        
        "denormalize": true              // Whether to denormalize output (default: true)
        "return_full": false             // Return full output (not just denormalized)
    }
    
    Either:
    - Provide "input" with combined u+v data (shape: ..., 2)
    - Or provide "u" and "v" separately
    
    Response:
    {
        "success": true,
        "inference_time_ms": 12.34,
        "output_shape": [64, 64, 2],
        "u": [[...]],  // U velocity prediction
        "v": [[...]],  // V velocity prediction
        "raw_output": [[...]]  // Only if return_full=true
    }
    """
    try:
        if not MODEL_LOADED or model_instance is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'No request body'
            }), 400
        
        denormalize = request_data.get('denormalize', True)
        return_full = request_data.get('return_full', False)
        
        # Get condition parameters
        height_param = request_data.get('height')
        width_param = request_data.get('width')
        vel_top = request_data.get('vel_top')
        density = request_data.get('density')
        viscosity = request_data.get('viscosity')
        dx = request_data.get('dx')
        dy = request_data.get('dy')
        
        has_condition_params = any([
            height_param is not None,
            width_param is not None,
            vel_top is not None,
            density is not None,
            viscosity is not None,
            dx is not None,
            dy is not None
        ])
        
        input_data = None
        if 'input' in request_data:
            input_data = np.array(request_data['input'], dtype=np.float32)
            logger.info(f"Prediction request received with input shape: {input_data.shape}")
            if has_condition_params:
                logger.info("Condition parameters provided but will be ignored for combined input")
        elif 'u' in request_data and 'v' in request_data:
            u_data = np.array(request_data['u'], dtype=np.float32)
            v_data = np.array(request_data['v'], dtype=np.float32)
            
            if u_data.shape != v_data.shape:
                return jsonify({
                    'success': False,
                    'error': f'u and v must have same shape: {u_data.shape} vs {v_data.shape}'
                }), 400
            
            if u_data.ndim == 2:
                height, width = u_data.shape
                
                if denormalize:
                    input_data = normalize_input(u_data, v_data)
                else:
                    input_data = np.stack([u_data, v_data], axis=-1)
                
                # Add condition parameters as additional channels if provided
                if has_condition_params:
                    condition_channels = create_condition_channels(
                        height, width,
                        height_param, width_param,
                        vel_top, density,
                        viscosity, dx, dy
                    )
                    input_data = np.concatenate([input_data, condition_channels], axis=-1)
                    logger.info(f"Added {condition_channels.shape[-1]} condition parameter channels")
            elif u_data.ndim == 3:
                if denormalize:
                    input_data = np.stack([
                        (u_data - NORMALIZATION_PARAMS['u_mean']) / NORMALIZATION_PARAMS['u_std'],
                        (v_data - NORMALIZATION_PARAMS['v_mean']) / NORMALIZATION_PARAMS['v_std']
                    ], axis=-1)
                else:
                    input_data = np.stack([u_data, v_data], axis=-1)
            else:
                return jsonify({
                    'success': False,
                    'error': f'Invalid u/v shape: {u_data.shape}'
                }), 400
            
            logger.info(f"Prediction request received: u shape={u_data.shape}, v shape={v_data.shape}")
            if has_condition_params:
                logger.info(f"Condition params: height={height_param}, width={width_param}, "
                          f"vel_top={vel_top}, density={density}, "
                          f"viscosity={viscosity}, dx={dx}, dy={dy}")
        else:
            return jsonify({
                'success': False,
                'error': 'Must provide either "input" or both "u" and "v"'
            }), 400
        
        result = run_prediction(input_data)
        
        response = {
            'success': True,
            'inference_time_ms': result['inference_time_ms'],
            'output_shape': result['output_shape']
        }
        
        if denormalize or return_full:
            output = np.array(result['output'], dtype=np.float32)
            if denormalize:
                denormalized = denormalize_output(output)
                response['u'] = denormalized['u'].tolist()
                response['v'] = denormalized['v'].tolist()
            
            if return_full:
                response['raw_output'] = output.tolist()
        
        logger.info(f"Prediction completed in {result['inference_time_ms']}ms")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/reload', methods=['POST'])
def reload_model():
    """
    Reload model (optionally with new path)
    
    Request body (optional):
    {
        "model_path": "new_model.pth",
        "framework": "pytorch"  // optional, auto-detected
    }
    """
    global model_instance, MODEL_LOADED, MODEL_PATH, MODEL_FRAMEWORK, MODEL_INFO, DEVICE, NORMALIZATION_PARAMS
    
    try:
        request_data = request.get_json(silent=True) or {}
        
        new_model_path = request_data.get('model_path', MODEL_PATH)
        new_framework = request_data.get('framework', "auto")
        
        if not new_model_path:
            return jsonify({
                'success': False,
                'error': 'No model path provided'
            }), 400
        
        if not os.path.exists(new_model_path):
            return jsonify({
                'success': False,
                'error': f'Model file not found: {new_model_path}'
            }), 400
        
        logger.info(f"Reloading model: {new_model_path}")
        
        model, model_info, device = load_model(new_model_path, new_framework)
        
        model_instance = model
        MODEL_LOADED = True
        MODEL_PATH = new_model_path
        MODEL_FRAMEWORK = model_info.get('framework', new_framework)
        MODEL_INFO = model_info
        DEVICE = device
        
        if 'normalization_params' not in request_data:
            NORMALIZATION_PARAMS = get_default_normalization_params()
        else:
            NORMALIZATION_PARAMS = request_data['normalization_params']
        
        return jsonify({
            'success': True,
            'model_path': MODEL_PATH,
            'framework': MODEL_FRAMEWORK,
            'device': DEVICE,
            'model_info': MODEL_INFO,
            'normalization_params': NORMALIZATION_PARAMS,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model reload error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/shutdown', methods=['GET', 'POST'])
def shutdown():
    """Gracefully shutdown the server"""
    global server_instance, MODEL_LOADED
    
    logger.info("Shutting down server...")
    
    def perform_shutdown():
        time.sleep(0.5)
        if server_instance:
            server_instance.shutdown()
    
    threading.Thread(target=perform_shutdown, daemon=True).start()
    
    return jsonify({
        'success': True,
        'message': 'Server shutting down...',
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


def create_app():
    """Create Flask application"""
    return app


def main():
    global model_instance, MODEL_LOADED, MODEL_PATH, MODEL_FRAMEWORK, MODEL_INFO, DEVICE, NORMALIZATION_PARAMS, server_instance
    
    parser = argparse.ArgumentParser(
        description='FNO2 Model Web Service (Flask)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='fno2_full_model.pth',
        help='Path to model file (.pth for PyTorch)'
    )
    
    parser.add_argument(
        '--framework',
        type=str,
        default='auto',
        choices=['auto', 'pytorch', 'paddle'],
        help='Framework to use (auto = detect from filename)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to listen on'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no_load',
        action='store_true',
        default=False,
        help='Do not load model on startup (load later via /reload)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FNO2 Model Web Service (Flask)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Model path: {args.model_path}")
    print(f"Framework: {args.framework}")
    print("=" * 70 + "\n")
    
    NORMALIZATION_PARAMS = get_default_normalization_params()
    print("Default normalization parameters:")
    print(f"  u: mean={NORMALIZATION_PARAMS['u_mean']:.6f}, std={NORMALIZATION_PARAMS['u_std']:.6f}")
    print(f"  v: mean={NORMALIZATION_PARAMS['v_mean']:.6f}, std={NORMALIZATION_PARAMS['v_std']:.6f}")
    print()
    
    if not args.no_load:
        try:
            print("Loading model...")
            model, model_info, device = load_model(args.model_path, args.framework)
            
            model_instance = model
            MODEL_LOADED = True
            MODEL_PATH = args.model_path
            MODEL_FRAMEWORK = model_info.get('framework', args.framework)
            MODEL_INFO = model_info
            DEVICE = device
            
            print(f"Model loaded successfully!")
            print(f"  Framework: {MODEL_FRAMEWORK}")
            print(f"  Device: {DEVICE}")
            print(f"  Modes: {MODEL_INFO.get('modes', 'N/A')}")
            print(f"  Width: {MODEL_INFO.get('width', 'N/A')}")
            print(f"  Input channels: {MODEL_INFO.get('in_channels', 'N/A')}")
            print(f"  Output channels: {MODEL_INFO.get('out_channels', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            print("You can load it later via /reload endpoint")
            print()
    else:
        print("Model loading skipped (--no_load specified)")
        print("Use /reload endpoint to load model later")
        print()
    
    print("API Endpoints:")
    print("  GET  /health    - Health check")
    print("  GET  /info      - Get model information")
    print("  POST /predict   - Predict velocity field")
    print("  POST /reload    - Reload model")
    print("  GET  /shutdown  - Shutdown server")
    print()
    print("=" * 70)
    print("Server starting...")
    print("=" * 70 + "\n")
    
    if args.debug:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    else:
        server_instance = make_server(args.host, args.port, app)
        logger.info(f"Server running on http://{args.host}:{args.port}")
        server_instance.serve_forever()


if __name__ == '__main__':
    main()
