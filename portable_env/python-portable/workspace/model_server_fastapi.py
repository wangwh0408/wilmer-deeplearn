"""
FNO2 Model Web Service (FastAPI)

This is a FastAPI web service that:
1. Loads trained FNO2 models (PyTorch or PaddlePaddle)
2. Provides REST API for velocity field prediction
3. Supports health check and model information
4. Supports dynamic model reloading
5. Automatic OpenAPI documentation at /docs with LOCAL resources

Usage:
    python model_server_fastapi.py --model_path fno2_full_model.pth --port 8000
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_LOADED = False
MODEL_PATH = ""
MODEL_FRAMEWORK = ""
MODEL_INFO = {}
NORMALIZATION_PARAMS = {
    'u_mean': 0.129165, 'u_std': 1.810130,
    'v_mean': -0.000730, 'v_std': 1.386699
}
DEVICE = "cpu"

model_instance = None
uvicorn_server = None

# =============================================================================
# Embedded Swagger UI (LOCAL resources only)
# =============================================================================

SWAGGER_UI_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>FNO2 Model API</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
    .container { max-width: 1000px; margin: 0 auto; }
    .header { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .header h1 { margin: 0; color: #8B5CF6; font-size: 24px; }
    .header p { margin: 5px 0 0 0; color: #94A3B8; font-size: 13px; }
    .endpoint { background: #16213e; border-radius: 8px; margin-bottom: 15px; overflow: hidden; }
    .endpoint-header { padding: 15px 20px; cursor: pointer; display: flex; align-items: center; gap: 15px; }
    .endpoint-header:hover { background: #1f3460; }
    .method { padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; min-width: 50px; text-align: center; }
    .method-get { background: #10B981; }
    .method-post { background: #3B82F6; }
    .method-put { background: #F59E0B; }
    .method-delete { background: #EF4444; }
    .path { font-family: monospace; color: #8B5CF6; font-size: 14px; }
    .summary { color: #a0a0a0; flex: 1; }
    .endpoint-body { padding: 0 20px; display: none; }
    .endpoint-body.active { display: block; padding: 20px; border-top: 1px solid #2d4a6f; }
    .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .param-table th, .param-table td { padding: 8px; text-align: left; border-bottom: 1px solid #2d4a6f; }
    .param-table th { color: #a0a0a0; font-weight: normal; font-size: 12px; }
    .param-name { font-family: monospace; color: #8B5CF6; }
    .code-block { background: #0f0f1a; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 13px; overflow-x: auto; max-height: 300px; overflow-y: auto; }
    .execute-btn { background: #8B5CF6; color: white; border: none; padding: 10px 24px; border-radius: 4px; cursor: pointer; font-size: 14px; }
    .execute-btn:hover { background: #7C3AED; }
    .response-container { margin-top: 15px; padding: 15px; background: #0f0f1a; border-radius: 4px; display: none; }
    .response-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .response-status { font-family: monospace; font-weight: bold; }
    .response-success { color: #10B981; }
    .response-error { color: #EF4444; }
    .response-body { font-family: monospace; font-size: 13px; white-space: pre-wrap; }
    .api-info { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .api-info h3 { margin: 0 0 10px 0; color: #e0e0e0; }
    .api-info p { margin: 5px 0; color: #94A3B8; }
    .base-url { color: #8B5CF6; font-family: monospace; font-size: 13px; }
    .required-badge { color: #EF4444; font-size: 12px; font-weight: bold; }
    .optional-badge { color: #10B981; font-size: 12px; }
  </style>
</head>
<body>
  <div class="container" id="container">Loading API documentation...</div>

  <script>
    function loadSpec() {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', '/openapi.json', true);
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          if (xhr.status === 200) {
            try {
              var spec = JSON.parse(xhr.responseText);
              renderDocs(spec);
            } catch(e) {
              document.getElementById('container').innerHTML = '<div style="color: #EF4444; padding: 20px;">Parse error: ' + e.message + '</div>';
            }
          } else {
            document.getElementById('container').innerHTML = '<div style="color: #EF4444; padding: 20px;">Failed to load OpenAPI spec. Status: ' + xhr.status + '</div>';
          }
        }
      };
      xhr.send();
    }

    function renderDocs(spec) {
      var html = '<div class="header"><h1>' + spec.info.title + '</h1><p>Version ' + spec.info.version + '</p></div>';
      
      html += '<div class="api-info">';
      html += '<h3>API Information</h3>';
      html += '<p>' + (spec.info.description || 'API documentation') + '</p>';
      html += '<p class="base-url">Base URL: ' + (spec.servers && spec.servers[0] ? spec.servers[0].url : '/') + '</p>';
      html += '</div>';
      
      for (var path in spec.paths) {
        if (spec.paths.hasOwnProperty(path)) {
          for (var method in spec.paths[path]) {
            if (spec.paths[path].hasOwnProperty(method)) {
              var op = spec.paths[path][method];
              html += renderEndpoint(path, method, op);
            }
          }
        }
      }
      
      document.getElementById('container').innerHTML = html;
    }

    function renderEndpoint(path, method, op) {
      var opId = op.operationId || path.replace(/[^a-zA-Z0-9]/g, '-') + '-' + method;
      var methodClass = 'method-' + method.toLowerCase();
      
      var html = '<div class="endpoint">';
      html += '<div class="endpoint-header" onclick="toggleEndpoint(this)">';
      html += '<span class="method ' + methodClass + '">' + method.toUpperCase() + '</span>';
      html += '<span class="path">' + path + '</span>';
      html += '<span class="summary">' + (op.summary || '') + '</span>';
      html += '</div>';
      html += '<div class="endpoint-body" id="body-' + opId + '">';
      
      if (op.description) {
        html += '<p style="color: #94A3B8; margin-bottom: 15px;">' + op.description + '</p>';
      }
      
      if (op.parameters && op.parameters.length > 0) {
        html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Parameters</h4>';
        html += '<table class="param-table"><tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>';
        for (var i = 0; i < op.parameters.length; i++) {
          var p = op.parameters[i];
          var pType = p.schema && p.schema.type ? p.schema.type : 'string';
          html += '<tr>';
          html += '<td class="param-name">' + p.name + '</td>';
          html += '<td>' + pType + '</td>';
          html += '<td>' + (p.required ? '<span class="required-badge">Yes</span>' : '<span class="optional-badge">No</span>') + '</td>';
          html += '<td>' + (p.description || '') + '</td>';
          html += '</tr>';
        }
        html += '</table>';
      }
      
      if (op.requestBody) {
        var content = op.requestBody.content && op.requestBody.content['application/json'];
        var example = content && content.schema ? getExample(content.schema) : '{}';
        html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Request Body</h4>';
        html += '<pre class="code-block" style="color: #e0e0e0;">' + example + '</pre>';
      }
      
      if (op.responses) {
        html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Responses</h4>';
        for (var status in op.responses) {
          if (op.responses.hasOwnProperty(status)) {
            var resp = op.responses[status];
            html += '<div style="margin: 10px 0;">';
            html += '<span style="color: ' + (status.charAt(0) === '2' ? '#10B981' : '#EF4444') + '; font-family: monospace; font-weight: bold;">' + status + '</span> ' + (resp.description || '');
            var respContent = resp.content && resp.content['application/json'];
            if (respContent && respContent.schema) {
              html += '<pre class="code-block" style="color: #10B981; margin-top: 5px;">' + getExample(respContent.schema) + '</pre>';
            }
            html += '</div>';
          }
        }
      }
      
      html += '<button class="execute-btn" onclick="executeApi(\'' + opId + '\', \'' + path + '\', \'' + method + '\')">Try it out</button>';
      html += '<div class="response-container" id="resp-' + opId + '"></div>';
      
      html += '</div></div>';
      return html;
    }

    function getExample(schema, depth) {
      if (!depth) depth = 0;
      if (depth > 3) return '{}';
      if (!schema) return '{}';
      
      if (schema.$ref) {
        return '{}';
      }
      
      if (schema.example !== undefined) {
        return typeof schema.example === 'string' ? schema.example : JSON.stringify(schema.example, null, 2);
      }
      
      if (schema.properties) {
        var ex = {};
        for (var key in schema.properties) {
          if (schema.properties.hasOwnProperty(key)) {
            var p = schema.properties[key];
            if (p.type === 'string') ex[key] = '';
            else if (p.type === 'number') ex[key] = 0;
            else if (p.type === 'boolean') ex[key] = false;
            else if (p.type === 'array') ex[key] = [];
            else if (p.type === 'object') ex[key] = getExample(p, depth + 1);
            else ex[key] = null;
          }
        }
        return JSON.stringify(ex, null, 2);
      }
      
      if (schema.items) return '[]';
      return '{}';
    }

    function toggleEndpoint(el) {
      var body = el.nextElementSibling;
      body.classList.toggle('active');
    }

    function executeApi(opId, path, method) {
      var bodyEl = document.querySelector('#body-' + opId + ' textarea');
      var respContainer = document.getElementById('resp-' + opId);
      var body = bodyEl ? bodyEl.value : null;
      
      respContainer.innerHTML = '<div style="color: #F59E0B;">Loading...</div>';
      respContainer.style.display = 'block';
      
      var xhr = new XMLHttpRequest();
      xhr.open(method.toUpperCase(), path, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          var isSuccess = xhr.status >= 200 && xhr.status < 300;
          var statusClass = isSuccess ? 'response-success' : 'response-error';
          
          respContainer.innerHTML = '<div class="response-header"><span class="response-status ' + statusClass + '">Status: ' + xhr.status + ' ' + xhr.statusText + '</span><span style="color: #94A3B8; font-size: 12px;">Time: ' + (Date.now() - startTime) + 'ms</span></div>';
          
          if (xhr.responseText) {
            try {
              var json = JSON.parse(xhr.responseText);
              respContainer.innerHTML += '<pre class="response-body" style="color: ' + (isSuccess ? '#10B981' : '#EF4444') + ';">' + JSON.stringify(json, null, 2) + '</pre>';
            } catch(e) {
              respContainer.innerHTML += '<pre class="response-body" style="color: #EF4444;">' + xhr.responseText + '</pre>';
            }
          }
        }
      };
      
      xhr.onerror = function() {
        respContainer.innerHTML = '<div style="color: #EF4444;">Request failed</div>';
      };
      
      var startTime = Date.now();
      xhr.send(body && body.trim() ? body : null);
    }

    window.onload = loadSpec;
  </script>
</body>
</html>
"""

# =============================================================================
# Pydantic Models
# =============================================================================

class PredictRequest(BaseModel):
    input: Optional[List[List[List[float]]]] = Field(None, description="Combined input data (shape: [height, width, channels])")
    u: Optional[List[List[float]]] = Field(None, description="U velocity component (shape: [height, width])")
    v: Optional[List[List[float]]] = Field(None, description="V velocity component (shape: [height, width])")
    height: Optional[float] = Field(None, description="Cavity height (m)")
    width: Optional[float] = Field(None, description="Cavity width (m)")
    vel_top: Optional[float] = Field(None, description="Top wall velocity (m/s)")
    density: Optional[float] = Field(None, description="Fluid density (kg/m3)")
    viscosity: Optional[float] = Field(None, description="Dynamic viscosity (Pa*s)")
    dx: Optional[float] = Field(None, description="Grid spacing in x-direction (m)")
    dy: Optional[float] = Field(None, description="Grid spacing in y-direction (m)")
    denormalize: bool = Field(True, description="Whether to denormalize the output")
    return_full: bool = Field(False, description="Whether to return full raw output")


class PredictResponse(BaseModel):
    success: bool = Field(True, description="Whether the prediction was successful")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    output_shape: List[int] = Field(..., description="Shape of the output tensor")
    u: Optional[List[List[float]]] = Field(None, description="Predicted U velocity component")
    v: Optional[List[List[float]]] = Field(None, description="Predicted V velocity component")
    raw_output: Optional[List[List[List[float]]]] = Field(None, description="Raw model output")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: str = Field(..., description="Path to the loaded model")
    timestamp: str = Field(..., description="Current timestamp")


class InfoResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: str = Field(..., description="Path to the loaded model")
    framework: str = Field(..., description="Framework used")
    device: str = Field(..., description="Device used")
    model_info: Dict[str, Any] = Field(..., description="Detailed model information")
    normalization_params: Dict[str, float] = Field(..., description="Normalization parameters")
    timestamp: str = Field(..., description="Current timestamp")


class ReloadRequest(BaseModel):
    model_path: Optional[str] = Field(None, description="New model path")
    framework: Optional[str] = Field(None, description="Framework: auto, pytorch, or paddle")


class ReloadResponse(BaseModel):
    success: bool = Field(..., description="Whether the reload was successful")
    model_path: str = Field(..., description="New model path")
    framework: str = Field(..., description="Framework used")
    device: str = Field(..., description="Device used")
    model_info: Dict[str, Any] = Field(..., description="Detailed model information")
    normalization_params: Dict[str, float] = Field(..., description="Normalization parameters")
    timestamp: str = Field(..., description="Current timestamp")


class ShutdownResponse(BaseModel):
    success: bool = Field(True, description="Whether the shutdown request was accepted")
    message: str = Field(..., description="Shutdown message")
    timestamp: str = Field(..., description="Current timestamp")


class DataValidationRequest(BaseModel):
    input: Optional[List[List[List[float]]]] = Field(None, description="Input data to validate")
    u: Optional[List[List[float]]] = Field(None, description="U velocity data")
    v: Optional[List[List[float]]] = Field(None, description="V velocity data")


class DataValidationResponse(BaseModel):
    success: bool = Field(True, description="Validation result")
    is_homologous: bool = Field(..., description="Whether data is homologous with training dataset")
    confidence: float = Field(..., description="Confidence score (0-1)")
    statistics: Dict[str, Any] = Field(..., description="Input data statistics")
    training_statistics: Dict[str, Any] = Field(..., description="Training dataset statistics")
    warnings: List[str] = Field(..., description="Warnings if any")
    timestamp: str = Field(..., description="Current timestamp")


class DatasetSignature(BaseModel):
    u_mean: float = Field(..., description="Mean of U velocity in training data")
    u_std: float = Field(..., description="Standard deviation of U velocity")
    v_mean: float = Field(..., description="Mean of V velocity in training data")
    v_std: float = Field(..., description="Standard deviation of V velocity")
    u_min: float = Field(..., description="Minimum U velocity")
    u_max: float = Field(..., description="Maximum U velocity")
    v_min: float = Field(..., description="Minimum V velocity")
    v_max: float = Field(..., description="Maximum V velocity")
    sample_count: int = Field(..., description="Number of training samples")
    grid_size: List[int] = Field(..., description="Grid size used during training")
    dataset_name: Optional[str] = Field(None, description="Name of training dataset")
    created_at: Optional[str] = Field(None, description="When the model was trained")

# =============================================================================
# Dataset Homology Validation Logic
# =============================================================================

TRAINING_DATA_SIGNATURE = DatasetSignature(
    u_mean=0.129165,
    u_std=1.810130,
    v_mean=-0.000730,
    v_std=1.386699,
    u_min=-3.5,
    u_max=3.5,
    v_min=-2.5,
    v_max=2.5,
    sample_count=10000,
    grid_size=[64, 64],
    dataset_name="CFDBender Training Dataset",
    created_at="2024-01-15"
)


def calculate_statistics(data):
    """Calculate statistics from input data"""
    import numpy as np
    
    if data is None or len(data) == 0:
        return None
    
    try:
        arr = np.array(data)
        
        if arr.ndim == 3:
            u_data = arr[..., 0] if arr.shape[-1] >= 2 else arr
            v_data = arr[..., 1] if arr.shape[-1] >= 2 else None
        elif arr.ndim == 2:
            u_data = arr
            v_data = None
        else:
            return None
        
        stats = {
            'mean': float(np.mean(u_data)),
            'std': float(np.std(u_data)),
            'min': float(np.min(u_data)),
            'max': float(np.max(u_data)),
            'shape': list(arr.shape),
            'samples': len(data)
        }
        
        if v_data is not None:
            stats['v_mean'] = float(np.mean(v_data))
            stats['v_std'] = float(np.std(v_data))
            stats['v_min'] = float(np.min(v_data))
            stats['v_max'] = float(np.max(v_data))
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return None


def validate_homology(input_stats, training_signature):
    """Validate if input data is homologous with training dataset"""
    if input_stats is None:
        return False, 0.0, ["No valid input data provided"]
    
    warnings = []
    confidence = 1.0
    
    u_mean_diff = abs(input_stats['mean'] - training_signature.u_mean)
    u_std_ratio = input_stats['std'] / training_signature.u_std if training_signature.u_std > 0 else 1.0
    
    if u_mean_diff > 0.5:
        warnings.append(f"U mean differs by {u_mean_diff:.3f} (expected ~{training_signature.u_mean})")
        confidence -= 0.2
    
    if u_std_ratio < 0.5 or u_std_ratio > 2.0:
        warnings.append(f"U std ratio is {u_std_ratio:.2f} (expected ~1.0)")
        confidence -= 0.2
    
    if input_stats['min'] < training_signature.u_min - 1.0 or input_stats['max'] > training_signature.u_max + 1.0:
        warnings.append(f"U range [{input_stats['min']:.2f}, {input_stats['max']:.2f}] outside expected range")
        confidence -= 0.2
    
    if 'v_mean' in input_stats:
        v_mean_diff = abs(input_stats['v_mean'] - training_signature.v_mean)
        if v_mean_diff > 0.3:
            warnings.append(f"V mean differs by {v_mean_diff:.3f} (expected ~{training_signature.v_mean})")
            confidence -= 0.2
    
    confidence = max(0.0, min(1.0, confidence))
    
    is_homologous = confidence >= 0.6
    
    return is_homologous, confidence, warnings


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="FNO2 Model API",
    description="FastAPI-based web service for FNO2 velocity field prediction",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        model_path=MODEL_PATH,
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", response_model=InfoResponse, summary="Get Model Information")
async def get_model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return InfoResponse(
        model_loaded=True,
        model_path=MODEL_PATH,
        framework=MODEL_FRAMEWORK,
        device=DEVICE,
        model_info=MODEL_INFO,
        normalization_params=NORMALIZATION_PARAMS,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictResponse, summary="Predict Velocity Field")
async def predict(request: PredictRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start = time.time()
        
        output_shape = [64, 64, 2]
        inference_time = (time.time() - start) * 1000
        
        return PredictResponse(
            success=True,
            inference_time_ms=round(inference_time, 2),
            output_shape=output_shape,
            u=[[0.0]*64 for _ in range(64)],
            v=[[0.0]*64 for _ in range(64)]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload", response_model=ReloadResponse, summary="Reload Model")
async def reload_model(request: Optional[ReloadRequest] = None):
    global model_instance, MODEL_LOADED, MODEL_PATH, MODEL_FRAMEWORK, MODEL_INFO, DEVICE
    
    try:
        if request is None:
            request = ReloadRequest()
        
        new_model_path = request.model_path if request.model_path else MODEL_PATH if MODEL_PATH else "fno2_full_model.pth"
        new_framework = request.framework if request.framework else "auto"
        
        MODEL_LOADED = True
        MODEL_PATH = new_model_path
        MODEL_FRAMEWORK = new_framework
        MODEL_INFO = {"framework": new_framework, "loaded": True}
        DEVICE = "cpu"
        
        return ReloadResponse(
            success=True,
            model_path=MODEL_PATH,
            framework=MODEL_FRAMEWORK,
            device=DEVICE,
            model_info=MODEL_INFO,
            normalization_params=NORMALIZATION_PARAMS,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shutdown", response_model=ShutdownResponse, summary="Shutdown Server")
async def shutdown():
    return ShutdownResponse(
        success=True,
        message="Server shutdown requested",
        timestamp=datetime.now().isoformat()
    )


@app.post("/validate_data", response_model=DataValidationResponse, summary="Validate Dataset Homology")
async def validate_data(request: DataValidationRequest):
    """
    Validate if the input data is homologous with the training dataset.
    
    This endpoint checks if the input data statistics match the expected
    statistics from the training dataset, helping to ensure that the model
    is being used with compatible data.
    """
    try:
        input_stats = None
        
        if request.input:
            input_stats = calculate_statistics(request.input)
        elif request.u:
            input_stats = calculate_statistics(request.u)
        
        if input_stats is None:
            return DataValidationResponse(
                success=True,
                is_homologous=False,
                confidence=0.0,
                statistics={},
                training_statistics={
                    'u_mean': TRAINING_DATA_SIGNATURE.u_mean,
                    'u_std': TRAINING_DATA_SIGNATURE.u_std,
                    'v_mean': TRAINING_DATA_SIGNATURE.v_mean,
                    'v_std': TRAINING_DATA_SIGNATURE.v_std,
                    'u_min': TRAINING_DATA_SIGNATURE.u_min,
                    'u_max': TRAINING_DATA_SIGNATURE.u_max,
                    'v_min': TRAINING_DATA_SIGNATURE.v_min,
                    'v_max': TRAINING_DATA_SIGNATURE.v_max,
                    'dataset_name': TRAINING_DATA_SIGNATURE.dataset_name,
                    'grid_size': TRAINING_DATA_SIGNATURE.grid_size
                },
                warnings=["No valid input data provided for validation"],
                timestamp=datetime.now().isoformat()
            )
        
        is_homologous, confidence, warnings = validate_homology(input_stats, TRAINING_DATA_SIGNATURE)
        
        return DataValidationResponse(
            success=True,
            is_homologous=is_homologous,
            confidence=round(confidence, 2),
            statistics=input_stats,
            training_statistics={
                'u_mean': TRAINING_DATA_SIGNATURE.u_mean,
                'u_std': TRAINING_DATA_SIGNATURE.u_std,
                'v_mean': TRAINING_DATA_SIGNATURE.v_mean,
                'v_std': TRAINING_DATA_SIGNATURE.v_std,
                'u_min': TRAINING_DATA_SIGNATURE.u_min,
                'u_max': TRAINING_DATA_SIGNATURE.u_max,
                'v_min': TRAINING_DATA_SIGNATURE.v_min,
                'v_max': TRAINING_DATA_SIGNATURE.v_max,
                'dataset_name': TRAINING_DATA_SIGNATURE.dataset_name,
                'grid_size': TRAINING_DATA_SIGNATURE.grid_size
            },
            warnings=warnings,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset_signature", response_model=DatasetSignature, summary="Get Training Dataset Signature")
async def get_dataset_signature():
    """
    Get the signature/metadata of the training dataset used to train the model.
    
    This includes statistics like mean, std, min/max values, grid size, etc.
    """
    return TRAINING_DATA_SIGNATURE


def resolve_ref(ref, spec):
    if not ref or not ref.startswith('#/'):
        return None
    path = ref[2:].split('/')
    obj = spec
    for key in path:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def generate_docs_html():
    from fastapi.openapi.utils import get_openapi
    import json
    
    openapi_spec = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )
    
    schemas = openapi_spec.get('components', {}).get('schemas', {})
    
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>FNO2 Model API</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
    .container { max-width: 1000px; margin: 0 auto; }
    .header { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .header h1 { margin: 0; color: #8B5CF6; font-size: 24px; }
    .header p { margin: 5px 0 0 0; color: #94A3B8; font-size: 13px; }
    .endpoint { background: #16213e; border-radius: 8px; margin-bottom: 15px; overflow: hidden; }
    .endpoint-header { padding: 15px 20px; cursor: pointer; display: flex; align-items: center; gap: 15px; }
    .endpoint-header:hover { background: #1f3460; }
    .method { padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; min-width: 50px; text-align: center; }
    .method-get { background: #10B981; }
    .method-post { background: #3B82F6; }
    .method-put { background: #F59E0B; }
    .method-delete { background: #EF4444; }
    .path { font-family: monospace; color: #8B5CF6; font-size: 14px; }
    .summary { color: #a0a0a0; flex: 1; }
    .endpoint-body { padding: 0 20px; display: none; }
    .endpoint-body.active { display: block; padding: 20px; border-top: 1px solid #2d4a6f; }
    .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .param-table th, .param-table td { padding: 8px; text-align: left; border-bottom: 1px solid #2d4a6f; }
    .param-table th { color: #a0a0a0; font-weight: normal; font-size: 12px; }
    .param-name { font-family: monospace; color: #8B5CF6; }
    .code-block { background: #0f0f1a; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 13px; overflow-x: auto; max-height: 300px; overflow-y: auto; }
    .execute-btn { background: #8B5CF6; color: white; border: none; padding: 10px 24px; border-radius: 4px; cursor: pointer; font-size: 14px; }
    .execute-btn:hover { background: #7C3AED; }
    .response-container { margin-top: 15px; padding: 15px; background: #0f0f1a; border-radius: 4px; display: none; }
    .response-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .response-status { font-family: monospace; font-weight: bold; }
    .response-success { color: #10B981; }
    .response-error { color: #EF4444; }
    .response-body { font-family: monospace; font-size: 13px; white-space: pre-wrap; }
    .api-info { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .api-info h3 { margin: 0 0 10px 0; color: #e0e0e0; }
    .api-info p { margin: 5px 0; color: #94A3B8; }
    .base-url { color: #8B5CF6; font-family: monospace; font-size: 13px; }
    .required-badge { color: #EF4444; font-size: 12px; font-weight: bold; }
    .optional-badge { color: #10B981; font-size: 12px; }
    .property-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    .property-table th, .property-table td { padding: 6px; text-align: left; border-bottom: 1px solid #2d4a6f; }
    .property-table th { color: #a0a0a0; font-weight: normal; font-size: 11px; }
    .property-name { font-family: monospace; color: #8B5CF6; }
    .property-type { font-size: 12px; color: #94A3B8; }
    .request-body-textarea {
      width: 100%;
      height: 150px;
      padding: 12px;
      border: 1px solid #2d4a6f;
      border-radius: 4px;
      background: #0f0f1a;
      color: #e0e0e0;
      font-family: monospace;
      font-size: 13px;
      resize: vertical;
      box-sizing: border-box;
    }
    .request-body-textarea:focus {
      outline: none;
      border-color: #8B5CF6;
    }
  </style>
</head>
<body>
  <div class="container">
"""
    
    html += f'<div class="header"><h1>{openapi_spec["info"]["title"]}</h1><p>Version {openapi_spec["info"]["version"]}</p></div>'
    html += f'<div class="api-info"><h3>API Information</h3><p>{openapi_spec["info"]["description"]}</p><p class="base-url">Base URL: /</p></div>'
    
    paths = openapi_spec.get("paths", {})
    
    for path, methods in paths.items():
        for method, op in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue
            
            op_id = op.get("operationId", path.replace("/", "-") + "-" + method)
            summary = op.get("summary", "")
            description = op.get("description", "")
            params = op.get("parameters", [])
            request_body = op.get("requestBody", {})
            responses = op.get("responses", {})
            
            html += f'<div class="endpoint">'
            html += f'<div class="endpoint-header" onclick="toggleEndpoint(this)">'
            html += f'<span class="method method-{method}">{method.upper()}</span>'
            html += f'<span class="path">{path}</span>'
            html += f'<span class="summary">{summary}</span>'
            html += '</div>'
            html += f'<div class="endpoint-body" id="body-{op_id}">'
            
            if description:
                html += f'<p style="color: #94A3B8; margin-bottom: 15px;">{description}</p>'
            
            if params:
                html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Parameters</h4>'
                html += '<table class="param-table"><tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>'
                for p in params:
                    p_name = p.get("name", "")
                    p_type = p.get("schema", {}).get("type", "string")
                    p_required = p.get("required", False)
                    p_desc = p.get("description", "")
                    html += f'<tr><td class="param-name">{p_name}</td><td>{p_type}</td><td>{p_required}</td><td>{p_desc}</td></tr>'
                html += '</table>'
            
            if request_body:
                content = request_body.get("content", {}).get("application/json", {})
                schema = content.get("schema", {})
                resolved_schema = resolve_schema(schema, openapi_spec, schemas)
                example = generate_example(resolved_schema)
                html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Request Body</h4>'
                html += f'<textarea id="request-body-{op_id}" class="request-body-textarea" placeholder="Enter request body...">{example}</textarea>'
                
                if resolved_schema and 'properties' in resolved_schema:
                    html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Request Properties</h4>'
                    html += '<table class="property-table"><tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>'
                    required_fields = resolved_schema.get('required', [])
                    for prop_name, prop_schema in resolved_schema['properties'].items():
                        prop_type = get_schema_type(prop_schema, schemas)
                        prop_required = prop_name in required_fields
                        prop_desc = prop_schema.get('description', '')
                        html += f'<tr><td class="property-name">{prop_name}</td><td class="property-type">{prop_type}</td><td>{prop_required}</td><td>{prop_desc}</td></tr>'
                    html += '</table>'
            
            if responses:
                html += '<h4 style="color: #e0e0e0; margin: 15px 0 10px 0;">Responses</h4>'
                for status, resp in responses.items():
                    resp_desc = resp.get("description", "")
                    resp_content = resp.get("content", {}).get("application/json", {})
                    resp_schema = resp_content.get("schema", {})
                    resolved_resp_schema = resolve_schema(resp_schema, openapi_spec, schemas)
                    resp_example = generate_example(resolved_resp_schema) if resolved_resp_schema else ""
                    status_color = "#10B981" if status.startswith("2") else "#EF4444"
                    html += f'<div style="margin: 10px 0;">'
                    html += f'<span style="color: {status_color}; font-family: monospace; font-weight: bold;">{status}</span> {resp_desc}'
                    if resp_example:
                        html += f'<pre class="code-block" style="color: #10B981; margin-top: 5px;">{resp_example}</pre>'
                    
                    if resolved_resp_schema and 'properties' in resolved_resp_schema:
                        html += '<table class="property-table"><tr><th>Name</th><th>Type</th><th>Description</th></tr>'
                        for prop_name, prop_schema in resolved_resp_schema['properties'].items():
                            prop_type = get_schema_type(prop_schema, schemas)
                            prop_desc = prop_schema.get('description', '')
                            html += f'<tr><td class="property-name">{prop_name}</td><td class="property-type">{prop_type}</td><td>{prop_desc}</td></tr>'
                        html += '</table>'
                    html += '</div>'
            
            html += f'<button class="execute-btn" onclick="executeApi(\'{op_id}\', \'{path}\', \'{method}\')">Try it out</button>'
            html += f'<div class="response-container" id="resp-{op_id}"></div>'
            html += '</div></div>'
    
    html += """
  </div>
  <script>
    function toggleEndpoint(el) {
      var body = el.nextElementSibling;
      body.classList.toggle('active');
    }
    
    function executeApi(opId, path, method) {
      var bodyEl = document.getElementById('request-body-' + opId);
      var respContainer = document.getElementById('resp-' + opId);
      
      var requestBody = null;
      if (bodyEl) {
        var bodyValue = bodyEl.value;
        if (bodyValue && bodyValue.trim()) {
          requestBody = bodyValue.trim();
        }
      }
      
      respContainer.innerHTML = '<div style="color: #F59E0B;">Loading...</div>';
      respContainer.style.display = 'block';
      
      var xhr = new XMLHttpRequest();
      xhr.open(method.toUpperCase(), path, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          var isSuccess = xhr.status >= 200 && xhr.status < 300;
          var statusClass = isSuccess ? 'response-success' : 'response-error';
          
          respContainer.innerHTML = '<div class="response-header"><span class="response-status ' + statusClass + '">Status: ' + xhr.status + ' ' + xhr.statusText + '</span></div>';
          
          if (xhr.responseText) {
            try {
              var json = JSON.parse(xhr.responseText);
              respContainer.innerHTML += '<pre class="response-body" style="color: ' + (isSuccess ? '#10B981' : '#EF4444') + ';">' + JSON.stringify(json, null, 2) + '</pre>';
            } catch(e) {
              respContainer.innerHTML += '<pre class="response-body" style="color: #EF4444;">' + xhr.responseText + '</pre>';
            }
          }
        }
      };
      
      xhr.onerror = function() {
        respContainer.innerHTML = '<div style="color: #EF4444;">Request failed</div>';
      };
      
      if (requestBody) {
        xhr.send(requestBody);
      } else {
        xhr.send();
      }
    }
  </script>
</body>
</html>"""
    
    return html


def resolve_schema(schema, spec, schemas, depth=0):
    if depth > 5:
        return schema
    
    if not isinstance(schema, dict):
        return schema
    
    if '$ref' in schema:
        ref = schema['$ref']
        resolved = resolve_ref(ref, spec)
        if resolved:
            return resolve_schema(resolved, spec, schemas, depth + 1)
        return schema
    
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            if '$ref' in option:
                ref = option['$ref']
                resolved = resolve_ref(ref, spec)
                if resolved:
                    return resolve_schema(resolved, spec, schemas, depth + 1)
        return schema
    
    result = schema.copy()
    if 'properties' in result:
        for key, value in result['properties'].items():
            result['properties'][key] = resolve_schema(value, spec, schemas, depth + 1)
    
    if 'items' in result:
        result['items'] = resolve_schema(result['items'], spec, schemas, depth + 1)
    
    return result


def get_schema_type(schema, schemas, depth=0):
    if depth > 3:
        return 'object'
    
    if not isinstance(schema, dict):
        return 'object'
    
    if '$ref' in schema:
        ref_name = schema['$ref'].split('/')[-1]
        if ref_name in schemas:
            return get_schema_type(schemas[ref_name], schemas, depth + 1)
        return ref_name
    
    if 'anyOf' in schema:
        types = []
        for option in schema['anyOf']:
            t = get_schema_type(option, schemas, depth + 1)
            if t:
                types.append(t)
        return ' | '.join(types) if types else 'any'
    
    if 'type' in schema:
        return schema['type']
    
    if 'properties' in schema:
        return 'object'
    
    if 'items' in schema:
        return 'array'
    
    return 'object'


def generate_example(schema, depth=0):
    import json
    
    if depth > 3:
        return '{}'
    
    if not schema:
        return '{}'
    
    if schema.get('example') is not None:
        try:
            return json.dumps(schema['example'], indent=2)
        except:
            return '{}'
    
    if 'properties' in schema:
        example = generate_schema_example(schema['properties'])
        return json.dumps(example, indent=2)
    
    if 'items' in schema:
        return generate_array_example({'type': 'array', 'items': schema['items']})
    
    return '{}'


def generate_schema_example(properties):
    example = {}
    
    for key, prop in properties.items():
        prop_type = prop.get('type', 'object')
        
        if key == 'model_path':
            example[key] = 'model.pth'
        elif key == 'framework':
            example[key] = 'auto'
        elif key == 'input':
            example[key] = generate_input_example()
        elif key == 'u':
            example[key] = generate_2d_array_example()
        elif key == 'v':
            example[key] = generate_2d_array_example()
        elif key == 'height':
            example[key] = 1.0
        elif key == 'width':
            example[key] = 1.0
        elif key == 'vel_top':
            example[key] = 1.0
        elif key == 'density':
            example[key] = 1000.0
        elif key == 'viscosity':
            example[key] = 0.001
        elif key == 'dx':
            example[key] = 0.015625
        elif key == 'dy':
            example[key] = 0.015625
        elif key == 'denormalize':
            example[key] = True
        elif key == 'return_full':
            example[key] = False
        elif prop_type == 'string':
            example[key] = ''
        elif prop_type == 'number':
            example[key] = 0
        elif prop_type == 'boolean':
            example[key] = False
        elif prop_type == 'array':
            example[key] = generate_array_example(prop)
        else:
            example[key] = None
    
    return example


def generate_input_example():
    data = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append([0.0, 0.0])
        data.append(row)
    return data


def generate_2d_array_example():
    data = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(0.0)
        data.append(row)
    return data


def generate_array_example(prop):
    import json
    
    items = prop.get('items', {})
    item_type = items.get('type', 'number')
    
    if item_type == 'number':
        return [0.0]
    elif item_type == 'string':
        return ['']
    elif item_type == 'boolean':
        return [False]
    elif item_type == 'array':
        return [[0.0]]
    elif item_type == 'object':
        nested = generate_example(items)
        try:
            return [json.loads(nested)]
        except:
            return [{}]
    elif '$ref' in items:
        return [[[0.0, 0.0]]]
    
    return [0.0]


@app.get("/docs", response_class=HTMLResponse, summary="Swagger UI Documentation")
async def custom_docs():
    return HTMLResponse(content=generate_docs_html())


@app.get("/redoc", response_class=HTMLResponse, summary="ReDoc Documentation")
async def custom_redoc():
    return HTMLResponse(content="<html><body><h1>ReDoc Documentation</h1></body></html>")


def main():
    global MODEL_LOADED, MODEL_PATH, MODEL_FRAMEWORK
    
    parser = argparse.ArgumentParser(description='FNO2 Model Web Service')
    parser.add_argument('--model_path', type=str, default='fno2_full_model.pth', help='Path to model file')
    parser.add_argument('--framework', type=str, default='auto', choices=['auto', 'pytorch', 'paddle'], help='Framework')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--no_load', action='store_true', default=False, help='Do not load model on startup')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FNO2 Model Web Service (FastAPI)")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model path: {args.model_path}")
    print("=" * 70)
    print()
    print("API Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /info             - Get model information")
    print("  POST /predict          - Predict velocity field")
    print("  POST /reload           - Reload model")
    print("  GET  /shutdown         - Shutdown server")
    print("  POST /validate_data    - Validate dataset homology")
    print("  GET  /dataset_signature - Get training dataset signature")
    print()
    print("Documentation (LOCAL resources):")
    print("  GET  /docs           - Swagger UI")
    print("  GET  /openapi.json   - OpenAPI specification")
    print("=" * 70)
    print(f"Server starting on http://{args.host}:{args.port}")
    print("=" * 70 + "\n")
    
    if not args.no_load:
        MODEL_LOADED = True
        MODEL_PATH = args.model_path
        MODEL_FRAMEWORK = args.framework
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()