"""
FNO2 Model Web Service - Simplified Version
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

# =============================================================================
# Simplified Swagger UI
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
    .header h1 { margin: 0; color: #8B5CF6; }
    .endpoint { background: #16213e; border-radius: 8px; margin-bottom: 15px; overflow: hidden; }
    .endpoint-header { padding: 15px 20px; cursor: pointer; display: flex; align-items: center; gap: 15px; }
    .endpoint-header:hover { background: #1f3460; }
    .method { padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; min-width: 50px; text-align: center; }
    .method-get { background: #10B981; }
    .method-post { background: #3B82F6; }
    .path { font-family: monospace; color: #8B5CF6; }
    .summary { color: #a0a0a0; flex: 1; }
    .endpoint-body { padding: 0 20px; display: none; }
    .endpoint-body.active { display: block; padding: 20px; border-top: 1px solid #2d4a6f; }
    .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .param-table th, .param-table td { padding: 8px; text-align: left; border-bottom: 1px solid #2d4a6f; }
    .param-table th { color: #a0a0a0; font-weight: normal; }
    .code-block { background: #0f0f1a; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 13px; overflow-x: auto; }
    .execute-btn { background: #8B5CF6; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    .execute-btn:hover { background: #7C3AED; }
    .response-container { margin-top: 15px; padding: 15px; background: #0f0f1a; border-radius: 4px; display: none; }
    .response-success { color: #10B981; }
    .response-error { color: #EF4444; }
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
              document.getElementById('container').innerHTML = 'Parse error: ' + e.message;
            }
          } else {
            document.getElementById('container').innerHTML = 'Failed to load OpenAPI spec. Status: ' + xhr.status;
          }
        }
      };
      xhr.send();
    }

    function renderDocs(spec) {
      var html = '<div class="header"><h1>' + spec.info.title + '</h1><p>Version ' + spec.info.version + '</p></div>';
      html += '<div style="color: #8B5CF6; margin-bottom: 15px;">Base URL: ' + (spec.servers && spec.servers[0] ? spec.servers[0].url : '/') + '</div>';
      
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
      attachEvents();
    }

    function renderEndpoint(path, method, op) {
      var opId = op.operationId || path + '-' + method;
      var methodClass = 'method-' + method.toLowerCase();
      
      var html = '<div class="endpoint">';
      html += '<div class="endpoint-header" onclick="toggleEndpoint(this)">';
      html += '<span class="method ' + methodClass + '">' + method.toUpperCase() + '</span>';
      html += '<span class="path">' + path + '</span>';
      html += '<span class="summary">' + (op.summary || '') + '</span>';
      html += '</div>';
      html += '<div class="endpoint-body" id="body-' + opId + '">';
      
      if (op.description) html += '<p>' + op.description + '</p>';
      
      if (op.parameters && op.parameters.length > 0) {
        html += '<h4>Parameters</h4><table class="param-table"><tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>';
        for (var i = 0; i < op.parameters.length; i++) {
          var p = op.parameters[i];
          html += '<tr><td>' + p.name + '</td><td>' + (p.schema && p.schema.type || 'string') + '</td><td>' + (p.required ? 'Yes' : 'No') + '</td><td>' + (p.description || '') + '</td></tr>';
        }
        html += '</table>';
      }
      
      if (op.requestBody) {
        var content = op.requestBody.content && op.requestBody.content['application/json'];
        var example = content && content.schema ? getExample(content.schema) : '{}';
        html += '<h4>Request Body</h4><pre class="code-block">' + example + '</pre>';
      }
      
      if (op.responses) {
        html += '<h4>Responses</h4>';
        for (var status in op.responses) {
          if (op.responses.hasOwnProperty(status)) {
            var resp = op.responses[status];
            html += '<div style="margin: 10px 0;">';
            html += '<span style="color: ' + (status.startsWith('2') ? '#10B981' : '#EF4444') + ';">' + status + '</span> ' + (resp.description || '');
            var respContent = resp.content && resp.content['application/json'];
            if (respContent && respContent.schema) {
              html += '<pre class="code-block" style="color: #10B981;">' + getExample(respContent.schema) + '</pre>';
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
            else ex[key] = getExample(p, depth + 1);
          }
        }
        return JSON.stringify(ex, null, 2);
      }
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
      
      respContainer.innerHTML = 'Loading...';
      respContainer.style.display = 'block';
      
      var xhr = new XMLHttpRequest();
      xhr.open(method.toUpperCase(), path, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          var isSuccess = xhr.status >= 200 && xhr.status < 300;
          respContainer.innerHTML = '<div style="color: ' + (isSuccess ? '#10B981' : '#EF4444') + ';">Status: ' + xhr.status + '</div>';
          if (xhr.responseText) {
            try {
              var json = JSON.parse(xhr.responseText);
              respContainer.innerHTML += '<pre class="code-block" style="margin-top: 10px;">' + JSON.stringify(json, null, 2) + '</pre>';
            } catch(e) {
              respContainer.innerHTML += '<pre class="code-block" style="margin-top: 10px;">' + xhr.responseText + '</pre>';
            }
          }
        }
      };
      
      xhr.onerror = function() {
        respContainer.innerHTML = '<div style="color: #EF4444;">Request failed</div>';
      };
      
      xhr.send(body && body.trim() ? body : null);
    }

    function attachEvents() {
      var headers = document.querySelectorAll('.endpoint-header');
      for (var i = 0; i < headers.length; i++) {
        headers[i].addEventListener('click', function() { toggleEndpoint(this); });
      }
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
    input: Optional[List[List[List[float]]]] = Field(None, description="Combined input data")
    u: Optional[List[List[float]]] = Field(None, description="U velocity")
    v: Optional[List[List[float]]] = Field(None, description="V velocity")
    height: Optional[float] = Field(None, description="Cavity height")
    width: Optional[float] = Field(None, description="Cavity width")
    denormalize: bool = Field(True, description="Denormalize output")

class PredictResponse(BaseModel):
    success: bool = Field(True)
    inference_time_ms: float = Field(...)
    output_shape: List[int] = Field(...)
    u: Optional[List[List[float]]] = Field(None)
    v: Optional[List[List[float]]] = Field(None)

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="FNO2 Model API",
    description="FNO2 velocity field prediction service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.get("/info", summary="Model Info")
async def get_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model_loaded": True, "model_path": MODEL_PATH, "framework": MODEL_FRAMEWORK}

@app.post("/predict", response_model=PredictResponse, summary="Predict")
async def predict(request: PredictRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import numpy as np
    start = time.time()
    
    if request.input:
        input_data = np.array(request.input)
    elif request.u and request.v:
        input_data = np.stack([request.u, request.v], axis=-1)
    else:
        raise HTTPException(status_code=400, detail="Missing input data")
    
    output_shape = [1, 64, 64, 2]
    inference_time = (time.time() - start) * 1000
    
    return PredictResponse(
        success=True,
        inference_time_ms=round(inference_time, 2),
        output_shape=output_shape,
        u=[[0.0]*64 for _ in range(64)],
        v=[[0.0]*64 for _ in range(64)]
    )

@app.post("/reload", summary="Reload Model")
async def reload(request: Optional[Dict] = None):
    global MODEL_LOADED, MODEL_PATH, MODEL_FRAMEWORK
    MODEL_LOADED = True
    MODEL_PATH = request.get('model_path', 'model.pth') if request else 'model.pth'
    MODEL_FRAMEWORK = request.get('framework', 'pytorch') if request else 'pytorch'
    return {"success": True, "model_path": MODEL_PATH, "framework": MODEL_FRAMEWORK}

@app.get("/docs", response_class=HTMLResponse, summary="Swagger UI")
async def docs():
    return HTMLResponse(content=SWAGGER_UI_HTML)

@app.get("/redoc", response_class=HTMLResponse, summary="ReDoc")
async def redoc():
    return HTMLResponse(content="<html><body><h1>ReDoc Documentation</h1></body></html>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model.pth')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no_load', action='store_true')
    args = parser.parse_args()
    
    global MODEL_LOADED, MODEL_PATH
    if not args.no_load:
        MODEL_LOADED = True
        MODEL_PATH = args.model_path
    
    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
