import os
import sys
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from schemas import (
    HealthResponse,
    ServiceInfoResponse,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    SampleInputResponse,
    LoadModelRequest,
    LoadModelResponse,
    ErrorResponse,
    SinglePredictionResult,
)
from model_manager import model_manager, lifespan


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

if settings.CORS_ENABLED:
    origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "timestamp": datetime.utcnow().isoformat(),
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "details": traceback.format_exc() if settings.DEBUG else None,
        },
    )


def check_model_loaded():
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not loaded. Use POST /api/model/load to load a model.",
        )
    return True


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the service",
    tags=["Health"],
)
async def health_check():
    model_loaded = model_manager.is_loaded()
    health_status = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow(),
        "config": {
            "host": settings.HOST,
            "port": settings.PORT,
            "device": settings.DEVICE,
            "debug": settings.DEBUG,
        },
    }
    
    if model_loaded:
        health_status["model_info"] = model_manager.get_model_info()
    
    return HealthResponse(**health_status)


@app.get(
    "/api/info",
    response_model=ServiceInfoResponse,
    summary="Service Information",
    description="Get detailed information about the service",
    tags=["Information"],
)
async def get_service_info():
    model_info = model_manager.get_model_info() if model_manager.is_loaded() else None
    
    info = {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "model_loaded": model_manager.is_loaded(),
        "model_info": model_info,
        "endpoints": {
            "GET /api/health": "Health check",
            "GET /api/info": "Service information",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation",
            "POST /api/model/load": "Load model",
            "POST /api/model/unload": "Unload model",
            "GET /api/model/info": "Get model information",
            "POST /api/predict": "Run prediction",
            "POST /api/predict/batch": "Run batch prediction",
            "GET /api/test/sample": "Get sample input data",
            "POST /api/test/predict": "Run prediction with sample data",
        },
    }
    
    return ServiceInfoResponse(
        success=True,
        message="Success",
        data=info,
    )


@app.post(
    "/api/model/load",
    response_model=LoadModelResponse,
    summary="Load Model",
    description="Load a model from file. If no path is provided, uses default path.",
    tags=["Model Management"],
)
async def load_model(request: Optional[LoadModelRequest] = None):
    model_path = request.model_path if request else None
    success, message, info = model_manager.load_model(model_path)
    
    if success:
        return LoadModelResponse(
            success=True,
            message=message,
            data={"model_info": info},
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


@app.post(
    "/api/model/unload",
    response_model=LoadModelResponse,
    summary="Unload Model",
    description="Unload the currently loaded model and free resources",
    tags=["Model Management"],
)
async def unload_model():
    model_manager.unload_model()
    return LoadModelResponse(
        success=True,
        message="Model unloaded successfully",
    )


@app.get(
    "/api/model/info",
    response_model=ServiceInfoResponse,
    summary="Get Model Information",
    description="Get detailed information about the loaded model",
    tags=["Model Management"],
)
async def get_model_info(_: bool = Depends(check_model_loaded)):
    return ServiceInfoResponse(
        success=True,
        message="Model information retrieved successfully",
        data=model_manager.get_model_info(),
    )


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Single Prediction",
    description="Run prediction on a single input",
    tags=["Prediction"],
)
async def predict(
    request: PredictRequest,
    _: bool = Depends(check_model_loaded),
):
    try:
        source_array = np.array(request.source, dtype=np.float32)
        
        if source_array.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty input data",
            )
        
        max_res = settings.MAX_RESOLUTION
        if (
            source_array.ndim >= 2 
            and (source_array.shape[-2] > max_res or source_array.shape[-1] > max_res)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input resolution too large. Max: {max_res}x{max_res}",
            )
        
        result = model_manager.predict(source_array, return_numpy=True)
        
        response_data = SinglePredictionResult(
            input_shape=result["input_shape"],
            output_shape=result["output_shape"],
            output=result["output"].tolist(),
            input=result["input"].tolist() if request.include_input else None,
        )
        
        return PredictResponse(
            success=True,
            message="Prediction completed successfully",
            data=response_data,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch Prediction",
    description="Run prediction on multiple inputs",
    tags=["Prediction"],
)
async def predict_batch(
    request: BatchPredictRequest,
    _: bool = Depends(check_model_loaded),
):
    try:
        max_batch = settings.MAX_BATCH_SIZE
        if len(request.sources) > max_batch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size too large. Max: {max_batch}",
            )
        
        source_arrays = [np.array(s, dtype=np.float32) for s in request.sources]
        results = model_manager.predict_batch(source_arrays, return_numpy=True)
        
        response_results = []
        for result in results:
            item = {
                "input_shape": result["input_shape"],
                "output_shape": result["output_shape"],
                "output": result["output"].tolist(),
            }
            if request.include_input:
                item["input"] = result["input"].tolist()
            response_results.append(item)
        
        return BatchPredictResponse(
            success=True,
            message="Batch prediction completed successfully",
            data={
                "predictions": response_results,
                "count": len(response_results),
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(
    "/api/test/sample",
    response_model=SampleInputResponse,
    summary="Get Sample Input",
    description="Generate sample input data for testing",
    tags=["Test"],
)
async def get_sample_input(
    resolution: int = 64,
    num_sources: int = 3,
):
    try:
        resolution = min(resolution, settings.MAX_RESOLUTION)
        
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
            "resolution": resolution,
            "num_sources": num_sources,
            "source": f.tolist(),
            "shape": [resolution, resolution],
        }
        
        return SampleInputResponse(
            success=True,
            message="Sample input generated successfully",
            data=sample,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample: {str(e)}",
        )


@app.post(
    "/api/test/predict",
    response_model=ServiceInfoResponse,
    summary="Test Prediction",
    description="Run prediction with automatically generated sample data",
    tags=["Test"],
)
async def test_predict(
    resolution: int = 64,
    num_sources: int = 3,
):
    try:
        if not model_manager.is_loaded():
            success, message, info = model_manager.load_model()
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model: {message}",
                )
        
        resolution = min(resolution, settings.MAX_RESOLUTION)
        
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
        
        result = model_manager.predict(f, return_numpy=True)
        output_np = result["output"]
        
        response_data = {
            "input_shape": result["input_shape"],
            "output_shape": result["output_shape"],
            "input": result["input"].tolist(),
            "output": result["output"].tolist(),
            "resolution": resolution,
            "model_info": model_manager.get_model_info(),
            "output_stats": {
                "min": float(output_np.min()),
                "max": float(output_np.max()),
                "mean": float(output_np.mean()),
                "std": float(output_np.std()),
            },
        }
        
        return ServiceInfoResponse(
            success=True,
            message="Test prediction completed successfully",
            data=response_data,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test prediction failed: {str(e)}",
        )


@app.get("/")
async def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/health",
            "info": "/api/info",
            "predict": "/api/predict",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting {settings.API_TITLE} on {settings.HOST}:{settings.PORT}")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"Device: {settings.DEVICE}")
    print(f"Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
