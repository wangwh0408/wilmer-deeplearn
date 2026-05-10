from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status: 'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Current timestamp")
    config: Optional[Dict[str, Any]] = Field(None, description="Service configuration")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information if loaded")


class ModelInfoResponse(BaseModel):
    path: str = Field(..., description="Model file path")
    modes: int = Field(..., description="Number of Fourier modes")
    width: int = Field(..., description="Number of channels in FNO layers")
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    resolution: int = Field(..., description="Default resolution")
    device: str = Field(..., description="Device (cpu/cuda)")
    total_parameters: int = Field(..., description="Total number of parameters")
    trainable_parameters: int = Field(..., description="Number of trainable parameters")


class PredictRequest(BaseModel):
    source: List[List[float]] = Field(
        ..., 
        description="Source term array (2D or 3D)",
        examples=[[[0.0] * 64 for _ in range(64)]]
    )
    include_input: bool = Field(
        default=False, 
        description="Whether to include input in response"
    )
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if not v:
            raise ValueError("Source cannot be empty")
        return v


class BatchPredictRequest(BaseModel):
    sources: List[List[List[float]]] = Field(
        ..., 
        description="List of source term arrays",
        examples=[[[[0.0] * 64 for _ in range(64)], [[0.0] * 64 for _ in range(64)]]]
    )
    include_input: bool = Field(
        default=False, 
        description="Whether to include input in response"
    )
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        if not v:
            raise ValueError("Sources cannot be empty")
        return v


class SinglePredictionResult(BaseModel):
    input_shape: List[int] = Field(..., description="Shape of input array")
    output_shape: List[int] = Field(..., description="Shape of output array")
    output: List[Any] = Field(..., description="Output array as nested lists")
    input: Optional[List[Any]] = Field(None, description="Input array (if include_input=True)")


class PredictResponse(BaseModel):
    success: bool = Field(default=True, description="Whether prediction succeeded")
    message: str = Field(default="Prediction completed successfully", description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    data: Optional[SinglePredictionResult] = Field(None, description="Prediction result")


class BatchPredictResponse(BaseModel):
    success: bool = Field(default=True, description="Whether prediction succeeded")
    message: str = Field(default="Batch prediction completed successfully", description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Batch prediction results")


class SampleInputResponse(BaseModel):
    success: bool = Field(default=True)
    message: str = Field(default="Sample input generated successfully")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = Field(None)


class LoadModelRequest(BaseModel):
    model_path: Optional[str] = Field(
        None, 
        description="Path to model file. If not provided, uses default path."
    )


class LoadModelResponse(BaseModel):
    success: bool = Field(..., description="Whether model loaded successfully")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = Field(None, description="Model information if loaded")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ServiceInfoResponse(BaseModel):
    success: bool = Field(default=True)
    message: str = Field(default="Success")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = Field(None)
