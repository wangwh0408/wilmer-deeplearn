"""
Example client for FNO Model Service (FastAPI version)

This client demonstrates how to interact with the FastAPI-based FNO model service.
"""

import numpy as np
import httpx
from typing import Dict, Any, Optional, List


class FNOClient:
    """Client for interacting with the FNO Model Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status."""
        response = self.client.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        response = self.client.get(f"{self.base_url}/api/info")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information."""
        response = self.client.get(f"{self.base_url}/api/model/info")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a model from file."""
        payload = {}
        if model_path:
            payload["model_path"] = model_path
        response = self.client.post(
            f"{self.base_url}/api/model/load",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model."""
        response = self.client.post(f"{self.base_url}/api/model/unload")
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        source: np.ndarray,
        include_input: bool = False
    ) -> Dict[str, Any]:
        """Run prediction on a single input."""
        payload = {
            "source": source.tolist(),
            "include_input": include_input
        }
        response = self.client.post(
            f"{self.base_url}/api/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        sources: List[np.ndarray],
        include_input: bool = False
    ) -> Dict[str, Any]:
        """Run prediction on multiple inputs."""
        source_list = [s.tolist() for s in sources]
        payload = {
            "sources": source_list,
            "include_input": include_input
        }
        response = self.client.post(
            f"{self.base_url}/api/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_sample_input(
        self,
        resolution: int = 64,
        num_sources: int = 3
    ) -> Dict[str, Any]:
        """Get sample input data."""
        params = {
            "resolution": resolution,
            "num_sources": num_sources
        }
        response = self.client.get(
            f"{self.base_url}/api/test/sample",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def test_predict(
        self,
        resolution: int = 64,
        num_sources: int = 3
    ) -> Dict[str, Any]:
        """Run a test prediction with sample data."""
        params = {
            "resolution": resolution,
            "num_sources": num_sources
        }
        response = self.client.post(
            f"{self.base_url}/api/test/predict",
            params=params
        )
        response.raise_for_status()
        return response.json()


def generate_sample_input(
    resolution: int = 64,
    num_sources: int = 3,
    seed: int = 42
) -> np.ndarray:
    """Generate a sample input array for testing."""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    f = np.zeros((resolution, resolution))
    np.random.seed(seed)
    for _ in range(num_sources):
        x0 = np.random.uniform(0.2, 0.8)
        y0 = np.random.uniform(0.2, 0.8)
        sigma = np.random.uniform(0.05, 0.15)
        amplitude = np.random.uniform(1.0, 5.0)
        f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    return f


def run_example():
    """Run a complete example of using the FNO client."""
    print("=" * 60)
    print("FNO Model Service (FastAPI) - Example Usage")
    print("=" * 60)
    print()
    
    with FNOClient("http://localhost:8000") as client:
        print("[1] Health Check...")
        health = client.health_check()
        print(f"    Status: {health['status']}")
        print(f"    Model Loaded: {health['model_loaded']}")
        print()
        
        print("[2] Get Service Info...")
        info = client.get_service_info()
        print(f"    Service: {info['data']['service']}")
        print(f"    Version: {info['data']['version']}")
        print()
        
        if not health['model_loaded']:
            print("[3] Loading Model...")
            try:
                result = client.load_model()
                if result['success']:
                    print(f"    Model loaded: {result['data']['model_info']['total_parameters']:,} parameters")
                else:
                    print(f"    Warning: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"    Warning: {e}")
            print()
        
        print("[4] Get Sample Input...")
        sample = client.get_sample_input(resolution=64, num_sources=3)
        print(f"    Resolution: {sample['data']['resolution']}")
        print(f"    Shape: {sample['data']['shape']}")
        print()
        
        print("[5] Test Prediction (using sample data)...")
        test_result = client.test_predict(resolution=64, num_sources=3)
        if test_result['success']:
            output_shape = test_result['data']['output_shape']
            stats = test_result['data']['output_stats']
            print(f"    Input Shape: {test_result['data']['input_shape']}")
            print(f"    Output Shape: {output_shape}")
            print(f"    Output Stats:")
            print(f"      Min:  {stats['min']:.6f}")
            print(f"      Max:  {stats['max']:.6f}")
            print(f"      Mean: {stats['mean']:.6f}")
            print(f"      Std:  {stats['std']:.6f}")
        else:
            print(f"    Error: {test_result.get('error', 'Unknown error')}")
        print()
        
        print("[6] Custom Prediction...")
        source = generate_sample_input(resolution=64, num_sources=2)
        prediction = client.predict(source, include_input=True)
        if prediction['success']:
            print(f"    Input Shape: {prediction['data']['input_shape']}")
            print(f"    Output Shape: {prediction['data']['output_shape']}")
            output_np = np.array(prediction['data']['output'])
            print(f"    Output Stats: min={output_np.min():.4f}, max={output_np.max():.4f}, mean={output_np.mean():.4f}")
        else:
            print(f"    Error: {prediction.get('error', 'Unknown error')}")
        print()
        
        print("[7] Batch Prediction...")
        sources = [
            generate_sample_input(resolution=64, num_sources=2, seed=1),
            generate_sample_input(resolution=64, num_sources=3, seed=2),
            generate_sample_input(resolution=64, num_sources=4, seed=3),
        ]
        batch_result = client.predict_batch(sources)
        if batch_result['success']:
            print(f"    Count: {batch_result['data']['count']}")
            for i, pred in enumerate(batch_result['data']['predictions']):
                print(f"    Sample {i+1}: {pred['output_shape']}")
        else:
            print(f"    Error: {batch_result.get('error', 'Unknown error')}")
        print()
        
        print("[8] API Documentation...")
        print(f"    Swagger UI:  {client.base_url}/docs")
        print(f"    ReDoc:       {client.base_url}/redoc")
        print(f"    OpenAPI:     {client.base_url}/openapi.json")
        print()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    run_example()
