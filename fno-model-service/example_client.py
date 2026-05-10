import numpy as np
import requests
import json
from typing import Dict, Any, Optional


class FNOModelClient:
    def __init__(self, base_url: str = 'http://localhost:5000'):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f'{self.base_url}/api/health')
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        response = requests.get(f'{self.base_url}/api/info')
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if model_path:
            payload['model_path'] = model_path
        response = requests.post(
            f'{self.base_url}/api/model/load',
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def unload_model(self) -> Dict[str, Any]:
        response = requests.post(f'{self.base_url}/api/model/unload')
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        response = requests.get(f'{self.base_url}/api/model/info')
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        source: np.ndarray,
        include_input: bool = False
    ) -> Dict[str, Any]:
        payload = {
            'source': source.tolist(),
            'include_input': include_input
        }
        response = requests.post(
            f'{self.base_url}/api/predict',
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        sources: list,
        include_input: bool = False
    ) -> Dict[str, Any]:
        source_list = [s.tolist() if isinstance(s, np.ndarray) else s for s in sources]
        payload = {
            'sources': source_list,
            'include_input': include_input
        }
        response = requests.post(
            f'{self.base_url}/api/predict/batch',
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_sample_input(
        self,
        resolution: int = 64,
        num_sources: int = 3
    ) -> Dict[str, Any]:
        params = {
            'resolution': resolution,
            'num_sources': num_sources
        }
        response = requests.get(
            f'{self.base_url}/api/test/sample',
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def test_predict(
        self,
        resolution: int = 64,
        num_sources: int = 3
    ) -> Dict[str, Any]:
        payload = {
            'resolution': resolution,
            'num_sources': num_sources
        }
        response = requests.post(
            f'{self.base_url}/api/test/predict',
            json=payload
        )
        response.raise_for_status()
        return response.json()


def generate_sample_input(resolution: int = 64, num_sources: int = 3) -> np.ndarray:
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
    
    return f


def run_example():
    print('=' * 60)
    print('FNO Model Service - Example Usage')
    print('=' * 60)
    print()
    
    client = FNOModelClient('http://localhost:5000')
    
    print('[1] Health Check...')
    health = client.health_check()
    print(f'    Status: {health["status"]}')
    print(f'    Model Loaded: {health["model_loaded"]}')
    print()
    
    print('[2] Get Service Info...')
    info = client.get_info()
    print(f'    Service: {info["data"]["service"]}')
    print(f'    Version: {info["data"]["version"]}')
    print()
    
    if not health['model_loaded']:
        print('[3] Loading Model...')
        result = client.load_model()
        if result['success']:
            print(f'    Model loaded: {result["data"]["model_info"]["total_parameters"]:,} parameters')
        else:
            print(f'    Warning: {result["error"]}')
        print()
    
    print('[4] Get Sample Input...')
    sample = client.get_sample_input(resolution=64, num_sources=3)
    print(f'    Resolution: {sample["data"]["resolution"]}')
    print(f'    Shape: {sample["data"]["shape"]}')
    print()
    
    print('[5] Test Prediction...')
    test_result = client.test_predict(resolution=64, num_sources=3)
    if test_result['success']:
        output_shape = test_result['data']['output_shape']
        print(f'    Input Shape: {test_result["data"]["input_shape"]}')
        print(f'    Output Shape: {output_shape}')
        print(f'    Model Info: {test_result["data"]["model_info"]}')
    else:
        print(f'    Error: {test_result["error"]}')
    print()
    
    print('[6] Custom Prediction...')
    source = generate_sample_input(resolution=64, num_sources=2)
    prediction = client.predict(source, include_input=True)
    if prediction['success']:
        print(f'    Input Shape: {prediction["data"]["input_shape"]}')
        print(f'    Output Shape: {prediction["data"]["output_shape"]}')
        output_np = np.array(prediction['data']['output'])
        print(f'    Output Stats: min={output_np.min():.4f}, max={output_np.max():.4f}, mean={output_np.mean():.4f}')
    else:
        print(f'    Error: {prediction["error"]}')
    print()
    
    print('[7] Batch Prediction...')
    sources = [
        generate_sample_input(resolution=64, num_sources=2),
        generate_sample_input(resolution=64, num_sources=3),
        generate_sample_input(resolution=64, num_sources=4),
    ]
    batch_result = client.predict_batch(sources)
    if batch_result['success']:
        print(f'    Count: {batch_result["data"]["count"]}')
        for i, pred in enumerate(batch_result['data']['predictions']):
            print(f'    Sample {i+1}: {pred["output_shape"]}')
    else:
        print(f'    Error: {batch_result["error"]}')
    print()
    
    print('=' * 60)
    print('All examples completed!')
    print('=' * 60)


if __name__ == '__main__':
    run_example()
