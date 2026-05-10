from model_server_fastapi import app
from fastapi.testclient import TestClient

client = TestClient(app)

r = client.get('/openapi.json')
print('Status:', r.status_code)
print('Headers:', r.headers)
print()
print('Content:')
import json
spec = r.json()
print(json.dumps(spec, indent=2))
