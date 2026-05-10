import urllib.request
import json

# Test OpenAPI spec
print('=== Testing OpenAPI Spec ===')
try:
    response = urllib.request.urlopen('http://127.0.0.1:9090/openapi.json', timeout=10)
    content = response.read().decode('utf-8')
    spec = json.loads(content)
    
    print('Status:', response.status)
    print('Content length:', len(content), 'bytes')
    print('OpenAPI version:', spec.get('openapi'))
    print('API title:', spec.get('info', {}).get('title'))
    print('API version:', spec.get('info', {}).get('version'))
    
    # Check paths
    paths = spec.get('paths', {})
    print('\nAPI Endpoints:')
    for path, methods in paths.items():
        http_methods = [m for m in ['get', 'post', 'put', 'delete', 'patch'] if m in methods]
        print(f'  {path}: {", ".join(http_methods)}')
    
    # Check schemas
    schemas = spec.get('components', {}).get('schemas', {})
    print(f'\nSchemas found: {len(schemas)}')
    for name in schemas.keys():
        print(f'  - {name}')
        
except Exception as e:
    print('ERROR:', e)
    import traceback
    traceback.print_exc()