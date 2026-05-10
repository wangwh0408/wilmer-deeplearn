import urllib.request

# Test Swagger UI
print('=== Testing Swagger UI (/docs) ===')
try:
    response = urllib.request.urlopen('http://127.0.0.1:9090/docs', timeout=10)
    content = response.read().decode('utf-8')
    print('Status:', response.status)
    print('Content length:', len(content), 'bytes')
    print('Contains swagger-ui:', 'swagger-ui' in content.lower())
    print('Contains FNO2:', 'FNO2' in content)
    print('SUCCESS: Swagger UI loaded from local resources!')
except Exception as e:
    print('ERROR:', e)

print()

# Test ReDoc
print('=== Testing ReDoc (/redoc) ===')
try:
    response = urllib.request.urlopen('http://127.0.0.1:9090/redoc', timeout=10)
    content = response.read().decode('utf-8')
    print('Status:', response.status)
    print('Content length:', len(content), 'bytes')
    print('Contains redoc:', 'redoc' in content.lower())
    print('Contains FNO2:', 'FNO2' in content)
    print('SUCCESS: ReDoc loaded from local resources!')
except Exception as e:
    print('ERROR:', e)

print()

# Test OpenAPI spec
print('=== Testing OpenAPI Spec (/openapi.json) ===')
try:
    response = urllib.request.urlopen('http://127.0.0.1:9090/openapi.json', timeout=10)
    content = response.read().decode('utf-8')
    print('Status:', response.status)
    print('Content length:', len(content), 'bytes')
    print('Contains openapi:', 'openapi' in content.lower())
    print('SUCCESS: OpenAPI spec available!')
except Exception as e:
    print('ERROR:', e)

print()
print('=== All tests completed! ===')