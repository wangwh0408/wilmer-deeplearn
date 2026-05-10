import urllib.request
import json

# Get and save OpenAPI spec
response = urllib.request.urlopen('http://127.0.0.1:9090/openapi.json', timeout=10)
content = response.read().decode('utf-8')
spec = json.loads(content)

# Save to file
with open('openapi_spec.json', 'w', encoding='utf-8') as f:
    json.dump(spec, f, indent=2)

print('OpenAPI spec saved to openapi_spec.json')

# Show predict endpoint details
print('\n=== Predict Endpoint Details ===')
predict_spec = spec['paths']['/predict']['post']
print('Summary:', predict_spec.get('summary'))
print('Description:', predict_spec.get('description'))

# Show request body
request_body = predict_spec.get('requestBody', {})
print('\nRequest Body:')
content = request_body.get('content', {}).get('application/json', {})
schema = content.get('schema', {})
print('Schema ref:', schema.get('$ref'))

# Show responses
print('\nResponses:')
responses = predict_spec.get('responses', {})
for status, response_def in responses.items():
    print(f'  {status}: {response_def.get("description")}')

# Show schemas
print('\n=== Schema Details ===')
schemas = spec['components']['schemas']
for name, schema in schemas.items():
    if 'properties' in schema:
        props = list(schema['properties'].keys())[:3]
        print(f'  {name}: {len(schema["properties"])} properties ({", ".join(props)}...)')