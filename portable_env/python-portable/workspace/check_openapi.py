import requests
import json

response = requests.get('http://localhost:8000/openapi.json')
spec = response.json()

print("=== OpenAPI Spec Structure ===")
print(f"Version: {spec.get('openapi')}")
print(f"Title: {spec.get('info', {}).get('title')}")
print(f"\n=== Paths ===")

for path, methods in spec.get('paths', {}).items():
    print(f"\nPath: {path}")
    for method, op in methods.items():
        if method in ['get', 'post', 'put', 'delete', 'patch']:
            print(f"  Method: {method.upper()}")
            print(f"    Summary: {op.get('summary', '')}")
            print(f"    Parameters: {len(op.get('parameters', []))}")
            for p in op.get('parameters', []):
                print(f"      - {p.get('name')} ({p.get('in', '')})")
            has_body = 'requestBody' in op
            print(f"    Has requestBody: {has_body}")
            if has_body:
                body = op.get('requestBody', {})
                content = body.get('content', {})
                print(f"      Content types: {list(content.keys())}")
                if 'application/json' in content:
                    schema = content['application/json'].get('schema', {})
                    print(f"      Schema has properties: {'properties' in schema}")
                    print(f"      Schema: {json.dumps(schema, indent=2)[:500]}...")
            print(f"    Responses: {list(op.get('responses', {}).keys())}")

print("\n=== Components ===")
components = spec.get('components', {})
print(f"Schemas: {list(components.get('schemas', {}).keys())}")