import urllib.request

# Get the full HTML content from /docs
print('=== Getting /docs content ===')
try:
    response = urllib.request.urlopen('http://127.0.0.1:9090/docs', timeout=10)
    content = response.read().decode('utf-8')
    
    print('Status:', response.status)
    print('Content length:', len(content), 'bytes')
    
    # Save to file for debugging
    with open('docs_debug.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Saved to docs_debug.html')
    
    # Check key parts
    print()
    print('=== Checking key content ===')
    print('Has swagger-ui container:', '<div id="swagger-ui"' in content)
    print('Has script tag:', '<script>' in content)
    print('Has style tag:', '<style>' in content)
    print('Has window.onload:', 'window.onload' in content)
    
except Exception as e:
    print('ERROR:', e)