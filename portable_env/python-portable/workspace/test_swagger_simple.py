"""Simple test for Swagger UI"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Test API", version="1.0.0", docs_url=None, redoc_url=None)

SWAGGER_SIMPLE = """
<!DOCTYPE html>
<html>
<head>
  <title>Test Swagger</title>
  <script>
    function loadSpec() {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', '/openapi.json', true);
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          if (xhr.status === 200) {
            try {
              var spec = JSON.parse(xhr.responseText);
              document.getElementById('content').innerHTML = 
                '<h2>Loaded API: ' + spec.info.title + '</h2>' +
                '<p>Version: ' + spec.info.version + '</p>' +
                '<h3>Endpoints:</h3>' +
                '<ul>' + Object.keys(spec.paths).map(p => '<li>' + p + '</li>').join('') + '</ul>';
            } catch(e) {
              document.getElementById('content').innerHTML = 'Parse error: ' + e.message;
            }
          } else {
            document.getElementById('content').innerHTML = 'Load failed: ' + xhr.status;
          }
        }
      };
      xhr.send();
    }
    window.onload = loadSpec;
  </script>
</head>
<body>
  <div id="content">Loading...</div>
</body>
</html>
"""

@app.get("/docs", response_class=HTMLResponse)
async def docs():
    return HTMLResponse(content=SWAGGER_SIMPLE)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)