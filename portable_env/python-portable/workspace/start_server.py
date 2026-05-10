import subprocess
import sys

result = subprocess.run([sys.executable, 'model_server_fastapi.py', '--no_load', '--port', '8000'], 
                       capture_output=True, text=True, timeout=10)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)