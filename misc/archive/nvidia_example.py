
import os
import requests
from dotenv import load_dotenv

load_dotenv()

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True


headers = {
  "Authorization": f"Bearer {os.getenv('NVIDIA_KEY')}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "qwen/qwen3-coder-480b-a35b-instruct",
  "messages": [{"role":"user","content":"Hello"}],
  "max_tokens": 2048,
  "temperature": 0.15,
  "top_p": 1.00,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}



response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
