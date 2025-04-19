import sys
import requests
import os
import json

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not set in environment variables.", file=sys.stderr)
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}"}
try:
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
    response.raise_for_status()
    models_data = response.json()
    models_list = models_data.get("data", [])
    # Print model ID and name
    for model in models_list:
        print(f"{model.get('id')}:{model.get('name', model.get('id'))}")
except Exception as e:
    print(f"Error fetching OpenRouter models: {e}", file=sys.stderr)
    sys.exit(1)