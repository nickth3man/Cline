import sys
import requests
import os
import json
import argparse

def fetch_models(api_key=None):
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key: OpenRouter API key. If None, will try to get from environment.
        
    Returns:
        List of model dictionaries or None on failure.
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment variables.", file=sys.stderr)
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        models_data = response.json()
        return models_data.get("data", [])
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch available models from OpenRouter API")
    parser.add_argument("--format", choices=["text", "json"], default="text", 
                        help="Output format (text or json)")
    parser.add_argument("--output", help="Output file path (if not specified, prints to stdout)")
    
    args = parser.parse_args()
    
    models_list = fetch_models()
    if not models_list:
        sys.exit(1)
        
    if args.format == "json":
        output = json.dumps(models_list, indent=2)
    else:
        # Text format: ID:Name
        output = "\n".join([f"{model.get('id')}:{model.get('name', model.get('id'))}" for model in models_list])
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)
