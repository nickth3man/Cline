import os
import yaml
import json
import requests
import logging
import time
from typing import Optional, Dict, List, Any

# Default configuration values (used if config file is missing or invalid)
DEFAULT_CONFIG = {
    "defaults": {
        "transcription_model": "openai/whisper-large-v3",
        "correction_model": "openai/gpt-4.1-mini",
    },
    "api_fetch": {
        "enabled": True,
        "cache_file": ".openrouter_models_cache.json",
        "cache_duration_hours": 24,
    },
}

OPENROUTER_API_MODELS_URL = "https://openrouter.ai/api/v1/models"


def load_config(config_path="config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        logging.warning(
            f"Configuration file not found at {config_path}. Using default config."
        )
        return DEFAULT_CONFIG
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Basic validation/merging with defaults could be added here
        # For now, assume the structure is correct if file exists
        if not config:  # Handle empty file case
            logging.warning(
                f"Configuration file {config_path} is empty. Using default config."
            )
            return DEFAULT_CONFIG
        # Simple merge logic (could be more sophisticated)
        merged_config = DEFAULT_CONFIG.copy()
        if "defaults" in config:
            merged_config["defaults"].update(config.get("defaults", {}))
        if "api_fetch" in config:
            merged_config["api_fetch"].update(config.get("api_fetch", {}))
        return merged_config
    except yaml.YAMLError as e:
        logging.error(
            f"Error parsing configuration file {config_path}: {e}. Using default config."
        )
        return DEFAULT_CONFIG
    except Exception as e:
        logging.error(
            f"Unexpected error loading configuration file {config_path}: {e}. Using default config."
        )
        return DEFAULT_CONFIG


def _fetch_models_from_api() -> Optional[List[Dict[str, Any]]]:
    """Fetches model list from OpenRouter API."""
    try:
        response = requests.get(
            OPENROUTER_API_MODELS_URL, timeout=10
        )  # 10 second timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]  # Return the list of model dictionaries
        else:
            logging.error(f"Unexpected format received from OpenRouter API: {data}")
            return None
    except requests.exceptions.Timeout:
        logging.error("Timeout occurred while fetching models from OpenRouter API.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching models from OpenRouter API: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from OpenRouter API: {e}")
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error fetching models from OpenRouter API: {e}", exc_info=True
        )
        return None


def get_available_models(config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Gets available LLM models (chat/completion only), using cache if valid, otherwise fetching from API.
    Only caches the filtered, relevant models for dropdowns.
    Returns None if fetching is disabled or fails and cache is invalid/missing.
    """
    api_fetch_config = config.get("api_fetch", DEFAULT_CONFIG["api_fetch"])
    if not api_fetch_config.get("enabled", True):
        logging.info("OpenRouter API fetching is disabled in config.")
        return None

    cache_file = api_fetch_config.get(
        "cache_file", DEFAULT_CONFIG["api_fetch"]["cache_file"]
    )
    cache_duration_sec = (
        api_fetch_config.get(
            "cache_duration_hours", DEFAULT_CONFIG["api_fetch"]["cache_duration_hours"]
        )
        * 3600
    )

    def filter_llm_models(models):
        filtered = []
        for model in models:
            model_id = model.get("id", "").lower()
            model_type = model.get("type", "").lower()
            # Exclude non-chat/completion, whisper, audio, image, or other non-text models
            if model_type not in ("chat", "completion"):
                continue
            if any(x in model_id for x in ("whisper", "audio", "image")) or any(
                x in model_type for x in ("whisper", "audio", "image")
            ):
                continue
            # Add supported features
            features = {
                "tool_calling": "tools" in model.get("supported_parameters", []),
                "structured_outputs": "structured_outputs"
                in model.get("supported_parameters", []),
                "max_context": model.get("context_length"),
                "variants": model.get("variants", []),
            }
            filtered.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name", model.get("id")),
                    "input_price_per_million": model.get("pricing", {}).get(
                        "input", None
                    ),
                    "output_price_per_million": model.get("pricing", {}).get(
                        "output", None
                    ),
                    "provider": model.get("provider", ""),
                    "description": model.get("description", ""),
                    "features": features,
                }
            )
        return filtered

    # Check cache
    if os.path.exists(cache_file):
        try:
            cache_mtime = os.path.getmtime(cache_file)
            if (time.time() - cache_mtime) < cache_duration_sec:
                if os.path.getsize(cache_file) > 5 * 1024 * 1024:
                    logging.warning(
                        f"Cache file {cache_file} is too large. Deleting and re-fetching."
                    )
                    os.remove(cache_file)
                else:
                    logging.info(f"Using cached model list from {cache_file}.")
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    if isinstance(cached_data, list) and all(
                        isinstance(item, dict) for item in cached_data
                    ):
                        return cached_data
                    else:
                        logging.warning(
                            f"Invalid data format found in cache file {cache_file}. Will attempt to re-fetch."
                        )
            else:
                logging.info(
                    "Cache file is outdated. Attempting to fetch fresh model list."
                )
        except Exception as e:
            logging.warning(
                f"Error accessing cache file {cache_file}: {e}. Will attempt to re-fetch."
            )

    # Fetch from API if cache is invalid/missing/outdated or fetch is forced
    logging.info("Fetching available models from OpenRouter API...")
    fetched_models = _fetch_models_from_api()

    if fetched_models is not None:
        filtered_models = filter_llm_models(fetched_models)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(filtered_models, f, indent=2)
            logging.info(
                f"Successfully fetched and cached filtered model list to {cache_file}."
            )
        except IOError as e:
            logging.error(f"Failed to write model list cache to {cache_file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error writing cache file {cache_file}: {e}")
        return filtered_models
    else:
        logging.error(
            "Failed to fetch models from OpenRouter API. No model list available."
        )
        return None


def get_available_llm_models_with_pricing(
    config: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Returns a list of all available OpenRouter LLM models (text-only, not Whisper/audio) with pricing info.
    Each entry: {
        'id': model_id,
        'name': model_name,
        'input_price_per_million': ...,
        'output_price_per_million': ...,
        ...
    }
    """
    all_models = get_available_models(config)
    if not all_models:
        return None
    llm_models = []
    for model in all_models:
        # Exclude Whisper/audio models (only allow text LLMs)
        if "whisper" in model.get("id", "").lower():
            continue
        # Only include models that support chat/completions
        if model.get("type") not in ("chat", "completion"):
            continue
        # Extract pricing info if available
        input_price = model.get("pricing", {}).get("input", None)
        output_price = model.get("pricing", {}).get("output", None)
        llm_models.append(
            {
                "id": model.get("id"),
                "name": model.get("name", model.get("id")),
                "input_price_per_million": input_price,
                "output_price_per_million": output_price,
                "provider": model.get("provider", ""),
                "description": model.get("description", ""),
            }
        )
    return llm_models


# Example usage (can be removed or put under if __name__ == "__main__")
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     cfg = load_config('../config.yaml') # Adjust path if running directly
#     print("Loaded Config:", cfg)
#     models = get_available_models(cfg)
#     if models:
#         print(f"\nFetched/Cached {len(models)} models.")
#         # Example: Print names of first 5 models
#         for model in models[:5]:
#             print(f"- {model.get('id')}")
#     else:
#         print("\nCould not retrieve model list.")
