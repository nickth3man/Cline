import os
import sys
import argparse
import spacy
import requests
import json
import logging
from typing import Optional, List, Dict, Any

# Add src directory to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import model_manager

def load_spacy_model():
    # Load the small English model for sentence segmentation and cleanup
    return spacy.load("en_core_web_sm")

def process_transcript(text, nlp):
    # Use SpaCy to split text into sentences and perform cleanup
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Join sentences with double newlines for markdown paragraph separation
    return "\n\n".join(sentences)

def process_all_transcripts(output_dir="output", video_ids=None):
    nlp = load_spacy_model()
    # Iterate over all folders in output directory
    for video_folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, video_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract video_id from folder name if possible
        # This is a simplification - in a real implementation, you might need a more robust way to map folders to video IDs
        video_id = video_folder.split("_")[-1] if "_" in video_folder else video_folder
        
        # Skip if video_ids is provided and this folder's video_id is not in the list
        if video_ids and video_id not in video_ids:
            continue

        # Construct raw transcript filename
        raw_transcript_filename = f"{video_folder}_transcript.txt"
        raw_transcript_path = os.path.join(folder_path, raw_transcript_filename)

        print(f"VIDEO_STATUS:{video_id}:Processing transcript for {video_folder}") # Structured status

        if not os.path.exists(raw_transcript_path):
            print(f"VIDEO_ERROR:{video_id}:Raw transcript not found for {video_folder}, skipping.") # Structured error
            continue

        try:
            # Read raw transcript
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Process transcript with SpaCy
            processed_text = process_transcript(raw_text, nlp)

            # Save processed transcript as _corrected.md
            corrected_filename = f"{video_folder}_corrected.md"
            corrected_path = os.path.join(folder_path, corrected_filename)
            with open(corrected_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            print(f"VIDEO_STATUS:{video_id}:Processed transcript saved to {corrected_path}") # Structured status

            # Delete raw transcript file after successful processing
            if os.path.exists(raw_transcript_path):
                os.remove(raw_transcript_path)
                print(f"VIDEO_STATUS:{video_id}:Deleted raw transcript file: {raw_transcript_path}") # Structured status

        except Exception as e:
            print(f"VIDEO_ERROR:{video_id}:Error processing transcript for {video_folder}: {e}") # Structured error

def fetch_openrouter_models() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch available OpenRouter models using model_manager utility.
    Returns list of model dicts or None on failure.
    """
    config = model_manager.load_config()
    if not config["api_fetch"]["enabled"]:
        logging.info("Dynamic model fetching disabled in config.")
        return None
    models = model_manager._fetch_models_from_api()
    if models is None:
        logging.error("Failed to fetch models from OpenRouter API.")
    return models

def estimate_price(tokens: int, model_info: Dict[str, Any]) -> float:
    """
    Estimate price for given token count and model pricing info.
    Assumes model_info contains 'price_per_1k_tokens' key in USD.
    """
    price_per_1k = model_info.get("price_per_1k_tokens", 0.0)
    return (tokens / 1000) * price_per_1k

def call_openrouter_llm(prompt: str, model_id: str, api_key: str) -> Optional[str]:
    """
    Call OpenRouter chat completions API with given prompt and model.
    Returns the generated text or None on failure.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Extract generated text from response
        choices = data.get("choices")
        if choices and len(choices) > 0:
            return choices[0].get("message", {}).get("content")
        else:
            logging.error(f"No choices found in OpenRouter response: {data}")
            return None
    except requests.RequestException as e:
        logging.error(f"OpenRouter API request failed: {e}")
        return None

def process_llm_correction_and_summarization(output_dir="output", model_id=None, api_key=None, video_ids=None):
    """
    For each _corrected.md transcript, perform LLM correction and summarization using OpenRouter.
    Save summary as _summary.md in the same folder.
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("OPENROUTER_API_KEY not set in environment or argument.")
        return

    if model_id is None:
        # Use default correction model from config if not specified
        config = model_manager.load_config()
        model_id = config["defaults"].get("correction_model", "openai/gpt-4.1-mini")

    for video_folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, video_folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Extract video_id from folder name if possible
        # This is a simplification - in a real implementation, you might need a more robust way to map folders to video IDs
        video_id = video_folder.split("_")[-1] if "_" in video_folder else video_folder
        
        # Skip if video_ids is provided and this folder's video_id is not in the list
        if video_ids and video_id not in video_ids:
            continue
            
        corrected_filename = f"{video_folder}_corrected.md"
        corrected_path = os.path.join(folder_path, corrected_filename)
        if not os.path.exists(corrected_path):
            print(f"VIDEO_ERROR:{video_id}:Corrected transcript not found for {video_folder}, skipping LLM step.")
            continue
        with open(corrected_path, "r", encoding="utf-8") as f:
            corrected_text = f.read()

        # Compose prompt for correction and summarization
        prompt = (
            "Please correct any errors in the following transcript and provide a concise summary:\n\n"
            f"{corrected_text}\n\nSummary:"
        )

        print(f"VIDEO_STATUS:{video_id}:Processing LLM correction and summarization using model {model_id}")

        summary = call_openrouter_llm(prompt, model_id, api_key)
        if summary is None:
            print(f"VIDEO_ERROR:{video_id}:LLM correction and summarization failed.")
            continue

        summary_filename = f"{video_folder}_summary.md"
        summary_path = os.path.join(folder_path, summary_filename)
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"VIDEO_STATUS:{video_id}:Summary saved to {summary_path}")
        except Exception as e:
            print(f"VIDEO_ERROR:{video_id}:Failed to save summary: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and summarize transcripts")
    parser.add_argument("--video_ids", help="Comma-separated list of video IDs to process")
    parser.add_argument("--model_name", help="OpenRouter model ID to use for summarization")
    parser.add_argument("--output_path", default="output", help="Path to the output directory")
    
    args = parser.parse_args()
    
    # Process video_ids if provided
    video_id_list = None
    if args.video_ids:
        video_id_list = args.video_ids.split(",")
    
    # Run transcript processing step
    process_all_transcripts(args.output_path, video_id_list)

    # Run LLM correction and summarization step with specified model
    process_llm_correction_and_summarization(args.output_path, args.model_name, video_ids=video_id_list)
