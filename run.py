import requests

def format_time(seconds):
    """Converts seconds into MM:SS format."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def process_diarization(file_path, api_key, deployment_id):
    url = f"https://inference.prod.openinnovation.ai/models/{deployment_id}/proxy/v1/audio/diarization"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }
    print(f"--- Uploading {file_path} ---")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(url, headers=headers, data=f)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"\n{'Speaker':<15} | {'Start':<10} | {'End':<10} | {'Duration'}")
        print("-" * 55)

        for turn in data:
            name = turn.get('speaker_name', 'Unknown')
            start = turn.get('turn_onset', 0)
            duration = turn.get('turn_duration', 0)
            end = start + duration

            print(f"{name:<15} | {format_time(start):<10} | {format_time(end):<10} | {duration:.2f}s")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

API_KEY = "sk-giKsMDFxOMVegFveBYTCAKjXvL2fZqKM9cqZ8JblEOQ"
DEPLOYMENT_ID = "e4b37d25-2459-4dd8-9d5e-a78b2ad2ed08"
AUDIO_FILE_PATH = "./podcast.wav"
if __name__ == "__main__":
    process_diarization(AUDIO_FILE_PATH, API_KEY, DEPLOYMENT_ID)