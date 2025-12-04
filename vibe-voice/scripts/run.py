import json
from pathlib import Path


config_path = Path.cwd() / "models/VibeVoice-1.5B/preprocessor_config.json"


def main():
    with open(config_path) as f:
        config = json.load(f)

    config["language_model_pretrained_name"] = "/app/models/Qwen2.5-1.5B"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        print(f"✅ Config Updated:\n{config_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Failed to update configuration:\n{e}")
