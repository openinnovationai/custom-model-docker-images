import whisper
import torch
import litserve as ls
import logging
import os
import shutil
import tempfile
from fastapi import UploadFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERBOSE_LOGGING = os.getenv("OICM_VERBOSE_LOGGING", False) == "True"

class WhisperAPI(ls.LitAPI):
    def setup(self, device):
        MODEL_NAME = "turbo"
        logger.info(f"Loading model '{MODEL_NAME}' on {device}...")
        self.model = whisper.load_model(MODEL_NAME, device=device)
        logger.info("Model loaded successfully")
    
    def decode_request(self, request):
        uploaded_file: UploadFile = request["audio"]
        language = request.get("language", None) # Default to None if not provided

        logger.info(f"Received request - filename: {uploaded_file.filename}, language: {language}")

        # Whisper requires a file path (str) or numpy array, not a file-like object
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            shutil.copyfileobj(uploaded_file.file, tmp)
            tmp_path = tmp.name
        
        return tmp_path, language
    
    def predict(self, inputs):
        audio_path, language = inputs
        logger.info(f"Starting transcription on {audio_path}...")
        
        try:
            result = self.model.transcribe(audio_path, language=language, verbose=VERBOSE_LOGGING)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        logger.info("Transcription completed")
        return result
    
    def encode_response(self, output):
        return {"result": output}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    api = WhisperAPI(api_path="/asr") # @NOTE: user modifiable endpoint for ASR
    server = ls.LitServer(api, accelerator=device, healthcheck_path="/health-check") # @NOTE: required health-check endpoint for OICM platform
    server.run(port=8080) # @NOTE: required port 8080 for OICM platform