## Whisper Deployment with segments

Reference: https://github.com/openai/whisper

## Download Model Weights

Update the `Dockerfile` based on the model weights downloaded.

- [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
- [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
- [medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
- [large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt)
- [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)
- [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)
- [large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt)

## Build and Run

```bash
# Build for NVIDIA GPU
make build-nvidia

# Build for CPU
make build-cpu

# Run NVIDIA GPU version
make run-nvidia

# Run CPU version
make run-cpu
```

## API Usage

### Health Check
```bash
curl http://localhost:8080/health-check
```

### Transcribe Audio

#### Upload Audio File (Recommended)
```bash
# Basic transcription (auto-detect language)
curl -X POST http://localhost:8080/asr \
  -F "audio=@/path/to/audio.mp3"

# Transcription with specific language
curl -X POST http://localhost:8080/asr \
  -F "audio=@/path/to/audio.mp3" \
  -F "language=en"
```

## Getting help

```sh
make

# OR 

make help
```