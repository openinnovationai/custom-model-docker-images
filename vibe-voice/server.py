import io, base64, torch, librosa
import numpy as np
import soundfile as sf
import litserve as ls
import logging
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

logging.basicConfig(level=logging.INFO)


class VibeVoiceLitAPI(ls.LitAPI):
    def setup(self, device):
        self.processor = VibeVoiceProcessor.from_pretrained("./models/VibeVoice-1.5B")
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            "./models/VibeVoice-1.5B",
            torch_dtype=torch.float16,
            device_map=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),  # @NOTE: use "mps" Mac
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=10)
        logging.info("VibeVoice model setup complete")

    def decode_request(self, request):
        logging.info(f"Received request: {request}")
        voice_samples = request.get("voice_samples")
        TARGET_SR = 24000

        if voice_samples is None:
            voice_samples = [
                "./voices/en-Alice_woman_bgm.wav",
                "./voices/en-Alice_woman.wav",
                "./voices/en-Carter_man.wav",
                "./voices/en-Frank_man.wav",
                "./voices/en-Maya_woman.wav",
                "./voices/in-Samuel_man.wav",
                "./voices/zh-Anchen_man_bgm.wav",
                "./voices/zh-Bowen_man.wav",
                "./voices/zh-Xinran_woman.wav",
            ]
        elif isinstance(voice_samples[0], (str, dict)) and not (
            isinstance(voice_samples[0], str) and voice_samples[0].startswith("./")
        ):
            processed_samples = []
            for sample in voice_samples:
                if isinstance(sample, dict):
                    audio_bytes = base64.b64decode(sample["audio_base64"])
                else:
                    audio_bytes = base64.b64decode(sample)

                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if audio_data.ndim > 1:
                    audio_data = librosa.to_mono(audio_data.T)
                if sr != TARGET_SR:
                    audio_data = librosa.resample(
                        audio_data, orig_sr=sr, target_sr=TARGET_SR
                    )

                processed_samples.append(audio_data)
            voice_samples = processed_samples

        return {"text": request["text"], "voice_samples": voice_samples}

    def predict(self, inputs):
        processor_inputs = self.processor(
            text=[inputs["text"]],
            voice_samples=inputs["voice_samples"],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        logging.info("Generation started...!")
        outputs = self.model.generate(
            **processor_inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
        )
        logging.info("Generation completed...!")

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Convert to numpy array for audio processing
            wav = outputs.speech_outputs[0].cpu().numpy()

            # The model may output with shape (1, n_samples), which scipy interprets as
            # n_samples channels. Squeeze to (n_samples,) for mono.
            if wav.ndim > 1:
                wav = wav.squeeze()

            wav = (wav * 32767).astype(np.int16)

            # Write to in-memory buffer
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wav, 24000, format="WAV")
            audio_data = audio_buffer.getvalue()
            audio_buffer.close()

            return {"audio_content": audio_data}
        else:
            return {"audio_content": None}

    def encode_response(self, prediction):
        if prediction["audio_content"] is not None:
            audio_content_base64 = base64.b64encode(prediction["audio_content"]).decode(
                "utf-8"
            )
            return {"audio_content": audio_content_base64, "content_type": "audio/wav"}
        else:
            return {"error": "No audio generated"}


if __name__ == "__main__":
    api = VibeVoiceLitAPI(api_path="/tts")
    server = ls.LitServer(api, healthcheck_path="/health-check")
    server.run(port=8080)
