import logging
from pathlib import Path
from typing import Optional
from backends.base import ASRBackend

log = logging.getLogger(__name__)

class VibeVoiceBackend(ASRBackend):
    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
        # Note: VibeVoice often uses specific config loading
    ):
        self.name = "vibevoice"
        self.device = device
        
        log.info(f"Loading VibeVoice ({model_name}) on {device}...")
        
        # Delayed imports to avoid dependency issues with other backends
        from transformers import AutoModel, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

    def transcribe(self, audio_path: str) -> dict:
        """
        Runs VibeVoice inference and returns normalized output.
        """
        import torch
        import torchaudio

        # 1. Load Audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            
        # 2. Inference
        inputs = self.processor(audio=audio.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # VibeVoice inference call (Adjust based on exact model method)
            generated_ids = self.model.generate(**inputs)
            
        # 3. Decode
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 4. Parse into format
        # VibeVoice returns structured text (Speaker: text). 
        # You need to split this into 'text' (plain) and 'dialogue' (speaker-labeled).
        return {
            "text": transcription, 
            "dialogue": transcription # If the model already returns "Speaker: Text", this works
        }