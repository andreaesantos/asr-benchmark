import logging
from typing import Optional
import pydantic
import exca as xk

from backends.base import ASRBackend
from utils.utils import format_timestamped_dialogue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)

class VibeVoiceTask(pydantic.BaseModel):
    model_name: str
    audio_path: str
    device: str
    compute_type: str
    batch_size: int
    language: Optional[str] = None

    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    @infra.apply
    def run(self) -> dict[str, str]:
        return VibeVoiceBackend._execute_inference(
            self.model_name, self.audio_path, self.device, 
            self.compute_type, self.batch_size, self.language
        )
    
class VibeVoiceBackend(ASRBackend):
    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        language: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self.name = "vibevoice"
        log.info(f"VibeVoice backend initialized ({model_name}).")

    @staticmethod
    def _execute_inference(
        model_name: str, 
        audio_path: str, 
        device: str, 
        compute_type: str, 
        batch_size: int, 
        language: Optional[str]
    ) -> dict[str, str]:
        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor

        # Load resources
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()

        # Load and resample Audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            
        # Inference
        inputs = processor(audio=audio.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            
        segments = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        plain_text = " ".join([seg['Content'] for seg in segments])
        dialogue = format_timestamped_dialogue(segments)
    
        return {
            "text": plain_text, 
            "dialogue": dialogue 
        }

    def transcribe(self, audio_path: str) -> dict:
        """Runs the cached Task."""
        task = VibeVoiceTask(
            model_name=self.model_name,
            audio_path=audio_path,
            device=self.device,
            compute_type=self.compute_type,
            batch_size=self.batch_size,
            language=self.language
        )
        return task.run()