import logging, os

from backends.base import ASRBackend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

class OmniBackend(ASRBackend):
    """
    Qwen2-Audio-Omni (Qwen2-Audio-7B-Instruct variant with Omni capabilities).
    Falls back gracefully to the standard Qwen2-Audio-7B-Instruct if the Omni
    checkpoint is unavailable.

    HuggingFace: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
    Omni:        https://huggingface.co/Qwen/Qwen2.5-Omni-7B  (when released)

    Requires: pip install transformers>=4.45 accelerate soundfile librosa
    GPU with ≥16 GB VRAM recommended (or use load_in_4bit=True).
    """

    # Prefer the Omni checkpoint; fall back to the standard instruct model.
    DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
    FALLBACK_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    PROMPT = "Please transcribe the spoken content in this audio exactly."

    def __init__(
        self,
        name:        str = "omni",
        model_name:   str = DEFAULT_MODEL,
        device:       str = "cuda",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoProcessor

        log.info(f"Loading Omni model ({model_name}) …")

        self.name      = name
        self.model_name = model_name
        self.processor = self._load_processor(model_name)
        self.model     = self._load_model(model_name, load_in_4bit, torch)
        self.device    = device
        log.info("Omni ready.")

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_processor(model_name: str):
        from transformers import AutoProcessor
        try:
            return AutoProcessor.from_pretrained(model_name)
        except Exception as exc:
            log.warning(
                f"Could not load processor for '{model_name}': {exc}\n"
                f"  → Falling back to {OmniBackend.FALLBACK_MODEL}"
            )
            return AutoProcessor.from_pretrained(OmniBackend.FALLBACK_MODEL)

    @staticmethod
    def _load_model(model_name: str, load_in_4bit: bool, torch):
        kwargs: dict = {"device_map": "auto"}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        # Try Omni-specific class first, then generic conditional generation.
        for cls_name in (
            "Qwen2_5OmniForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
        ):
            try:
                import transformers
                cls = getattr(transformers, cls_name)
                return cls.from_pretrained(model_name, **kwargs)
            except (AttributeError, OSError, Exception):
                continue

        raise RuntimeError(
            f"Could not load any compatible model class for '{model_name}'. "
            "Ensure transformers>=4.45 is installed and the checkpoint exists."
        )

    # ── public API ───────────────────────────────────────────────────────────

    def _build_inputs(self, text_prompt: str, audio):
        """Build processor inputs robustly across transformers/Qwen variants."""
        audio_keys = ("audio", "audios", "input_audio", "speech")
        feature_keys = {"input_features", "audio_values", "feature_attention_mask"}

        for key in audio_keys:
            try:
                inputs = self.processor(
                    text=text_prompt,
                    **{key: [audio]},
                    return_tensors="pt",
                )
                if any(k in inputs for k in feature_keys):
                    return inputs
            except Exception:
                continue

        # Last resort: text-only (may still produce output, but not true ASR)
        log.warning(
            "Omni processor did not accept audio input keys; using text-only fallback."
        )
        return self.processor(text=text_prompt, return_tensors="pt")

    def transcribe(self, audio_path: str) -> dict[str, str]:
        import librosa, torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text",  "text": self.PROMPT},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        sr        = self.processor.feature_extractor.sampling_rate
        audio, _  = librosa.load(audio_path, sr=sr, mono=True)

        inputs = self._build_inputs(text_prompt, audio)

        # With device_map="auto", model.device can be "meta". Don't force inputs there.
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            try:
                inputs = inputs.to(model_device)
            except Exception:
                pass

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=512)

        # Normalize generate outputs across model variants.
        if isinstance(generated, tuple):
            generated_ids = generated[0]
        elif hasattr(generated, "sequences"):
            generated_ids = generated.sequences
        else:
            generated_ids = generated

        # Strip prompt tokens when possible.
        prompt_len = 0
        if isinstance(inputs, dict) and "input_ids" in inputs:
            prompt_len = int(inputs["input_ids"].size(1))

        if prompt_len and getattr(generated_ids, "ndim", 0) == 2 and generated_ids.size(1) > prompt_len:
            generated_ids = generated_ids[:, prompt_len:]

        plain = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {
            "text": plain,
            "dialogue": f"SPEAKER_00: {plain}" if plain else "",
        }
