import inspect, getpass, logging
from datetime import datetime
from pathlib import Path

from mne import Report as MneReport

from backends.whisperx import WhisperXBackend
from backends.omni import OmniBackend
from backends.vibevoice import VibeVoiceBackend

from . import paths

from yaml import safe_load
from exca import ConfDict
import yaml
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

class Report(MneReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, fname=None, *args, open_browser=False, overwrite=True, **kwargs):
        if fname is None:
            fname = inspect.stack()[1]
            importer_file = Path(fname.filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{importer_file}_{timestamp}.html"
        super().save(paths.results / fname, *args, open_browser=open_browser, overwrite=overwrite, **kwargs)

def read_config(path):
    with open(path, "r") as f:
        content = f.read()
    return ConfDict(safe_load(content))


def format_timestamped_dialogue(segments: list[dict]) -> str:
    """
    Normalizes diverse ASR outputs into: 
    [start---> end][speaker][content]
    """
    lines = []
    for seg in segments:
        start = seg.get("Start") or seg.get("start") or 0.0
        end = seg.get("End") or seg.get("end") or 0.0
        speaker = seg.get("Speaker") or seg.get("speaker") or "0"
        content = seg.get("Content") or seg.get("text") or ""
        
        # Ensure speaker is normalized (e.g., 0 -> SPEAKER_00)
        spk_label = f"SPEAKER_{int(speaker):02d}"
        
        lines.append(f"[{start:.2f}---> {end:.2f}][{spk_label}][{content.strip()}]")
        
    return "\n".join(lines)


def get_models_per_user():
    user = getpass.getuser()

    if user == "andreasantos":
        log.info("User identified as 'andreasantos' (Mac); initializing smallest models.")
        return [
            WhisperXBackend(model_name="tiny", device="cpu"),
            OmniBackend(model_name="omni-small", device="cpu"),
            VibeVoiceBackend(model_name="microsoft/VibeVoice-ASR", device="cpu")    
        ]
    elif user == "asantos":
        log.info("User identified as 'asantos' (Alienware); initializing full models.")
        return [
            WhisperXBackend(model_name="large-v3", device="cuda"),
            OmniBackend(model_name="omni", device="cuda"),
            VibeVoiceBackend(model_name="microsoft/VibeVoice-ASR", device="cuda")    
        ]
