#backends/base.py

from abc import ABC, abstractmethod

class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_file):
        """
        Expects: path to audio file
        Return: {'text': str, 'dialogue':str}
        """
        pass