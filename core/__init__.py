"""
Движок транскрибации аудио и видео.

Основные классы:
- Transcriber: главный класс для работы с движком
- Config: конфигурация всех модулей
- TranscriptResult: результат транскрибации

Пример использования:
    from core import Transcriber
    
    transcriber = Transcriber()
    job = transcriber.transcribe("audio.mp3", output="audio.txt")
    print(job.result.full_text)
"""

from .transcriber import Transcriber, transcribe, TranscriptionJob, TranscriberError
from .config import Config
from .models import TranscriptResult, Utterance, SpeakerSegment

__all__ = [
    'Transcriber',
    'transcribe',
    'TranscriptionJob',
    'TranscriberError',
    'Config',
    'TranscriptResult',
    'Utterance',
    'SpeakerSegment',
]

__version__ = "1.0.0"