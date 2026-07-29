"""
Общие структуры данных для движка транскрибации.

Здесь определены dataclass'ы, которые используются несколькими модулями:
- Utterance, TranscriptResult (merge, export)
- SpeakerSegment, DiarizationResult (diarization, merge)
- SpeechSegment (vad, merge)

Вынесение в отдельный модуль предотвращает циклические импорты.
"""

from dataclasses import dataclass
from typing import List


# --- VAD ---

@dataclass
class SpeechSegment:
    """Участок с речью (результат VAD)."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- Диаризация ---

@dataclass
class SpeakerSegment:
    """Сегмент речи одного спикера (результат диаризации)."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Результат диаризации."""
    segments: List[SpeakerSegment]
    num_speakers: int
    duration: float

    def get_speaker_labels(self) -> List[str]:
        return sorted(set(seg.speaker for seg in self.segments))


# --- Транскрипт ---

@dataclass
class Utterance:
    """Одна реплика в транскрипте."""
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_time(self, seconds: float) -> str:
        """Форматирует секунды в HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @property
    def start_str(self) -> str:
        return self.format_time(self.start)

    @property
    def end_str(self) -> str:
        return self.format_time(self.end)


@dataclass
class TranscriptResult:
    """Финальный результат транскрибации."""
    utterances: List[Utterance]
    speakers: List[str]
    full_text: str
    duration: float
