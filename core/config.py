"""
Конфигурация движка.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    temp_dir: str = "./tmp"
    supported_formats: list = field(default_factory=lambda: [
        ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".ogg", ".flac",
        ".aac", ".wma", ".webm", ".mkv"
    ])


@dataclass
class AsrConfig:
    model_name: str = "v3_e2e_rnnt"
    device: str = "auto"
    batch_size: int = 1
    use_longform: bool = True


@dataclass
class DiarizationConfig:
    """Настройки диаризации."""
    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"  # Streaming v2 вместо v1
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    exact_speakers: Optional[int] = None


@dataclass
class VadConfig:
    """Настройки VAD (определение участков с речью)."""
    enabled: bool = True
    # Минимальная длительность участка речи (сек)
    min_speech_duration: float = 0.25
    # Минимальная пауза между участками (сек)
    min_silence_duration: float = 0.1
    # Порог вероятности речи (0-1)
    speech_threshold: float = 0.5

@dataclass
class PostprocessConfig:
    """Настройки постобработки текста."""
    enabled: bool = True
    remove_fillers: bool = True          # удалять "эээ", "ммм" и т.д.
    remove_stuttering: bool = True       # удалять "я я я" -> "я"
    normalize_punctuation: bool = True   # чистить "!!!", "???"
    normalize_spaces: bool = True        # чистить двойные пробелы
    capitalize_sentences: bool = True    # заглавная в начале предложения

@dataclass
class MergeConfig:
    merge_gap_threshold: float = 1.0
    min_segment_duration: float = 0.3
    speaker_prefix: str = "Спикер"


@dataclass
class LongFormConfig:
    """Настройки обработки длинных записей."""
    # Размер чанка в секундах (10 минут = 600 сек)
    chunk_size: int = 600
    # Перекрытие между чанками в секундах (чтобы не резать реплики)
    chunk_overlap: int = 5
    # Использовать VAD
    use_vad: bool = True


@dataclass
class ExportConfig:
    default_format: str = "txt"
    formats: list = field(default_factory=lambda: ["txt", "json", "srt", "vtt", "md"])


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    long_form: LongFormConfig = field(default_factory=LongFormConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)  # 🆕
    export: ExportConfig = field(default_factory=ExportConfig)
    language: str = "ru"

    @classmethod
    def default(cls) -> "Config":
        return cls()