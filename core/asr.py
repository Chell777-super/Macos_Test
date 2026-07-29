"""
Модуль распознавания речи (ASR).

Использует модель GigaAM V3 e2e_rnnt для транскрибации русского аудио.
Поддерживает MPS (Apple Silicon) с автоматическим fallback на CPU.

Ключевые особенности:
- Ленивая загрузка модели (экономит память)
- Кэширование модели между вызовами
- Корректная работа с длинными файлами через transcribe_longform
- Word-level тайминги для точного сопоставления с диаризацией
"""

import os
import torch
import gigaam
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable

from .config import AsrConfig
from .logger import logger


# --- Исключения --------------------------------------------------------------

class AsrError(Exception):
    """Базовое исключение для ошибок ASR-модуля."""
    pass


class ModelLoadError(AsrError):
    """Не удалось загрузить модель."""
    pass


class TranscriptionError(AsrError):
    """Ошибка при распознавании."""
    pass


# --- Структуры данных --------------------------------------------------------

@dataclass
class WordTiming:
    """Информация о тайминге одного слова."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """
    Результат распознавания аудио.

    Содержит полный текст и список слов с таймингами.
    """
    text: str
    words: List[WordTiming] = None
    duration: float = 0.0

    def __post_init__(self):
        if self.words is None:
            self.words = []


# --- Основной класс ----------------------------------------------------------

class AsrEngine:
    """
    Движок распознавания речи.

    Использует GigaAM V3 для транскрибации.
    Модель загружается лениво и кэшируется.
    """

    _cached_model = None
    _cached_model_name: Optional[str] = None

    def __init__(self, config: Optional[AsrConfig] = None):
        self.config = config or AsrConfig()
        self._model = None
        self.device = self._select_device()
        logger.info(f"AsrEngine инициализирован. device={self.device}, model={self.config.model_name}")

    def _select_device(self) -> str:
        """Выбирает устройство (MPS или CPU)."""
        if self.config.device == "cpu":
            return "cpu"

        if self.config.device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning("MPS запрошен, но недоступен. Используем CPU.")
                return "cpu"

        # auto
        if torch.backends.mps.is_available():
            logger.info("Автоматически выбран MPS")
            return "mps"
        else:
            logger.info("MPS недоступен, используется CPU")
            return "cpu"

    def _load_model(self) -> None:
        """Ленивая загрузка модели с кэшированием."""
        if (AsrEngine._cached_model is not None and
                AsrEngine._cached_model_name == self.config.model_name):
            self._model = AsrEngine._cached_model
            logger.debug(f"Модель {self.config.model_name} взята из кэша")
            return

        logger.info(f"Загрузка модели GigaAM: {self.config.model_name}...")

        try:
            model = gigaam.load_model(self.config.model_name)
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise ModelLoadError(f"Не удалось загрузить модель {self.config.model_name}: {e}")

        AsrEngine._cached_model = model
        AsrEngine._cached_model_name = self.config.model_name
        self._model = model

        logger.info(f"Модель {self.config.model_name} загружена")

    def transcribe(
        self,
        wav_path: str,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> TranscriptionResult:
        """
        Распознаёт аудио из WAV-файла.

        Args:
            wav_path: путь к WAV-файлу
            cancel_token: функция отмены

        Returns:
            TranscriptionResult с текстом и таймингами
        """
        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise TranscriptionError(f"WAV-файл не найден: {wav_path}")

        if cancel_token and cancel_token():
            logger.info("Распознавание отменено")
            raise TranscriptionError("Операция отменена")

        if self._model is None:
            self._load_model()

        logger.info(f"Начинаем распознавание: {wav_path.name}")

        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(wav_path))
            duration = waveform.shape[1] / sample_rate

            if cancel_token and cancel_token():
                raise TranscriptionError("Операция отменена")

            # Решаем, какой метод использовать
            use_longform = self.config.use_longform and duration > 30

            if use_longform:
                logger.info(f"Используем transcribe_longform (длительность {duration:.1f}с)")
                
                # Устанавливаем HF_TOKEN для longform (нужен для pyannote внутри)
                from dotenv import load_dotenv
                load_dotenv()
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token

                result = self._model.transcribe_longform(str(wav_path))

                # Извлекаем текст и тайминги из longform результата
                if hasattr(result, 'text'):
                    text = result.text
                elif hasattr(result, '__iter__') and not isinstance(result, str):
                    # Если результат — список сегментов
                    text_parts = []
                    for segment in result:
                        if hasattr(segment, 'text'):
                            text_parts.append(segment.text)
                        elif isinstance(segment, dict) and 'transcription' in segment:
                            text_parts.append(segment['transcription'])
                        else:
                            text_parts.append(str(segment))
                    text = " ".join(text_parts)
                else:
                    text = str(result)

                # Пытаемся извлечь тайминги
                words = []
                if hasattr(result, 'words') and result.words:
                    for word_info in result.words:
                        word_text = getattr(word_info, 'text', 
                                          getattr(word_info, 'word', ''))
                        start = float(getattr(word_info, 'start', 0.0))
                        end = float(getattr(word_info, 'end', 0.0))
                        if word_text:
                            words.append(WordTiming(word=word_text, start=start, end=end))

            else:
                logger.info(f"Используем transcribe (длительность {duration:.1f}с)")
                result = self._model.transcribe(str(wav_path))

                # Извлекаем текст
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)

                # Извлекаем тайминги слов
                words = []
                if hasattr(result, 'words') and result.words:
                    for word_info in result.words:
                        # Пробуем разные варианты атрибутов
                        word_text = getattr(word_info, 'text',
                                          getattr(word_info, 'word',
                                                 getattr(word_info, 'token', '')))
                        start = float(getattr(word_info, 'start',
                                            getattr(word_info, 'begin', 0.0)))
                        end = float(getattr(word_info, 'end',
                                          getattr(word_info, 'stop', 0.0)))
                        
                        if word_text:
                            words.append(WordTiming(word=word_text, start=start, end=end))

            logger.info(
                f"Распознавание завершено: {len(text)} символов, "
                f"{len(words)} слов, {duration:.2f}с аудио"
            )

            return TranscriptionResult(
                text=text,
                words=words,
                duration=duration
            )

        except TranscriptionError:
            raise
        except Exception as e:
            logger.error(f"Ошибка при распознавании: {e}", exc_info=True)
            raise TranscriptionError(f"Не удалось распознать аудио: {e}")

    def unload(self) -> None:
        """Выгружает модель из экземпляра."""
        self._model = None
        logger.info("Модель выгружена из экземпляра")

    @classmethod
    def clear_cache(cls) -> None:
        """Очищает классовый кэш модели."""
        cls._cached_model = None
        cls._cached_model_name = None
        import gc
        gc.collect()
        logger.info("Классовый кэш модели очищен")