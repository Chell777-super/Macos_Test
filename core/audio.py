"""
Модуль обработки аудио.

Отвечает за:
- конвертацию любых аудио/видео в 16 kHz mono WAV
- получение длительности файла
- проверку валидности файла
- работу с временными файлами
"""

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .config import AudioConfig
from .logger import logger


# --- Исключения --------------------------------------------------------------

class AudioError(Exception):
    """Базовое исключение для всех ошибок аудио-модуля."""
    pass


class UnsupportedFormatError(AudioError):
    """Формат файла не поддерживается."""
    pass


class CorruptedFileError(AudioError):
    """Файл повреждён или не содержит аудиодорожку."""
    pass


class FFmpegNotFoundError(AudioError):
    """ffmpeg или ffprobe не установлены или не найдены в PATH."""
    pass


# --- Структуры данных --------------------------------------------------------

@dataclass
class WavInfo:
    """Информация об обработанном WAV-файле."""
    path: Path              # путь к WAV-файлу
    duration: float         # длительность в секундах
    sample_rate: int        # частота дискретизации
    channels: int           # количество каналов
    original_path: Path     # путь к исходному файлу


# --- Основной класс ----------------------------------------------------------

class AudioProcessor:
    """
    Обработчик аудио-файлов.

    Конвертирует любые входные форматы в 16 kHz mono WAV,
    который нужен ASR-моделям.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._check_ffmpeg()

        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Список созданных временных файлов для последующей очистки
        self._temp_files: list = []

        logger.info(f"AudioProcessor инициализирован. temp_dir={self.temp_dir}")

    def _check_ffmpeg(self) -> None:
        """Проверяет, что ffmpeg и ffprobe установлены."""
        for tool in ["ffmpeg", "ffprobe"]:
            try:
                subprocess.run(
                    [tool, "-version"],
                    capture_output=True,
                    check=True
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                raise FFmpegNotFoundError(
                    f"{tool} не найден. Установи через 'brew install ffmpeg'"
                )

    def _run_ffprobe(self, path: Path) -> dict:
        """
        Запускает ffprobe и возвращает JSON с метаданными.

        Args:
            path: путь к файлу

        Returns:
            Словарь с метаданными файла
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path)
        ]

        logger.debug(f"Запуск ffprobe для: {path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка ffprobe: {e.stderr}")
            raise CorruptedFileError(f"Не удалось прочитать файл: {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Не удалось распарсить JSON от ffprobe: {e}")
            raise CorruptedFileError(f"Некорректный ответ ffprobe для {path}")

    def validate_audio(self, path: Union[str, Path]) -> dict:
        """
        Проверяет, что файл валидный и содержит аудиодорожку.

        Args:
            path: путь к файлу

        Returns:
            Словарь с метаданными файла

        Raises:
            UnsupportedFormatError: формат не поддерживается
            CorruptedFileError: файл битый
        """
        path = Path(path)

        if not path.exists():
            raise CorruptedFileError(f"Файл не найден: {path}")

        if not path.is_file():
            raise CorruptedFileError(f"Не файл: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.config.supported_formats:
            raise UnsupportedFormatError(
                f"Формат {suffix} не поддерживается. "
                f"Допустимые: {', '.join(self.config.supported_formats)}"
            )

        info = self._run_ffprobe(path)

        streams = info.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        if not audio_streams:
            raise CorruptedFileError(f"Файл не содержит аудиодорожки: {path}")

        duration = float(info.get("format", {}).get("duration", 0))
        if duration <= 0:
            raise CorruptedFileError(f"Не удалось определить длительность: {path}")

        logger.info(f"Файл валиден: {path.name}, длительность={duration:.2f}с")
        return info

    def get_duration(self, path: Union[str, Path]) -> float:
        """
        Возвращает длительность файла в секундах.

        Args:
            path: путь к файлу

        Returns:
            Длительность в секундах
        """
        path = Path(path)
        info = self._run_ffprobe(path)
        duration = float(info.get("format", {}).get("duration", 0))

        if duration <= 0:
            raise CorruptedFileError(f"Не удалось определить длительность: {path}")

        return duration

    def convert_to_wav(self, source: Union[str, Path]) -> WavInfo:
        """
        Конвертирует любой аудио/видео в 16 kHz mono WAV.

        Args:
            source: путь к исходному файлу

        Returns:
            WavInfo с информацией о сконвертированном файле
        """
        source = Path(source)
        logger.info(f"Начинаем конвертацию: {source.name}")

        # Сначала проверяем файл
        self.validate_audio(source)

        # Создаём уникальное имя для временного файла
        unique_id = uuid.uuid4().hex[:8]
        output_path = self.temp_dir / f"{source.stem}_{unique_id}.wav"

        # Команда ffmpeg:
        # -y         : перезаписывать без вопросов
        # -i         : входной файл
        # -vn        : без видео
        # -ac 1      : mono
        # -ar 16000  : 16 kHz
        # -acodec pcm_s16le : 16-bit PCM
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(source),
            "-vn",
            "-ac", str(self.config.channels),
            "-ar", str(self.config.sample_rate),
            "-acodec", "pcm_s16le",
            str(output_path)
        ]

        logger.debug(f"Запуск: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации: {e.stderr}")
            raise AudioError(f"Не удалось конвертировать файл: {e.stderr}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise AudioError(f"Конвертация не создала выходной файл")

        # Запоминаем временный файл для последующей очистки
        self._temp_files.append(output_path)

        # Получаем фактическую длительность выходного файла
        duration = self.get_duration(output_path)

        logger.info(f"Конвертация завершена: {output_path} ({duration:.2f}с)")

        return WavInfo(
            path=output_path,
            duration=duration,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            original_path=source
        )

    def cleanup(self, wav_info: Optional[WavInfo] = None) -> None:
        """
        Удаляет временные файлы.

        Args:
            wav_info: если задан — удаляет только этот файл,
                     иначе удаляет все временные файлы
        """
        if wav_info is not None:
            if wav_info.path.exists():
                try:
                    wav_info.path.unlink()
                    logger.debug(f"Удалён временный файл: {wav_info.path}")
                except OSError as e:
                    logger.warning(f"Не удалось удалить {wav_info.path}: {e}")

            if wav_info.path in self._temp_files:
                self._temp_files.remove(wav_info.path)
        else:
            for path in self._temp_files:
                if path.exists():
                    try:
                        path.unlink()
                        logger.debug(f"Удалён временный файл: {path}")
                    except OSError as e:
                        logger.warning(f"Не удалось удалить {path}: {e}")
            self._temp_files.clear()
            logger.info("Очистка временных файлов завершена")

    def __enter__(self):
        """Поддержка контекстного менеджера (with AudioProcessor() as ap: ...)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Автоматическая очистка при выходе из with-блока."""
        self.cleanup()
