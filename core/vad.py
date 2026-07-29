"""
Модуль VAD (Voice Activity Detection).

Используем Silero VAD из локального vendors/silero-vad.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, List, Callable, Tuple

import torch
import torchaudio

from .config import VadConfig
from .logger import logger
from .models import SpeechSegment


class VadError(Exception):
    pass


class VadEngine:
    """Движок VAD на базе Silero VAD."""

    _cached_model = None
    _cached_utils = None

    def __init__(self, config: Optional[VadConfig] = None):
        self.config = config or VadConfig()
        self._model = None
        self._utils = None
        self.device = torch.device("cpu")
        self.local_repo = Path(__file__).parent.parent / "vendors" / "silero-vad"
        logger.info(f"VadEngine инициализирован. enabled={self.config.enabled}")

    def _get_audio_info(self, wav_path: str) -> Tuple[int, int, float]:
        """Получает sample_rate, num_frames и duration через ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(wav_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            sample_rate = 16000
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sample_rate = int(stream.get("sample_rate", 16000))
                    break

            duration = float(data.get("format", {}).get("duration", 0))
            if duration <= 0:
                raise VadError(f"Не удалось определить длительность: {wav_path}")

            num_frames = int(duration * sample_rate)
            return sample_rate, num_frames, duration
        except subprocess.CalledProcessError as e:
            raise VadError(f"Ошибка ffprobe: {e}")

    def _load_model(self) -> None:
        """Ленивая загрузка Silero VAD."""
        if VadEngine._cached_model is not None:
            self._model = VadEngine._cached_model
            self._utils = VadEngine._cached_utils
            logger.debug("Silero VAD взят из кэша")
            return

        if not self.local_repo.exists():
            raise VadError(
                f"Репозиторий Silero VAD не найден: {self.local_repo}"
            )

        logger.info(f"Загрузка Silero VAD из {self.local_repo}...")

        try:
            model, utils = torch.hub.load(
                repo_or_dir=str(self.local_repo),
                model="silero_vad",
                source="local",
                trust_repo=True
            )
            model.eval()
            model.to(self.device)
        except Exception as e:
            logger.error(f"Ошибка загрузки Silero VAD: {e}")
            raise VadError(f"Не удалось загрузить Silero VAD: {e}")

        VadEngine._cached_model = model
        VadEngine._cached_utils = utils
        self._model = model
        self._utils = utils

        logger.info("Silero VAD загружен")

    def detect(
        self,
        wav_path: str,
        cancel_token: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[SpeechSegment]:
        """Находит участки речи в аудио."""
        if not self.config.enabled:
            _, _, duration = self._get_audio_info(str(wav_path))
            return [SpeechSegment(start=0.0, end=duration)]

        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise VadError(f"WAV не найден: {wav_path}")

        if self._model is None:
            self._load_model()

        if progress_callback:
            progress_callback(0, "VAD: анализ аудио...")

        if cancel_token and cancel_token():
            raise VadError("Операция отменена")

        try:
            sample_rate, num_frames, duration = self._get_audio_info(str(wav_path))
            get_speech_timestamps = self._utils[0]

            # Короткий файл — целиком в память
            if duration < 60:
                waveform, sr = torchaudio.load(str(wav_path))
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze()
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000

                speech_timestamps = get_speech_timestamps(
                    waveform, self._model, sampling_rate=sr,
                    threshold=self.config.speech_threshold,
                    min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
                    min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
                )

                segments = [
                    SpeechSegment(start=ts["start"] / sr, end=ts["end"] / sr)
                    for ts in speech_timestamps
                ]

                if not segments:
                    segments = [SpeechSegment(start=0.0, end=duration)]

                speech_duration = sum(s.duration for s in segments)
                logger.info(
                    f"VAD: {len(segments)} участков, "
                    f"{speech_duration:.1f}с из {duration:.1f}с"
                )

                if progress_callback:
                    progress_callback(100, f"VAD: {len(segments)} участков речи")

                return segments

            # Длинный файл — стриминг
            logger.info(f"Длинный файл ({duration:.0f}с), стриминг VAD")

            chunk_duration = 30.0
            chunk_frames = int(chunk_duration * sample_rate)
            overlap_frames = int(0.5 * sample_rate)

            all_segments = []
            current_position = 0

            while current_position < num_frames:
                if cancel_token and cancel_token():
                    raise VadError("Операция отменена")

                end_position = min(current_position + chunk_frames, num_frames)
                waveform, sr = torchaudio.load(
                    str(wav_path),
                    frame_offset=current_position,
                    num_frames=end_position - current_position
                )

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze()

                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000

                speech_timestamps = get_speech_timestamps(
                    waveform, self._model, sampling_rate=sr,
                    threshold=self.config.speech_threshold,
                    min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
                    min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
                )

                chunk_offset = current_position / sample_rate
                for ts in speech_timestamps:
                    all_segments.append(SpeechSegment(
                        start=ts["start"] / sr + chunk_offset,
                        end=ts["end"] / sr + chunk_offset
                    ))

                logger.debug(
                    f"VAD стриминг: чанк {int(current_position // chunk_frames + 1)}/"
                    f"{int(num_frames // chunk_frames + 1)} "
                    f"({current_position/sample_rate:.0f}с из {duration:.0f}с)"
                )

                # Если обработали до конца файла — выходим (избегаем бесконечного цикла)
                if end_position >= num_frames:
                    break

                current_position = end_position - overlap_frames

                progress = (current_position / num_frames) * 100
                if progress_callback:
                    progress_callback(progress, f"VAD: {progress:.0f}%")

                del waveform

            # Убираем дубликаты на границах
            if all_segments:
                merged = [all_segments[0]]
                for seg in all_segments[1:]:
                    last = merged[-1]
                    if seg.start - last.end < 0.5:
                        merged[-1] = SpeechSegment(start=last.start, end=seg.end)
                    else:
                        merged.append(seg)
                all_segments = merged

            if not all_segments:
                all_segments = [SpeechSegment(start=0.0, end=duration)]

            speech_duration = sum(s.duration for s in all_segments)
            logger.info(
                f"VAD (стриминг): {len(all_segments)} участков, "
                f"{speech_duration:.1f}с из {duration:.1f}с"
            )

            if progress_callback:
                progress_callback(100, f"VAD: {len(all_segments)} участков речи")

            return all_segments

        except VadError:
            raise
        except Exception as e:
            logger.error(f"Ошибка VAD: {e}", exc_info=True)
            raise VadError(f"Не удалось выполнить VAD: {e}")

    @classmethod
    def clear_cache(cls) -> None:
        cls._cached_model = None
        cls._cached_utils = None
        import gc
        gc.collect()
        logger.info("Кэш VAD очищен")