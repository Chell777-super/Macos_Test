"""
Модуль диаризации на базе NVIDIA Sortformer.

Sortformer — end-to-end модель от NVIDIA, работает в 30-60 раз быстрее pyannote.
Для длинных записей использует чанкинг по 90 секунд (ограничение модели).
"""

import gc
import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional, List, Callable, Tuple

import torchaudio

from .config import DiarizationConfig
from .logger import logger
from .models import SpeakerSegment, DiarizationResult

class DiarizationError(Exception):
    pass


class DiarizationEngine:
    """
    Движок диаризации на базе NVIDIA Sortformer.
    
    Преимущества:
    - 30-60x realtime (против 24x у pyannote)
    - End-to-end модель (один проход)
    - Качество на уровне pyannote
    """

    _cached_model = None
    _cached_model_name: Optional[str] = None

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self._model = None
        # Sortformer работает чанками по 90 секунд максимум
        self.max_chunk_duration = 90.0  # секунд
        logger.info(f"DiarizationEngine (Sortformer) инициализирован. model={self.config.model_name}")

    def _get_audio_info(self, wav_path: str) -> Tuple[int, int, float]:
        """
        Получает sample_rate, num_frames и duration через ffprobe.
        Не загружает файл в память.
        
        Returns:
            (sample_rate, num_frames, duration)
        """
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
                raise DiarizationError(f"Не удалось определить длительность: {wav_path}")

            num_frames = int(duration * sample_rate)
            return sample_rate, num_frames, duration
        except subprocess.CalledProcessError as e:
            raise DiarizationError(f"Ошибка ffprobe: {e}")

    def _load_model(self) -> None:
        """Ленивая загрузка модели Sortformer."""
        if (DiarizationEngine._cached_model is not None and
                DiarizationEngine._cached_model_name == self.config.model_name):
            self._model = DiarizationEngine._cached_model
            logger.debug(f"Sortformer {self.config.model_name} взят из кэша")
            return

        logger.info(f"Загрузка Sortformer: {self.config.model_name}...")

        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            model = SortformerEncLabelModel.from_pretrained(self.config.model_name)
            model.eval()
        except Exception as e:
            logger.error(f"Ошибка загрузки Sortformer: {e}")
            raise DiarizationError(f"Не удалось загрузить {self.config.model_name}: {e}")

        DiarizationEngine._cached_model = model
        DiarizationEngine._cached_model_name = self.config.model_name
        self._model = model

        logger.info(f"Sortformer {self.config.model_name} загружен")

    def _parse_rttm(self, rttm_lines: List[str], offset: float = 0.0) -> List[SpeakerSegment]:
        """
        Парсит RTTM-строки в SpeakerSegment.
        
        Формат RTTM: "start end speaker_label"
        
        Args:
            rttm_lines: список RTTM-строк
            offset: смещение в глобальное время (для чанкинга)
        """
        segments = []
        for line in rttm_lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0]) + offset
                end = float(parts[1]) + offset
                speaker = parts[2]
                if end > start:
                    segments.append(SpeakerSegment(start=start, end=end, speaker=speaker))
            except (ValueError, IndexError):
                continue
        return segments

    def _merge_adjacent_segments(
        self,
        segments: List[SpeakerSegment],
        threshold: float = 0.5
    ) -> List[SpeakerSegment]:
        """Склеивает соседние сегменты одного спикера."""
        if not segments:
            return []

        segments = sorted(segments, key=lambda s: s.start)

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start - last.end

            if seg.speaker == last.speaker and gap <= threshold:
                merged[-1] = SpeakerSegment(
                    start=last.start,
                    end=seg.end,
                    speaker=last.speaker
                )
            else:
                merged.append(seg)

        return merged

    def diarize(
        self,
        wav_path: str,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> DiarizationResult:
        """
        Выполняет диаризацию через Sortformer.
        
        Для длинных файлов (> 90 сек) использует чанкинг с перекрытием.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise DiarizationError(f"WAV не найден: {wav_path}")

        if cancel_token and cancel_token():
            raise DiarizationError("Операция отменена")

        if self._model is None:
            self._load_model()

        logger.info(f"Диаризация (Sortformer): {wav_path.name}")

        try:
            # Получаем информацию через ffprobe
            sample_rate, num_frames, duration = self._get_audio_info(str(wav_path))

            if duration <= self.max_chunk_duration:
                # Короткий файл — одна диаризация
                logger.info(f"Короткий файл ({duration:.1f}с), одна диаризация")
                results = self._model.diarize([str(wav_path)])

                if not results or not results[0]:
                    return DiarizationResult(segments=[], num_speakers=0, duration=duration)

                segments = self._parse_rttm(results[0])

            else:
                # Длинный файл — чанкинг
                logger.info(
                    f"Длинный файл ({duration:.1f}с), "
                    f"чанкинг по {self.max_chunk_duration}с"
                )

                all_segments = []
                chunk_duration = self.max_chunk_duration
                overlap = 2.0  # 2 секунды перекрытия
                chunk_start = 0.0

                chunk_idx = 0
                while chunk_start < duration:
                    if cancel_token and cancel_token():
                        raise DiarizationError("Операция отменена")

                    chunk_end = min(chunk_start + chunk_duration, duration)
                    chunk_idx += 1

                    logger.info(
                        f"  Чанк {chunk_idx}: {chunk_start:.1f}с - {chunk_end:.1f}с "
                        f"({chunk_end - chunk_start:.1f}с)"
                    )

                    # Вырезаем чанк во временный файл
                    temp_dir = Path("./tmp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_wav = temp_dir / f"chunk_{uuid.uuid4().hex[:8]}.wav"

                    chunk_start_frame = int(chunk_start * sample_rate)
                    chunk_end_frame = min(int(chunk_end * sample_rate), num_frames)

                    waveform, sr = torchaudio.load(
                        str(wav_path),
                        frame_offset=chunk_start_frame,
                        num_frames=chunk_end_frame - chunk_start_frame
                    )
                    torchaudio.save(str(temp_wav), waveform, sr)

                    # Освобождаем память
                    del waveform

                    try:
                        results = self._model.diarize([str(temp_wav)])

                        if results and results[0]:
                            chunk_segments = self._parse_rttm(results[0], offset=chunk_start)
                            all_segments.extend(chunk_segments)

                    finally:
                        try:
                            temp_wav.unlink(missing_ok=True)
                        except Exception:
                            pass

                    chunk_start = chunk_end - overlap
                    if chunk_end >= duration:
                        break

                segments = self._merge_adjacent_segments(all_segments, threshold=overlap)

            num_speakers = len(set(s.speaker for s in segments)) if segments else 0

            logger.info(
                f"Диаризация завершена: {len(segments)} сегментов, "
                f"{num_speakers} спикеров, {duration:.2f}с аудио"
            )

            return DiarizationResult(
                segments=segments,
                num_speakers=num_speakers,
                duration=duration
            )

        except DiarizationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}", exc_info=True)
            raise DiarizationError(f"Не удалось выполнить диаризацию: {e}")

    def unload(self) -> None:
        """Выгружает модель."""
        self._model = None
        logger.info("Модель выгружена из экземпляра")

    @classmethod
    def clear_cache(cls) -> None:
        """Очищает кэш модели."""
        cls._cached_model = None
        cls._cached_model_name = None
        gc.collect()
        logger.info("Кэш модели Sortformer очищен")