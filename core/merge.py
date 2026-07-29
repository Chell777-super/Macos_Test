"""
Модуль слияния ASR + диаризация + VAD.

Поддерживает длинные записи через стриминг и чанкинг.
"""

import gc
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Callable, Tuple

import torch
import torchaudio

from .asr import AsrEngine
from .diarization import DiarizationEngine, SpeakerSegment
from .vad import VadEngine, SpeechSegment
from .config import (
    MergeConfig, AsrConfig, DiarizationConfig, AudioConfig,
    VadConfig, LongFormConfig, PostprocessConfig, ExportConfig
)
from .logger import logger
from .postprocess import TextPostprocessor
from .export import TranscriptExporter
from .models import (
    Utterance, TranscriptResult,
    SpeakerSegment, SpeechSegment
)


class MergeError(Exception):
    pass


class Merger:
    """
    Сливающий модуль.

    Координирует VAD, ASR и диаризацию.
    """

    def __init__(
        self,
        merge_config: Optional[MergeConfig] = None,
        asr_config: Optional[AsrConfig] = None,
        diarization_config: Optional[DiarizationConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        vad_config: Optional[VadConfig] = None,
        long_form_config: Optional[LongFormConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
        export_config: Optional[ExportConfig] = None,
    ):
        self.merge_config = merge_config or MergeConfig()
        self.asr = AsrEngine(asr_config or AsrConfig())
        self.diarization = DiarizationEngine(diarization_config or DiarizationConfig())
        self.audio_config = audio_config or AudioConfig()
        self.vad = VadEngine(vad_config or VadConfig())
        self.long_form_config = long_form_config or LongFormConfig()
        self.postprocess = TextPostprocessor(postprocess_config or PostprocessConfig())
        self.exporter = TranscriptExporter(export_config or ExportConfig())
        logger.info("Merger инициализирован")
    def _get_audio_info(self, wav_path: str) -> Tuple[int, int, float]:
        """Получает sample_rate, num_frames, duration через ffprobe."""
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
                raise MergeError(f"Не удалось определить длительность: {wav_path}")

            num_frames = int(duration * sample_rate)
            return sample_rate, num_frames, duration
        except subprocess.CalledProcessError as e:
            raise MergeError(f"Ошибка ffprobe: {e}")

    def _rename_speakers(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """SPEAKER_00 -> 'Спикер 1' и т.д."""
        mapping = {}
        counter = 1
        for seg in segments:
            if seg.speaker not in mapping:
                mapping[seg.speaker] = f"{self.merge_config.speaker_prefix} {counter}"
                counter += 1

        renamed = [
            SpeakerSegment(
                start=seg.start,
                end=seg.end,
                speaker=mapping[seg.speaker]
            )
            for seg in segments
        ]
        logger.info(f"Переименование спикеров: {mapping}")
        return renamed

    def _merge_adjacent(self, utterances: List[Utterance]) -> List[Utterance]:
        """Склеивает соседние реплики одного спикера."""
        if not utterances:
            return []

        merged = [utterances[0]]
        threshold = self.merge_config.merge_gap_threshold

        for utt in utterances[1:]:
            last = merged[-1]
            gap = utt.start - last.end

            if utt.speaker == last.speaker and gap <= threshold:
                merged[-1] = Utterance(
                    speaker=last.speaker,
                    start=last.start,
                    end=utt.end,
                    text=f"{last.text} {utt.text}".strip()
                )
            else:
                merged.append(utt)

        return merged

    def _extract_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        start: float,
        end: float
    ) -> Path:
        """Вырезает сегмент аудио во временный WAV."""
        import uuid
        temp_dir = Path(self.audio_config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        start_sample = max(0, int(start * sample_rate))
        end_sample = min(waveform.shape[1], int(end * sample_rate))

        if end_sample <= start_sample:
            raise MergeError(f"Пустой сегмент: {start}-{end}")

        segment_waveform = waveform[:, start_sample:end_sample]
        temp_path = temp_dir / f"seg_{uuid.uuid4().hex[:8]}.wav"
        torchaudio.save(str(temp_path), segment_waveform, sample_rate)

        return temp_path

    def _process_segment_group(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        speech_segments: List[SpeechSegment],
        global_offset: float,
        cancel_token: Optional[Callable[[], bool]],
        progress_callback: Optional[Callable[[float, str], None]],
        progress_start: float,
        progress_end: float,
    ) -> List[Utterance]:
        """Обрабатывает группу сегментов речи."""
        utterances = []

        if not speech_segments:
            return utterances

        min_start = min(s.start for s in speech_segments)
        max_end = max(s.end for s in speech_segments)

        logger.debug(f"Обработка группы: {min_start:.1f}с - {max_end:.1f}с ({max_end-min_start:.1f}с)")

        try:
            temp_wav = self._extract_segment(waveform, sample_rate, min_start, max_end)
        except MergeError:
            return utterances

        logger.debug("Запускаем диаризацию...")
        try:
            diar_result = self.diarization.diarize(str(temp_wav), cancel_token=cancel_token)
            logger.debug(f"Диаризация завершена: {len(diar_result.segments)} сегментов")
            segments = self._rename_speakers(diar_result.segments)
            for seg in segments:
                seg.start += min_start
                seg.end += min_start
        except Exception as e:
            logger.warning(f"Диаризация не удалась: {e}")
            segments = [SpeakerSegment(
                start=min_start, end=max_end,
                speaker=f"{self.merge_config.speaker_prefix} 1"
            )]
        finally:
            try:
                temp_wav.unlink(missing_ok=True)
            except Exception:
                pass

        segments = [s for s in segments if s.duration >= self.merge_config.min_segment_duration]
        
        # 🆕 РАЗБИВКА ДЛИННЫХ СЕГМЕНТОВ
        # GigaAM не может обработать > 60 секунд за раз,
        # поэтому режем длинные сегменты на куски по max_asr_segment_duration
        max_asr_duration = 30.0  # секунд
        expanded_segments = []
        for seg in segments:
            if seg.duration <= max_asr_duration:
                expanded_segments.append(seg)
            else:
                # Разбиваем длинный сегмент
                chunk_start = seg.start
                while chunk_start < seg.end:
                    chunk_end = min(chunk_start + max_asr_duration, seg.end)
                    expanded_segments.append(SpeakerSegment(
                        start=chunk_start,
                        end=chunk_end,
                        speaker=seg.speaker
                    ))
                    chunk_start = chunk_end
        
        if len(expanded_segments) > len(segments):
            logger.debug(f"Длинные сегменты разбиты: {len(segments)} -> {len(expanded_segments)}")
        
        segments = expanded_segments
        total_segments = len(segments)

        logger.debug(f"Сегментов для ASR: {total_segments}")

        for i, seg in enumerate(segments):
            if cancel_token and cancel_token():
                raise MergeError("Операция отменена")

            progress = progress_start + (progress_end - progress_start) * (i / total_segments) if total_segments else progress_end
            if progress_callback:
                progress_callback(progress, f"Распознавание: {seg.speaker} ({seg.duration:.1f}с)")

            try:
                temp_wav = self._extract_segment(waveform, sample_rate, seg.start, seg.end)
            except MergeError:
                continue

            try:
                asr_result = self.asr.transcribe(str(temp_wav), cancel_token=cancel_token)
                text = asr_result.text.strip()
            except Exception as e:
                logger.error(f"Ошибка ASR: {e}")
                text = ""
            finally:
                try:
                    temp_wav.unlink(missing_ok=True)
                except Exception:
                    pass

            if text:
                utterances.append(Utterance(
                    speaker=seg.speaker,
                    start=seg.start + global_offset,
                    end=seg.end + global_offset,
                    text=text
                ))

        return utterances
    def merge(
        self,
        wav_path: str,
        cancel_token: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TranscriptResult:
        """Выполняет полное слияние: VAD + диаризация + распознавание."""
        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise MergeError(f"WAV-файл не найден: {wav_path}")

        def report(percent: float, message: str) -> None:
            logger.info(f"[{percent:.0f}%] {message}")
            if progress_callback:
                try:
                    progress_callback(percent, message)
                except Exception:
                    pass

        try:
            # 1. Метаданные через ffprobe (без загрузки в память)
            report(0, "Чтение метаданных аудио...")
            sample_rate, num_frames, duration = self._get_audio_info(str(wav_path))

            if cancel_token and cancel_token():
                raise MergeError("Операция отменена")

            # 2. VAD
            report(2, "VAD: поиск участков речи...")
            speech_segments = self.vad.detect(str(wav_path), cancel_token=cancel_token)
            report(5, f"Найдено {len(speech_segments)} участков речи")
            # 3. Чанкинг
            chunk_size = self.long_form_config.chunk_size
            use_chunking = duration > chunk_size and chunk_size > 0

            if use_chunking:
                report(6, f"Разбиение на чанки по {chunk_size}с...")
                chunks = []
                chunk_start = 0.0
                while chunk_start < duration:
                    chunk_end = min(chunk_start + chunk_size, duration)
                    chunks.append((chunk_start, chunk_end))
                    chunk_start = chunk_end - self.long_form_config.chunk_overlap
                    if chunk_end >= duration:
                        break
                logger.info(f"Создано {len(chunks)} чанков")
            else:
                chunks = [(0.0, duration)]

            # 4. Обрабатываем чанки
            all_utterances = []
            total_chunks = len(chunks)

            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                if cancel_token and cancel_token():
                    raise MergeError("Операция отменена")

                chunk_progress_start = 10 + 85 * (chunk_idx / total_chunks)
                chunk_progress_end = 10 + 85 * ((chunk_idx + 1) / total_chunks)

                if use_chunking:
                    report(
                        chunk_progress_start,
                        f"Чанк {chunk_idx+1}/{total_chunks} ({chunk_start/60:.1f}-{chunk_end/60:.1f} мин)"
                    )

                chunk_speech = [
                    s for s in speech_segments
                    if s.start >= chunk_start and s.end <= chunk_end
                ]

                if not chunk_speech:
                    logger.info(f"Чанк {chunk_idx+1}: нет речи, пропускаем")
                    continue

                chunk_start_frame = int(chunk_start * sample_rate)
                chunk_end_frame = min(int(chunk_end * sample_rate), num_frames)

                chunk_waveform, _ = torchaudio.load(
                    str(wav_path),
                    frame_offset=chunk_start_frame,
                    num_frames=chunk_end_frame - chunk_start_frame
                )

                local_speech = [
                    SpeechSegment(start=s.start - chunk_start, end=s.end - chunk_start)
                    for s in chunk_speech
                ]

                chunk_utterances = self._process_segment_group(
                    waveform=chunk_waveform,
                    sample_rate=sample_rate,
                    speech_segments=local_speech,
                    global_offset=chunk_start,
                    cancel_token=cancel_token,
                    progress_callback=progress_callback,
                    progress_start=chunk_progress_start,
                    progress_end=chunk_progress_end,
                )

                all_utterances.extend(chunk_utterances)

                del chunk_waveform, local_speech, chunk_utterances
                gc.collect()

            # 5. Склейка соседних реплик
            report(98, "Склейка соседних реплик...")
            all_utterances = self._merge_adjacent(all_utterances)

            # 🆕 5.5 Постобработка текста
            report(99, "Постобработка текста...")
            all_utterances = self.postprocess.process_utterances(all_utterances)

            # 6. Финальный результат
            speakers = sorted(set(u.speaker for u in all_utterances))
            full_text = " ".join(u.text for u in all_utterances)

            report(100, f"Готово! {len(all_utterances)} реплик, {len(speakers)} спикеров")

            logger.info(
                f"Слияние: {len(all_utterances)} реплик, "
                f"{len(speakers)} спикеров, {duration:.2f}с"
            )

            return TranscriptResult(
                utterances=all_utterances,
                speakers=speakers,
                full_text=full_text,
                duration=duration
            )

        except MergeError:
            raise
        except Exception as e:
            logger.error(f"Ошибка слияния: {e}", exc_info=True)
            raise MergeError(f"Не удалось выполнить слияние: {e}")

    def unload(self) -> None:
        """Выгружает все модели."""
        self.asr.unload()
        self.diarization.unload()
        logger.info("Все модели выгружены")
    def export(
        self,
        result: TranscriptResult,
        output_path: str,
        format: Optional[str] = None
    ) -> Path:
        """
        Экспортирует результат транскрибации в файл.
        
        Args:
            result: результат от merge()
            output_path: путь для сохранения
            format: формат (txt, json, srt, vtt, md)
        
        Returns:
            Путь к созданному файлу
        """
        return self.exporter.export(result, output_path, format)
    @classmethod
    def clear_all_caches(cls) -> None:
        """Очищает все кэши."""
        AsrEngine.clear_cache()
        DiarizationEngine.clear_cache()
        VadEngine.clear_cache()
        logger.info("Все кэши очищены")