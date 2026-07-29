"""
Публичный API движка транскрибации.

Класс Transcriber — это единая точка входа для всех операций:
- Транскрибация одного файла
- Пакетная транскрибация
- Настройка всех параметров через конфиг

Пример использования:
    from core.transcriber import Transcriber
    
    # Простой способ
    transcriber = Transcriber()
    result = transcriber.transcribe("interview.mp3", output="interview.txt")
    
    # С указанием числа спикеров
    result = transcriber.transcribe(
        "meeting.mp3",
        output="meeting",
        format="srt",
        num_speakers=3
    )
    
    # Пакетная обработка
    results = transcriber.transcribe_batch(
        ["file1.mp3", "file2.mp3"],
        output_dir="./transcripts"
    )
"""

from pathlib import Path
from typing import Optional, List, Union, Dict, Callable
from dataclasses import dataclass

from .audio import AudioProcessor
from .merge import Merger
from .export import TranscriptExporter
from .models import TranscriptResult
from .config import (
    Config, AudioConfig, AsrConfig, DiarizationConfig,
    VadConfig, MergeConfig, LongFormConfig, PostprocessConfig, ExportConfig
)
from .logger import logger


class TranscriberError(Exception):
    pass


@dataclass
class TranscriptionJob:
    """Результат одной транскрибации."""
    input_path: Path
    output_path: Optional[Path]
    result: TranscriptResult
    duration: float
    processing_time: float
    
    @property
    def realtime_factor(self) -> float:
        """Во сколько раз быстрее реального времени."""
        return self.duration / self.processing_time if self.processing_time > 0 else 0


class Transcriber:
    """
    Главный класс для транскрибации аудио и видео.
    
    Объединяет все модули в удобный интерфейс:
    - AudioProcessor: конвертация форматов
    - Merger: VAD + диаризация + ASR + слияние
    - TranscriptExporter: экспорт в разные форматы
    
    Использование:
        transcriber = Transcriber()
        result = transcriber.transcribe("audio.mp3")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Инициализирует транскрайбер.
        
        Args:
            config: конфигурация движка (или None для дефолтной)
            hf_token: HuggingFace токен для gated моделей (можно задать через .env)
        """
        self.config = config or Config.default()
        
        # Устанавливаем HF_TOKEN если передан
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token
        
        # Инициализируем модули
        self.audio_processor = AudioProcessor(self.config.audio)
        self.merger = Merger(
            merge_config=self.config.merge,
            asr_config=self.config.asr,
            diarization_config=self.config.diarization,
            audio_config=self.config.audio,
            vad_config=self.config.vad,
            long_form_config=self.config.long_form,
            postprocess_config=self.config.postprocess,
            export_config=self.config.export,
        )
        
        logger.info("Transcriber инициализирован")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
        num_speakers: Optional[int] = None,
        cancel_token: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TranscriptionJob:
        """
        Транскрибирует аудио/видео файл.
        
        Args:
            audio_path: путь к аудио/видео файлу (mp3, wav, m4a, mp4, ...)
            output: путь для сохранения результата (если None — не сохраняется)
            format: формат экспорта (txt, json, srt, vtt, md).
                    Если None — определяется по расширению output
            num_speakers: точное число спикеров (ускоряет и улучшает качество)
            cancel_token: функция, возвращающая True для отмены
            progress_callback: функция (percent, message) для прогресса
        
        Returns:
            TranscriptionJob с результатом
        
        Example:
            # Простое использование
            job = transcriber.transcribe("interview.mp3")
            print(job.result.full_text)
            
            # С экспортом в субтитры
            job = transcriber.transcribe("video.mp4", output="video.srt")
            
            # С указанием числа спикеров
            job = transcriber.transcribe(
                "meeting.mp3",
                output="meeting.txt",
                num_speakers=3
            )
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise TranscriberError(f"Файл не найден: {audio_path}")
        
        import time
        start_time = time.time()
        
        logger.info(f"Транскрибация: {audio_path.name}")
        
        wav_info = None
        try:
            # 1. Конвертируем в WAV (если нужно)
            wav_info = self.audio_processor.convert_to_wav(audio_path)
            wav_path = wav_info.path
            logger.info(f"WAV создан: {wav_path}")
            
            # 2. Если указано число спикеров — обновляем конфиг диаризации
            if num_speakers is not None:
                self.merger.diarization.config.exact_speakers = num_speakers
            
            # 3. Транскрибируем
            result = self.merger.merge(
                str(wav_path),
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )
            
            # 4. Экспортируем (если указан output)
            output_path = None
            if output is not None:
                output_path = self.merger.exporter.export(
                    result,
                    str(output),
                    format=format
                )
                logger.info(f"Результат сохранён: {output_path}")
            
            # 5. Формируем job
            processing_time = time.time() - start_time
            job = TranscriptionJob(
                input_path=audio_path,
                output_path=output_path,
                result=result,
                duration=result.duration,
                processing_time=processing_time,
            )
            
            logger.info(
                f"Транскрибация завершена: "
                f"{result.duration:.1f}с аудио за {processing_time:.1f}с "
                f"({job.realtime_factor:.1f}x realtime)"
            )
            
            return job
        
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}", exc_info=True)
            raise TranscriberError(f"Не удалось транскрибировать {audio_path}: {e}")
        
        finally:
            # 6. Очищаем временный WAV-файл
            if wav_info is not None:
                self.audio_processor.cleanup(wav_info)
    
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        format: str = "txt",
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[TranscriptionJob]:
        """
        Транскрибирует несколько файлов.
        
        Args:
            audio_paths: список путей к аудио/видео файлам
            output_dir: директория для сохранения (или None)
            format: формат экспорта
            num_speakers: точное число спикеров
            progress_callback: функция (current, total, message)
        
        Returns:
            Список TranscriptionJob (с ошибками=None для упавших файлов)
        """
        results = []
        total = len(audio_paths)
        
        for i, audio_path in enumerate(audio_paths, 1):
            audio_path = Path(audio_path)
            logger.info(f"[{i}/{total}] Обработка: {audio_path.name}")
            
            if progress_callback:
                progress_callback(i, total, f"Обработка {audio_path.name}")
            
            # Определяем путь сохранения
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{audio_path.stem}.{format}"
            
            try:
                job = self.transcribe(
                    audio_path,
                    output=output_path,
                    format=format,
                    num_speakers=num_speakers,
                )
                results.append(job)
            except TranscriberError as e:
                logger.error(f"Пропуск {audio_path.name}: {e}")
                results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch завершён: {successful}/{total} успешно")
        
        return results
    
    def unload(self) -> None:
        """Выгружает все модели из памяти."""
        self.merger.unload()
        logger.info("Все модели выгружены")
    
    @classmethod
    def clear_all_caches(cls) -> None:
        """Очищает все кэши моделей."""
        Merger.clear_all_caches()
        logger.info("Все кэши очищены")


# --- Convenience function ---

def transcribe(
    audio_path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
    num_speakers: Optional[int] = None,
) -> TranscriptionJob:
    """
    Функция быстрого доступа для одноразовой транскрибации.
    
    Создаёт Transcriber, выполняет транскрибацию и выгружает модели.
    
    Example:
        from core.transcriber import transcribe
        
        job = transcribe("interview.mp3", output="interview.txt")
        print(job.result.full_text)
    """
    transcriber = Transcriber()
    try:
        return transcriber.transcribe(
            audio_path,
            output=output,
            format=format,
            num_speakers=num_speakers,
        )
    finally:
        transcriber.unload()
