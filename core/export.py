"""
Модуль экспорта транскрипта в разные форматы.

Поддерживаемые форматы:
- TXT: читаемый текст с таймкодами
- JSON: структурированные данные
- SRT: субтитры для видео
- VTT: WebVTT для веб-видео
- MD: Markdown для документации
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import TranscriptResult, Utterance
from .config import ExportConfig
from .logger import logger


class ExportError(Exception):
    pass


class TranscriptExporter:
    """
    Экспортёр транскрипта в различные форматы.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        logger.info(f"TranscriptExporter инициализирован. default_format={self.config.default_format}")

    # --- Форматирование времени ---

    def _format_time_srt(self, seconds: float) -> str:
        """Форматирует время для SRT: HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _format_time_vtt(self, seconds: float) -> str:
        """Форматирует время для VTT: HH:MM:SS.mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _format_time_simple(self, seconds: float) -> str:
        """Форматирует время: HH:MM:SS или MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # --- Форматы экспорта ---

    def to_txt(self, result: TranscriptResult, include_timestamps: bool = True) -> str:
        """
        Экспорт в читаемый текстовый формат.
        
        Пример:
        [00:00:15] Спикер 1: Привет, как дела?
        [00:00:20] Спикер 2: Всё хорошо, спасибо!
        """
        lines = []
        
        if include_timestamps:
            for utt in result.utterances:
                timestamp = self._format_time_simple(utt.start)
                lines.append(f"[{timestamp}] {utt.speaker}: {utt.text}")
        else:
            # Простой текст без таймкодов
            for utt in result.utterances:
                lines.append(f"{utt.speaker}: {utt.text}")
        
        return "\n".join(lines)

    def to_json(self, result: TranscriptResult) -> str:
        """
        Экспорт в JSON формат.
        
        Структура:
        {
            "duration": 1800.0,
            "speakers": ["Спикер 1", "Спикер 2"],
            "utterances": [
                {"speaker": "Спикер 1", "start": 15.0, "end": 20.0, "text": "Привет"},
                ...
            ],
            "full_text": "Привет. Как дела?"
        }
        """
        data = {
            "duration": result.duration,
            "speakers": result.speakers,
            "num_speakers": len(result.speakers),
            "num_utterances": len(result.utterances),
            "utterances": [
                {
                    "speaker": utt.speaker,
                    "start": utt.start,
                    "end": utt.end,
                    "duration": utt.duration,
                    "text": utt.text,
                    "start_formatted": self._format_time_simple(utt.start),
                    "end_formatted": self._format_time_simple(utt.end),
                }
                for utt in result.utterances
            ],
            "full_text": result.full_text,
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2)

    def to_srt(self, result: TranscriptResult, max_chars_per_line: int = 42) -> str:
        """
        Экспорт в формат субтитров SRT.
        
        Пример:
        1
        00:00:15,000 --> 00:00:20,000
        Спикер 1: Привет, как дела?
        
        2
        00:00:20,500 --> 00:00:25,000
        Спикер 2: Всё хорошо, спасибо!
        """
        lines = []
        
        for i, utt in enumerate(result.utterances, 1):
            start = self._format_time_srt(utt.start)
            end = self._format_time_srt(utt.end)
            
            # Добавляем имя спикера к тексту
            text_with_speaker = f"{utt.speaker}: {utt.text}"
            
            # Разбиваем длинные строки (стандарт SRT: ~42 символа)
            text_lines = self._wrap_text(text_with_speaker, max_chars_per_line)
            
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.extend(text_lines)
            lines.append("")  # пустая строка между субтитрами
        
        return "\n".join(lines)

    def to_vtt(self, result: TranscriptResult, max_chars_per_line: int = 42) -> str:
        """
        Экспорт в формат WebVTT.
        
        Пример:
        WEBVTT
        
        00:00:15.000 --> 00:00:20.000
        <v Спикер 1>Привет, как дела?
        
        00:00:20.500 --> 00:00:25.000
        <v Спикер 2>Всё хорошо, спасибо!
        """
        lines = ["WEBVTT", ""]
        
        for utt in result.utterances:
            start = self._format_time_vtt(utt.start)
            end = self._format_time_vtt(utt.end)
            
            # VTT поддерживает теги спикеров: <v Speaker>text
            text_with_speaker = f"<v {utt.speaker}>{utt.text}"
            
            # Разбиваем длинные строки
            text_lines = self._wrap_text(text_with_speaker, max_chars_per_line)
            
            lines.append(f"{start} --> {end}")
            lines.extend(text_lines)
            lines.append("")
        
        return "\n".join(lines)

    def to_md(self, result: TranscriptResult) -> str:
        """
        Экспорт в Markdown формат.
        
        Пример:
        # Транскрипт
        
        **Длительность:** 30:00
        **Спикеров:** 2
        
        ---
        
        **[00:00:15] Спикер 1:** Привет, как дела?
        
        **[00:00:20] Спикер 2:** Всё хорошо, спасибо!
        """
        lines = [
            "# Транскрипт",
            "",
            f"**Длительность:** {self._format_time_simple(result.duration)}",
            f"**Спикеров:** {len(result.speakers)}",
            f"**Реплик:** {len(result.utterances)}",
            "",
            "---",
            "",
        ]
        
        for utt in result.utterances:
            timestamp = self._format_time_simple(utt.start)
            lines.append(f"**[{timestamp}] {utt.speaker}:** {utt.text}")
            lines.append("")
        
        # Добавляем полный текст в конце
        lines.extend([
            "---",
            "",
            "## Полный текст",
            "",
            result.full_text,
        ])
        
        return "\n".join(lines)

    # --- Вспомогательные методы ---

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """
        Разбивает длинный текст на строки по max_chars символов.
        Старается не разрывать слова.
        """
        if len(text) <= max_chars:
            return [text]
        
        lines = []
        words = text.split()
        current_line = []
        current_length = 0
        
        for word in words:
            # +1 для пробела
            if current_length + len(word) + (1 if current_line else 0) > max_chars:
                # Сохраняем текущую строку и начинаем новую
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + (1 if len(current_line) > 1 else 0)
        
        # Добавляем последнюю строку
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines

    # --- Универсальный метод экспорта ---

    def export(
        self,
        result: TranscriptResult,
        output_path: str,
        format: Optional[str] = None
    ) -> Path:
        """
        Экспортирует транскрипт в файл указанного формата.
        
        Args:
            result: результат транскрибации
            output_path: путь для сохранения (расширение добавится автоматически)
            format: формат экспорта (txt, json, srt, vtt, md).
                    Если None — определяется по расширению output_path.
        
        Returns:
            Путь к созданному файлу
        """
        output_path = Path(output_path)
        
        # Определяем формат
        if format is None:
            # Пытаемся определить по расширению
            ext = output_path.suffix.lower().lstrip('.')
            if ext in self.config.formats:
                format = ext
            else:
                format = self.config.default_format
                # Добавляем расширение, если его нет
                if not output_path.suffix:
                    output_path = output_path.with_suffix(f'.{format}')
        else:
            # Формат указан явно — добавляем расширение, если нужно
            if not output_path.suffix:
                output_path = output_path.with_suffix(f'.{format}')
        
        # Проверяем, что формат поддерживается
        if format not in self.config.formats:
            raise ExportError(
                f"Неподдерживаемый формат: {format}. "
                f"Доступны: {self.config.formats}"
            )
        
        # Выбираем метод экспорта
        exporters = {
            'txt': self.to_txt,
            'json': self.to_json,
            'srt': self.to_srt,
            'vtt': self.to_vtt,
            'md': self.to_md,
        }
        
        export_method = exporters[format]
        content = export_method(result)
        
        # Создаём директорию, если нужно
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем файл
        output_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Транскрипт экспортирован: {output_path} ({format})")
        
        return output_path

    def export_all_formats(
        self,
        result: TranscriptResult,
        output_dir: str,
        base_name: str = "transcript"
    ) -> Dict[str, Path]:
        """
        Экспортирует транскрипт во все поддерживаемые форматы.
        
        Args:
            result: результат транскрибации
            output_dir: директория для сохранения
            base_name: базовое имя файлов (без расширения)
        
        Returns:
            Словарь {format: path}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for format in self.config.formats:
            output_path = output_dir / f"{base_name}.{format}"
            try:
                path = self.export(result, str(output_path), format=format)
                results[format] = path
            except Exception as e:
                logger.error(f"Ошибка экспорта в {format}: {e}")
                results[format] = None
        
        logger.info(
            f"Экспорт во все форматы завершён: "
            f"{len([p for p in results.values() if p])}/{len(self.config.formats)} успешно"
        )
        
        return results
