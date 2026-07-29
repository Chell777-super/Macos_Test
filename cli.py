#!/usr/bin/env python3
"""
CLI для движка транскрибации аудио и видео.

Использование:
    python cli.py audio.mp3                        # простая транскрибация
    python cli.py audio.mp3 -o output.txt          # с сохранением в файл
    python cli.py audio.mp3 -o output.srt          # субтитры SRT
    python cli.py audio.mp3 -n 3                   # указать число спикеров
    python cli.py *.mp3 -o ./out/ --format txt     # пакетная обработка
    python cli.py audio.mp3 --no-diarize           # без диаризации (быстрее)
    python cli.py audio.mp3 --no-vad               # без VAD
    python cli.py audio.mp3 -v                     # подробный вывод
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

# Подавляем предупреждения NeMo и torch
warnings.filterwarnings("ignore")
os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from core import Transcriber, Config
from core.config import (
    AsrConfig, DiarizationConfig, VadConfig,
    MergeConfig, LongFormConfig, PostprocessConfig, ExportConfig
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Транскрибация аудио и видео с разделением по спикерам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s audio.mp3                     Транскрибировать и вывести в консоль
  %(prog)s audio.mp3 -o out.txt          Сохранить в текстовый файл
  %(prog)s video.mp4 -o subs.srt         Создать субтитры SRT
  %(prog)s meeting.mp3 -n 4              Указать 4 спикера
  %(prog)s *.wav -o ./out/ --format md   Пакетная обработка в Markdown
  %(prog)s lecture.mp3 --no-diarize      Без диаризации (один спикер)
        """
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Путь к аудио/видео файлу(ам)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (файл или директория)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "json", "srt", "vtt", "md"],
        default=None,
        help="Формат экспорта (по умолчанию определяется по расширению -o)"
    )
    parser.add_argument(
        "-n", "--num-speakers",
        type=int,
        default=None,
        help="Точное число спикеров (ускоряет и улучшает качество)"
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Отключить диаризацию (быстрее для одного спикера)"
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить VAD (обработка всего файла без пропуска тишины)"
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Отключить постобработку текста"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Размер чанка для длинных записей в секундах (по умолчанию: 600)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Тихий режим (только результат)"
    )

    return parser.parse_args()


def print_result(result, quiet: bool = False) -> None:
    """Выводит результат транскрибации в консоль."""
    if quiet:
        print(result.full_text)
        return

    print(f"\n{'='*70}")
    print(f"📝 Транскрипт ({result.duration:.1f}с аудио, {len(result.speakers)} спикеров)")
    print(f"{'='*70}\n")

    for utt in result.utterances:
        timestamp = utt.start_str
        print(f"[{timestamp}] {utt.speaker}: {utt.text}")

    print(f"\n{'='*70}")
    print(f"💬 Реплик: {len(result.utterances)}")
    print(f"👥 Спикеров: {', '.join(result.speakers)}")
    print(f"{'='*70}\n")


def main() -> int:
    args = parse_args()

    # Собираем входные файлы
    input_files: List[Path] = []
    for pattern in args.input:
        p = Path(pattern)
        if p.exists():
            input_files.append(p)
        else:
            # Пробуем как glob-паттерн
            import glob
            matches = glob.glob(pattern)
            if matches:
                input_files.extend(Path(m) for m in matches)
            else:
                print(f"❌ Файл не найден: {pattern}", file=sys.stderr)

    if not input_files:
        print("❌ Нет файлов для обработки", file=sys.stderr)
        return 1

    # Проверяем существующие файлы
    existing = [f for f in input_files if f.exists()]
    if len(existing) != len(input_files):
        missing = [f for f in input_files if not f.exists()]
        for m in missing:
            print(f"⚠️  Пропуск (не найден): {m}", file=sys.stderr)
        input_files = existing

    if not input_files:
        print("❌ Нет доступных файлов", file=sys.stderr)
        return 1

    # Настраиваем конфиг
    config = Config.default()

    if args.no_diarize:
        # Отключаем диаризацию — устанавливаем exact_speakers=1
        config.diarization.exact_speakers = 1

    if args.no_vad:
        config.vad.enabled = False

    if args.no_postprocess:
        config.postprocess.enabled = False

    config.long_form.chunk_size = args.chunk_size

    # Определяем формат вывода
    output_format = args.format
    if output_format is None and args.output:
        output_path = Path(args.output)
        if output_path.suffix:
            ext = output_path.suffix.lstrip('.').lower()
            if ext in ["txt", "json", "srt", "vtt", "md"]:
                output_format = ext

    # Создаём транскрайбер
    if not args.quiet and not args.verbose:
        # В обычном режиме подавляем логи ядра, оставляем только CLI-вывод
        logging.getLogger("core").setLevel(logging.WARNING)
    elif args.quiet:
        logging.getLogger("core").setLevel(logging.CRITICAL)

    if not args.quiet:
        print(f"🔧 Инициализация движка...")

    transcriber = Transcriber(config=config)

    total_start = time.time()
    successful = 0
    failed = 0

    try:
        if len(input_files) == 1:
            # Один файл
            audio_path = input_files[0]

            if not args.quiet:
                print(f"📂 Обработка: {audio_path.name}")

            # Определяем путь сохранения
            output_path = None
            if args.output:
                op = Path(args.output)
                if op.is_dir() or (not op.suffix and len(input_files) > 1):
                    op.mkdir(parents=True, exist_ok=True)
                    fmt = output_format or "txt"
                    output_path = op / f"{audio_path.stem}.{fmt}"
                else:
                    output_path = op

            job = transcriber.transcribe(
                audio_path,
                output=output_path,
                format=output_format,
                num_speakers=args.num_speakers,
            )

            # Выводим результат в консоль (если не сохранён в файл или verbose)
            if not args.quiet:
                if output_path:
                    print(f"✅ Сохранено: {output_path}")
                print(f"⏱️  Время: {job.processing_time:.1f}с ({job.realtime_factor:.1f}x realtime)")

            # Если нет output — выводим в консоль
            if not output_path or args.verbose:
                print_result(job.result, quiet=args.quiet)

            successful = 1

        else:
            # Несколько файлов — пакетная обработка
            if not args.quiet:
                print(f"📂 Пакетная обработка: {len(input_files)} файлов")

            # Определяем output_dir
            output_dir = None
            if args.output:
                op = Path(args.output)
                if op.is_dir() or not op.suffix:
                    output_dir = op
                else:
                    # Если указан файл, но файлов несколько — используем директорию файла
                    output_dir = op.parent

            jobs = transcriber.transcribe_batch(
                input_files,
                output_dir=output_dir,
                format=output_format or "txt",
                num_speakers=args.num_speakers,
            )

            for i, job in enumerate(jobs):
                if job is not None:
                    successful += 1
                    if not args.quiet:
                        status = f"✅ {job.input_path.name}"
                        if job.output_path:
                            status += f" → {job.output_path.name}"
                        status += f" ({job.realtime_factor:.1f}x)"
                        print(status)
                else:
                    failed += 1
                    if not args.quiet:
                        print(f"❌ {input_files[i].name}: ошибка")

            if not args.quiet:
                total_time = time.time() - total_start
                print(f"\n📊 Итого: {successful}/{len(input_files)} успешно за {total_time:.1f}с")

    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        transcriber.unload()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
