"""
Тест модуля core.merge.
Проверяет полную цепочку: диаризация + распознавание по сегментам + склейка.
"""

import time
from core.merge import Merger
from core.config import MergeConfig, AsrConfig, DiarizationConfig

print("=== Тест Merger ===\n")

def print_progress(percent, message):
    print(f"  [{percent:5.1f}%] {message}")

# Тест 1: автоматическое определение числа спикеров
print("Тест 1: файл multi_speaker.wav (авто число спикеров)")
merger = Merger(
    merge_config=MergeConfig(),
    asr_config=AsrConfig(),
    diarization_config=DiarizationConfig()
)

try:
    start = time.time()
    result = merger.merge(
        "test_data/multi_speaker.wav",
        progress_callback=print_progress
    )
    elapsed = time.time() - start

    print(f"\n  Время: {elapsed:.2f}с")
    print(f"  Длительность аудио: {result.duration:.2f}с")
    print(f"  Спикеров: {len(result.speakers)} -> {result.speakers}")
    print(f"  Реплик: {len(result.utterances)}")
    print(f"\n  === Транскрипт ===")
    for u in result.utterances:
        print(f"  [{u.start_str} -> {u.end_str}] {u.speaker}: {u.text}")
    print(f"\n  === Полный текст ===")
    print(f"  {result.full_text}")

except Exception as e:
    print(f"  Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 2: с указанием точного числа спикеров
print("\n\nТест 2: то же, но с exact_speakers=3")
merger2 = Merger(
    merge_config=MergeConfig(),
    asr_config=AsrConfig(),
    diarization_config=DiarizationConfig(exact_speakers=3)
)

try:
    start = time.time()
    result2 = merger2.merge(
        "test_data/multi_speaker.wav",
        progress_callback=print_progress
    )
    elapsed = time.time() - start

    print(f"\n  Время: {elapsed:.2f}с")
    print(f"  Спикеров: {len(result2.speakers)} -> {result2.speakers}")
    print(f"  Реплик: {len(result2.utterances)}")
    print(f"\n  === Транскрипт ===")
    for u in result2.utterances:
        print(f"  [{u.start_str} -> {u.end_str}] {u.speaker}: {u.text}")

except Exception as e:
    print(f"  Ошибка: {e}")

# Тест 3: отмена
print("\n\nТест 3: отмена")
def always_cancel():
    return True

try:
    merger.merge("test_data/multi_speaker.wav", cancel_token=always_cancel)
    print("  Должна была быть ошибка отмены")
except Exception as e:
    print(f"  Отмена сработала: {type(e).__name__}")

# Очистка
merger.unload()
Merger.clear_all_caches()

print("\n=== Тесты завершены ===")