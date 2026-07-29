"""
Тест производительности на длинной записи (30 минут).
"""

import time
from core.merge import Merger
from core.config import (
    MergeConfig, AsrConfig, DiarizationConfig,
    VadConfig, LongFormConfig
)

print("=== Тест производительности ===\n")

# Тест 1: короткая запись (baseline)
print("Тест 1: короткая запись (speech.wav)")
merger = Merger(
    merge_config=MergeConfig(),
    asr_config=AsrConfig(),
    diarization_config=DiarizationConfig(),
    vad_config=VadConfig(),
    long_form_config=LongFormConfig()
)

try:
    start = time.time()
    result = merger.merge("test_data/speech.wav")
    elapsed = time.time() - start

    print(f"  Время: {elapsed:.2f}с")
    print(f"  Длительность: {result.duration:.2f}с")
    print(f"  Скорость: {result.duration / elapsed:.1f}x realtime")
    print(f"  Реплик: {len(result.utterances)}")
    print(f"  Спикеров: {len(result.speakers)}")
except Exception as e:
    print(f"  Ошибка: {e}")

# Тест 2: длинная запись (30 минут)
print("\nТест 2: длинная запись (long_30min.wav, 30 минут)")
merger2 = Merger(
    merge_config=MergeConfig(),
    asr_config=AsrConfig(),
    diarization_config=DiarizationConfig(),
    vad_config=VadConfig(),
    long_form_config=LongFormConfig(chunk_size=600, chunk_overlap=5)
)

def print_progress(percent, message):
    if percent % 10 < 1:  # печатаем каждые ~10%
        print(f"  [{percent:5.1f}%] {message}")

try:
    start = time.time()
    result2 = merger2.merge(
        "test_data/long_30min.wav",
        progress_callback=print_progress
    )
    elapsed = time.time() - start

    print(f"\n  Время: {elapsed:.2f}с ({elapsed/60:.1f} мин)")
    print(f"  Длительность: {result2.duration:.2f}с ({result2.duration/60:.1f} мин)")
    print(f"  Скорость: {result2.duration / elapsed:.1f}x realtime")
    print(f"  Реплик: {len(result2.utterances)}")
    print(f"  Спикеров: {len(result2.speakers)}")
    print(f"\n  Первые 3 реплики:")
    for u in result2.utterances[:3]:
        print(f"    [{u.start_str} -> {u.end_str}] {u.speaker}: {u.text[:50]}...")
except Exception as e:
    print(f"  Ошибка: {e}")
    import traceback
    traceback.print_exc()

merger.unload()
merger2.unload()
Merger.clear_all_caches()

print("\n=== Тесты завершены ===")
