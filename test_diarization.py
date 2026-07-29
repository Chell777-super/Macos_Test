"""
Тест модуля core.diarization.
"""

import time
from core.diarization import DiarizationEngine
from core.config import DiarizationConfig

print("=== Тест DiarizationEngine ===\n")

engine = DiarizationEngine(DiarizationConfig())

print("\nТест 1: диаризация файла с 3 спикерами")
try:
    start = time.time()
    result = engine.diarize("test_data/multi_speaker.wav")
    elapsed = time.time() - start

    print(f"  Диаризация завершена за {elapsed:.2f}с")
    print(f"  Длительность аудио: {result.duration:.2f}с")
    print(f"  Обнаружено спикеров: {result.num_speakers}")
    print(f"  Всего сегментов: {len(result.segments)}")
    print(f"  Метки спикеров: {result.get_speaker_labels()}")
    print(f"  Сегменты:")
    for seg in result.segments:
        print(f"    {seg.start:.2f} - {seg.end:.2f}: {seg.speaker} ({seg.duration:.2f}с)")
except Exception as e:
    print(f"  Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\nТест 2: указание точного числа спикеров")
try:
    engine2 = DiarizationEngine(DiarizationConfig(exact_speakers=3))
    result2 = engine2.diarize("test_data/multi_speaker.wav")
    print(f"  Обнаружено спикеров: {result2.num_speakers} (ожидалось: 3)")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\nТест 3: отмена через cancel_token")
def always_cancel():
    return True

try:
    engine.diarize("test_data/multi_speaker.wav", cancel_token=always_cancel)
    print("  Должна была быть ошибка отмены")
except Exception as e:
    print(f"  Отмена сработала: {type(e).__name__}")

print("\nТест 4: очистка кэша")
try:
    DiarizationEngine.clear_cache()
    print("  Кэш очищен")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n=== Тесты завершены ===")
