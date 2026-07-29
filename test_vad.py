"""
Тест модуля core.vad с локальным Silero VAD.
"""

import time
from core.vad import VadEngine
from core.config import VadConfig

print("=== Тест VadEngine ===\n")

vad = VadEngine(VadConfig())

print("Тест 1: VAD на multi_speaker.wav")
try:
    start = time.time()
    segments = vad.detect("test_data/multi_speaker.wav")
    elapsed = time.time() - start

    print(f"  Время: {elapsed:.2f}с")
    print(f"  Найдено участков речи: {len(segments)}")
    total_speech = sum(s.duration for s in segments)
    print(f"  Всего речи: {total_speech:.2f}с")
    print(f"  Участки:")
    for s in segments:
        print(f"    {s.start:.2f} - {s.end:.2f} ({s.duration:.2f}с)")
except Exception as e:
    print(f"  Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\nТест 2: VAD на speech.wav (один спикер)")
try:
    start = time.time()
    segments = vad.detect("test_data/speech.wav")
    elapsed = time.time() - start

    print(f"  Время: {elapsed:.2f}с")
    print(f"  Участков: {len(segments)}")
    for s in segments:
        print(f"    {s.start:.2f} - {s.end:.2f} ({s.duration:.2f}с)")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\nТест 3: VAD выключен")
try:
    vad_off = VadEngine(VadConfig(enabled=False))
    segments = vad_off.detect("test_data/multi_speaker.wav")
    print(f"  Участков: {len(segments)} (ожидался 1)")
    for s in segments:
        print(f"    {s.start:.2f} - {s.end:.2f}")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n=== Тесты завершены ===")
