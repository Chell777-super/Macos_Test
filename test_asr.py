"""
Тест модуля core.asr.
"""

import time
from core.asr import AsrEngine
from core.config import AsrConfig

print("=== Тест AsrEngine ===\n")

engine = AsrEngine(AsrConfig())
print(f"Выбранное устройство: {engine.device}")
print(f"Модель: {engine.config.model_name}")

print("\nТест 1: распознавание test_data/speech.wav")
try:
    start = time.time()
    result = engine.transcribe("test_data/speech.wav")
    elapsed = time.time() - start

    print(f"  Распознавание завершено за {elapsed:.2f}с")
    print(f"  Длительность аудио: {result.duration:.2f}с")
    print(f"  Длина текста: {len(result.text)} символов")
    print(f"  Word timings получено: {len(result.words)}")
    print(f"  Текст: {result.text}")

    if result.words:
        print(f"  Первые 5 таймингов:")
        for w in result.words[:5]:
            print(f"      {w.start:.2f}-{w.end:.2f}: {w.word}")
    else:
        print(f"  Тайминги слов не получены (будет fallback в merge)")

except Exception as e:
    print(f"  Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\nТест 2: повторный вызов (кэш)")
try:
    start = time.time()
    result2 = engine.transcribe("test_data/speech.wav")
    elapsed = time.time() - start
    print(f"  Повторный вызов: {elapsed:.2f}с")
    print(f"  Текст совпадает: {result2.text == result.text}")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\nТест 3: отмена через cancel_token")
def always_cancel():
    return True

try:
    engine.transcribe("test_data/speech.wav", cancel_token=always_cancel)
    print("  Должна была быть ошибка отмены")
except Exception as e:
    print(f"  Отмена сработала: {type(e).__name__}")

print("\nТест 4: очистка кэша")
try:
    AsrEngine.clear_cache()
    print("  Кэш очищен")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n=== Тесты завершены ===")
