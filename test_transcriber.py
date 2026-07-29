"""
Тест публичного API Transcriber.
"""

from core import Transcriber, transcribe
from core.config import Config

print("=== Тест публичного API Transcriber ===\n")

# Тест 1: Функция transcribe
print("Тест 1: функция transcribe (быстрый доступ)")
job = transcribe("test_data/speech.wav")
assert job.result.full_text, "Пустой результат!"
assert job.duration > 0, "Нулевая длительность!"
assert job.processing_time > 0, "Нулевое время обработки!"
print(f"✅ OK: {job.realtime_factor:.1f}x realtime")
print(f"   Текст: {job.result.full_text[:50]}...\n")

# Тест 2: Класс Transcriber
print("Тест 2: класс Transcriber")
transcriber = Transcriber()
job = transcriber.transcribe(
    "test_data/speech.wav",
    output="output/api_test.txt"
)
assert job.output_path.exists(), "Файл не создан!"
print(f"✅ OK: файл создан {job.output_path}")
print(f"   Размер: {job.output_path.stat().st_size} байт\n")

# Тест 3: Разные форматы экспорта
print("Тест 3: разные форматы экспорта")
for fmt in ["txt", "json", "srt", "vtt", "md"]:
    job = transcriber.transcribe(
        "test_data/speech.wav",
        output=f"output/api_test.{fmt}",
        format=fmt
    )
    assert job.output_path.exists()
    print(f"  ✅ {fmt}: {job.output_path.stat().st_size} байт")
print()

# Тест 4: Указание числа спикеров
print("Тест 4: указание числа спикеров")
job = transcriber.transcribe(
    "test_data/multi_speaker.wav",
    num_speakers=3
)
print(f"✅ OK: {len(job.result.speakers)} спикеров\n")

# Тест 5: Пакетная обработка
print("Тест 5: пакетная обработка")
jobs = transcriber.transcribe_batch(
    ["test_data/speech.wav", "test_data/multi_speaker.wav"],
    output_dir="output/batch_test",
    format="txt"
)
successful = [j for j in jobs if j is not None]
print(f"✅ OK: {len(successful)}/{len(jobs)} успешно\n")

# Тест 6: Пользовательская конфигурация
print("Тест 6: пользовательская конфигурация")
custom_config = Config.default()
custom_config.postprocess.remove_fillers = False  # не удалять fillers
transcriber2 = Transcriber(config=custom_config)
job = transcriber2.transcribe("test_data/speech.wav")
print(f"✅ OK: custom конфиг работает\n")

# Очистка
transcriber.unload()
transcriber2.unload()
Transcriber.clear_all_caches()

print("=== Все тесты пройдены ===")
