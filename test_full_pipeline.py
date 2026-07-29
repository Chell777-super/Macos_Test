"""
Тест полного pipeline: транскрибация + экспорт.
"""

import time
from core.merge import Merger

print("=== Тест полного pipeline ===\n")

merger = Merger()

print("Транскрибируем speech.wav...")
start = time.time()
result = merger.merge("test_data/speech.wav")
elapsed = time.time() - start

print(f"✅ Транскрибация за {elapsed:.2f}с")
print(f"   Реплик: {len(result.utterances)}")
print(f"   Спикеров: {len(result.speakers)}")

print("\nЭкспортируем во все форматы...")
results = merger.exporter.export_all_formats(
    result,
    "./output",
    "speech"
)

print("\n✅ Созданные файлы:")
for fmt, path in results.items():
    if path:
        print(f"   {fmt}: {path} ({path.stat().st_size} байт)")

# Показываем содержимое TXT
txt_path = results.get('txt')
if txt_path:
    print(f"\n{'='*60}")
    print(f"Содержимое {txt_path.name}:")
    print('='*60)
    print(txt_path.read_text(encoding='utf-8'))

merger.unload()

print("\n=== Тест завершён ===")
