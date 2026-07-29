"""
Тест модуля core.export.
"""

from core.export import TranscriptExporter
from core.merge import TranscriptResult, Utterance

print("=== Тест TranscriptExporter ===\n")

# Создаём тестовый результат
result = TranscriptResult(
    utterances=[
        Utterance(
            speaker="Спикер 1",
            start=0.5,
            end=3.2,
            text="Привет, как дела?"
        ),
        Utterance(
            speaker="Спикер 2",
            start=4.0,
            end=7.5,
            text="Всё хорошо, спасибо! А у тебя?"
        ),
        Utterance(
            speaker="Спикер 1",
            start=8.2,
            end=12.8,
            text="Отлично! Сегодня отличная погода для прогулки."
        ),
    ],
    speakers=["Спикер 1", "Спикер 2"],
    full_text="Привет, как дела? Всё хорошо, спасибо! А у тебя? Отлично! Сегодня отличная погода для прогулки.",
    duration=15.0
)

exporter = TranscriptExporter()

print("Тест 1: Экспорт в TXT")
txt = exporter.to_txt(result)
print(txt)
print()

print("="*60)
print("Тест 2: Экспорт в JSON")
json_output = exporter.to_json(result)
print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
print()

print("="*60)
print("Тест 3: Экспорт в SRT")
srt = exporter.to_srt(result)
print(srt)
print()

print("="*60)
print("Тест 4: Экспорт в VTT")
vtt = exporter.to_vtt(result)
print(vtt)
print()

print("="*60)
print("Тест 5: Экспорт в Markdown")
md = exporter.to_md(result)
print(md)
print()

print("="*60)
print("Тест 6: Сохранение в файлы")
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # Экспорт в один формат
    path = exporter.export(result, f"{tmpdir}/test", format="txt")
    print(f"✅ TXT сохранён: {path}")
    print(f"   Размер: {path.stat().st_size} байт")
    
    # Экспорт во все форматы
    results = exporter.export_all_formats(result, f"{tmpdir}/all_formats", "test")
    print(f"\n✅ Экспорт во все форматы:")
    for fmt, path in results.items():
        if path:
            print(f"   {fmt}: {path.name} ({path.stat().st_size} байт)")
        else:
            print(f"   {fmt}: ❌ ОШИБКА")

print("\n=== Тесты завершены ===")
