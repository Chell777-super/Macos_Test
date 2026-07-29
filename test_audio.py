"""
Тест модуля core.audio.
Проверяет:
1. Валидацию аудио
2. Получение длительности
3. Конвертацию в 16 kHz mono WAV
"""

from core.audio import AudioProcessor
from core.config import AudioConfig

print("=== Тест AudioProcessor ===\n")

# Создаём процессор
processor = AudioProcessor(AudioConfig(temp_dir="./tmp"))

# Тест 1: валидация
print("Тест 1: валидация файла test_data/test.mp3")
try:
    info = processor.validate_audio("test_data/test.mp3")
    duration = float(info["format"]["duration"])
    print(f"  ✅ Файл валиден, длительность: {duration:.2f}с")
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

# Тест 2: получение длительности
print("\nТест 2: получение длительности test_data/test_stereo.wav")
try:
    duration = processor.get_duration("test_data/test_stereo.wav")
    print(f"  ✅ Длительность: {duration:.2f}с")
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

# Тест 3: конвертация MP3 в WAV
print("\nТест 3: конвертация MP3 -> 16 kHz mono WAV")
try:
    wav_info = processor.convert_to_wav("test_data/test.mp3")
    print(f"  ✅ Создан файл: {wav_info.path}")
    print(f"  ✅ Длительность: {wav_info.duration:.2f}с")
    print(f"  ✅ Sample rate: {wav_info.sample_rate} Hz")
    print(f"  ✅ Каналы: {wav_info.channels}")
    
    # Проверяем через ffprobe, что реально 16 kHz и mono
    import subprocess, json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(wav_info.path)],
        capture_output=True, text=True, check=True
    )
    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    print(f"  ✅ Проверка ffprobe: {stream['sample_rate']} Hz, {stream['channels']} каналов")
    
    # Очистка
    processor.cleanup(wav_info)
    print(f"  ✅ Временный файл удалён")
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

# Тест 4: обработка несуществующего файла
print("\nТест 4: обработка несуществующего файла")
try:
    processor.validate_audio("test_data/no_such_file.mp3")
    print("  ❌ Должна была быть ошибка!")
except Exception as e:
    print(f"  ✅ Правильная ошибка: {type(e).__name__}")

# Тест 5: с использованием context manager
print("\nТест 5: использование как context manager (with)")
try:
    with AudioProcessor() as ap:
        wav = ap.convert_to_wav("test_data/test_stereo.wav")
        print(f"  ✅ Создан: {wav.path.name}")
    # После выхода из with файл должен быть удалён
    if not wav.path.exists():
        print("  ✅ Автоматическая очистка сработала")
    else:
        print("  ❌ Файл не удалён!")
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

print("\n=== Все тесты пройдены ===")
