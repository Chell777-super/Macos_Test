# 🎙️ Transcriber

Движок транскрибации аудио и видео с разделением по спикерам для macOS (Apple Silicon).

## ✨ Возможности

- **Распознавание речи** на русском языке (GigaAM V3 e2e_rnnt)
- **Разделение по спикерам** (NVIDIA Sortformer streaming v2)
- **Определение участков речи** (Silero VAD)
- **Постобработка текста** (удаление filler-слов, повторов, нормализация)
- **Экспорт в 5 форматов**: TXT, JSON, SRT, VTT, Markdown
- **Поддержка любых форматов**: mp3, m4a, wav, flac, mp4, mov, webm, mkv
- **Обработка длинных записей** (1-3 часа) через стриминг и чанкинг
- **CLI и Python API**

## 🔧 Модели

| Компонент | Модель | Версия | Назначение |
|-----------|--------|--------|------------|
| ASR | GigaAM V3 e2e_rnnt | latest | Распознавание русской речи с пунктуацией |
| Диаризация | NVIDIA Sortformer | **v2.1 streaming** | Разделение по спикерам (end-to-end) |
| VAD | Silero VAD | **v6.2.1** | Определение участков с речью (3x быстрее v4) |

## ⚡ Производительность

На MacBook Air M4 16GB:

| Запись | Время обработки | Скорость |
|--------|----------------|----------|
| 10 секунд | ~5-7 сек | 1.5-2x realtime |
| 30 минут | ~56 сек | **31.9x realtime** |
| 1 час | ~2 мин | ~30x realtime |

## 📦 Установка

### Требования

- macOS (Apple Silicon: M1/M2/M3/M4)
- Python 3.10+
- ffmpeg

### Установка зависимостей

```bash
# Клонируем репозиторий
git clone <repo-url> transcriber
cd transcriber

# Создаём виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Клонируем Silero VAD (локально, обходит GitHub rate limit)
mkdir -p vendors
git clone https://github.com/snakers4/silero-vad.git vendors/silero-vad
