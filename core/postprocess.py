"""
Модуль постобработки текста.

Выполняет:
- Удаление filler-слов (эээ, мм, ага) с учётом контекста
- Удаление повторов слов (stuttering)
- Очистку "осиротевших" знаков препинания
- Нормализацию пробелов (с гарантией пробела после знаков препинания)
- Капитализацию предложений (с сохранением пробелов)

Важно: НЕ трогает пунктуацию, которую ставит GigaAM e2e_rnnt,
она уже высокого качества.
"""

import re
from dataclasses import dataclass
from typing import Optional, List

from .config import PostprocessConfig
from .logger import logger


class PostprocessError(Exception):
    pass


class TextPostprocessor:
    """
    Пост-обработчик текста после ASR.
    
    Применяет ряд правил для очистки текста от артефактов распознавания.
    """

    # Filler-слова (русские)
    # Не включаем "типа", "короче" — они могут использоваться в обычном контексте
    FILLER_WORDS = {
        "эээ", "э", "ээ", "ээээ",
        "ммм", "мм", "мммм",
        "ааа", "аа", "аааа",
        "ооо", "оо",
        "ууу", "уу",
        "ага", "угу", "нуу",
    }

    # Базовые регулярки
    RE_TRIPLE_REPEATED = re.compile(r'\b(\w+)\s+\1\s+\1\b', re.IGNORECASE)
    RE_REPEATED_WORD = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
    RE_MULTI_EXCLAIM = re.compile(r'!{2,}')
    RE_MULTI_QUESTION = re.compile(r'\?{2,}')

    def __init__(self, config: Optional[PostprocessConfig] = None):
        self.config = config or PostprocessConfig()
        logger.info(f"TextPostprocessor инициализирован. enabled={self.config.enabled}")

    def _remove_stuttering(self, text: str) -> str:
        """Удаляет повторы слов (stuttering)."""
        if not self.config.remove_stuttering:
            return text
        
        # Сначала тройные повторы: "я я я" -> "я"
        text = self.RE_TRIPLE_REPEATED.sub(r'\1', text)
        
        # Затем двойные повторы (до 5 итераций для цепочек)
        for _ in range(5):
            new_text = self.RE_REPEATED_WORD.sub(r'\1', text)
            if new_text == text:
                break
            text = new_text
        
        return text

    def _remove_fillers(self, text: str) -> str:
        """
        Удаляет filler-слова с учётом контекста (запятых вокруг).
        
        Примеры:
        - "Привет, эээ, как дела?" -> "Привет, как дела?"
        - "Ну, эээ, я пошёл" -> "Ну, я пошёл"
        - "эээ, как дела" -> "Как дела"
        """
        if not self.config.remove_fillers:
            return text
        
        # Сортируем по длине (длинные сначала, чтобы "эээ" удалялось раньше "э")
        fillers = sorted(self.FILLER_WORDS, key=len, reverse=True)
        
        for w in fillers:
            escaped = re.escape(w)
            
            # Случай 1: filler между запятыми: ", эээ," -> ", "
            # \b гарантирует, что это отдельное слово, а не часть "это"
            text = re.sub(rf',\s*\b{escaped}\b,?\s*', ', ', text, flags=re.IGNORECASE)
            
            # Случай 2: filler в начале с запятой: "эээ, " -> ""
            text = re.sub(rf'^\s*\b{escaped}\b,?\s*', '', text, flags=re.IGNORECASE)
            
            # Случай 3: filler в конце: " эээ" -> ""
            text = re.sub(rf',?\s*\b{escaped}\b\s*$', '', text, flags=re.IGNORECASE)
            
            # Случай 4: filler без запятых: " эээ " -> " "
            text = re.sub(rf'\b{escaped}\b', ' ', text, flags=re.IGNORECASE)
        
        return text

    def _clean_orphan_punctuation(self, text: str) -> str:
        """
        Удаляет "осиротевшие" знаки препинания, которые остались после удаления filler'ов.
        """
        # Запятые подряд: ",, " -> ", "
        text = re.sub(r',{2,}', ',', text)
        
        # Запятая в начале текста: ", Привет" -> "Привет"
        text = re.sub(r'^\s*,+\s*', '', text)
        
        # Запятая перед точкой: ",." -> "."
        text = re.sub(r',+\s*\.', '.', text)
        
        # Пробел перед запятой: "Привет , как" -> "Привет, как"
        text = re.sub(r'\s+,', ',', text)
        
        # Точка в начале: ". Привет" -> "Привет"
        text = re.sub(r'^\s*[.!?]+\s*', '', text)
        
        # Запятая перед закрывающей скобкой: "текст, )" -> "текст)"
        text = re.sub(r',\s*([)\]}])', r'\1', text)
        
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Нормализует пунктуацию (убирает избыточные восклицания/вопросы)."""
        if not self.config.normalize_punctuation:
            return text
        
        # "!!!" -> "!", "???" -> "?"
        text = self.RE_MULTI_EXCLAIM.sub('!', text)
        text = self.RE_MULTI_QUESTION.sub('?', text)
        
        return text

    def _normalize_spaces(self, text: str) -> str:
        """
        Нормализует пробелы:
        - Убирает пробелы ПЕРЕД знаками препинания
        - ГАРАНТИРУЕТ пробел ПОСЛЕ точек, запятых, восклицаний, вопросов
        - Несколько пробелов -> один
        """
        if not self.config.normalize_spaces:
            return text
        
        # Убираем пробелы ПЕРЕД знаками препинания
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        
        # Убираем пробелы после открывающих скобок
        text = re.sub(r'([(\[{])\s+', r'\1', text)
        
        # Убираем пробелы перед закрывающими скобками
        text = re.sub(r'\s+([)\]}])', r'\1', text)
        
        # ГАРАНТИРУЕМ пробел ПОСЛЕ . ! ? (если там идёт буква или цифра)
        text = re.sub(r'([.!?])([А-Яа-яA-Za-z0-9А-Я])', r'\1 \2', text)
        
        # ГАРАНТИРУЕМ пробел ПОСЛЕ запятой
        text = re.sub(r',([А-Яа-яA-Za-z0-9А-Я])', r', \1', text)
        
        # Несколько пробелов -> один
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()

    def _capitalize_sentences(self, text: str) -> str:
        """
        Делает первую букву каждого предложения заглавной.
        
        ВАЖНО: сохраняет пробелы после знаков препинания!
        """
        if not self.config.capitalize_sentences or not text:
            return text
        
        # Первая буква текста - заглавная
        text = text[0].upper() + text[1:]
        
        # После . ! ? + пробелов делаем следующую букву заглавной
        def capitalize_after(match):
            return match.group(1) + match.group(2).upper()
        
        text = re.sub(r'([.!?]\s+)([а-яa-z])', capitalize_after, text)
        
        return text

    def process(self, text: str) -> str:
        """
        Применяет всю постобработку к тексту.
        
        Args:
            text: исходный текст от ASR
            
        Returns:
            Очищенный текст
        """
        if not self.config.enabled or not text:
            return text

        original = text
        
        # Порядок применения критически важен!
        text = self._remove_stuttering(text)
        text = self._remove_fillers(text)
        text = self._clean_orphan_punctuation(text)
        text = self._normalize_punctuation(text)
        text = self._normalize_spaces(text)  # С гарантией пробелов после знаков!
        text = self._capitalize_sentences(text)  # С сохранением пробелов!

        if text != original:
            logger.debug(
                f"Постобработка: '{original[:50]}...' -> '{text[:50]}...'"
            )

        return text

    def process_utterances(self, utterances: list) -> list:
        """Применяет постобработку ко всем репликам."""
        if not self.config.enabled:
            return utterances
        
        # Создаём новые объекты, чтобы не мутировать оригиналы
        from .merge import Utterance
        return [
            Utterance(
                speaker=u.speaker,
                start=u.start,
                end=u.end,
                text=self.process(u.text)
            )
            for u in utterances
        ]