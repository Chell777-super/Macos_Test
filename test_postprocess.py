"""
Тест модуля core.postprocess.
"""

from core.postprocess import TextPostprocessor
from core.config import PostprocessConfig

print("=== Тест TextPostprocessor ===\n")

# Создаём с настройками по умолчанию
pp = TextPostprocessor(PostprocessConfig())

test_cases = [
    (
        "Привет, эээ, как дела?",
        "Удаление filler-слов (эээ)"
    ),
    (
        "Я я я пошёл в магазин.",
        "Удаление stuttering (я я я)"
    ),
    (
        "это было очень круто!!! правда правда.",
        "Очистка пунктуации и повторов"
    ),
    (
        "  Привет,  мм,  как  дела?  ",
        "Нормализация пробелов"
    ),
    (
        "привет. как дела? всё хорошо.",
        "Капитализация предложений"
    ),
    (
        "Ну, эээ, я, мм, пошёл в магазин, типа, за хлебом.",
        "Комплексный тест"
    ),
    (
        "Привет. Это тестовая запись для проверки распознавания речи.",
        "Проверка, что нормальный текст не портится"
    ),
]

for input_text, description in test_cases:
    output_text = pp.process(input_text)
    changed = "✅ ИЗМЕНЁН" if input_text != output_text else "➖ НЕ ИЗМЕНЁН"
    print(f"Тест: {description}")
    print(f"  Вход:  '{input_text}'")
    print(f"  Выход: '{output_text}'")
    print(f"  {changed}\n")

# Тест с выключенной обработкой
print("="*60)
print("Тест с выключенной обработкой:")
pp_off = TextPostprocessor(PostprocessConfig(enabled=False))
test = "Привет, эээ, как дела?"
result = pp_off.process(test)
print(f"  Вход:  '{test}'")
print(f"  Выход: '{result}'")
print(f"  {'✅ Не изменён' if test == result else '❌ Изменён!'}")

print("\n=== Тесты завершены ===")

print("\n" + "="*60)
print("Дополнительные тесты: сохранение 'типа' в техническом контексте")
print("="*60)

extra_cases = [
    ("Это переменная типа boolean.", "Технический контекст (типа boolean)"),
    ("Какого типа эта функция?", "Вопрос с 'типа'"),
    ("Эээ, это типа тест.", "Filler 'эээ' + 'типа'"),
    ("Это было круто, э, правда.", "Одиночный filler 'э' не трогает 'это'"),
]

for input_text, description in extra_cases:
    output_text = pp.process(input_text)
    changed = "✅ ИЗМЕНЁН" if input_text != output_text else "➖ НЕ ИЗМЕНЁН"
    print(f"\nТест: {description}")
    print(f"  Вход:  '{input_text}'")
    print(f"  Выход: '{output_text}'")
    print(f"  {changed}")
    # Проверяем, что "типа" сохранилось, если было во входе
    if "типа" in input_text and "типа" not in output_text:
        print(f"  ❌ ОШИБКА: 'типа' исчезло!")

print("\n=== Все тесты завершены ===")

print("\n" + "="*60)
print("Дополнительные тесты: сохранение 'типа' в техническом контексте")
print("="*60)

extra_cases = [
    ("Это переменная типа boolean.", "Технический контекст (типа boolean)"),
    ("Какого типа эта функция?", "Вопрос с 'типа'"),
    ("Эээ, это типа тест.", "Filler 'эээ' + 'типа'"),
    ("Это было круто, э, правда.", "Одиночный filler 'э' не трогает 'это'"),
]

for input_text, description in extra_cases:
    output_text = pp.process(input_text)
    changed = "✅ ИЗМЕНЁН" if input_text != output_text else "➖ НЕ ИЗМЕНЁН"
    print(f"\nТест: {description}")
    print(f"  Вход:  '{input_text}'")
    print(f"  Выход: '{output_text}'")
    print(f"  {changed}")
    # Проверяем, что "типа" сохранилось, если было во входе
    if "типа" in input_text and "типа" not in output_text:
        print(f"  ❌ ОШИБКА: 'типа' исчезло!")

print("\n=== Все тесты завершены ===")
