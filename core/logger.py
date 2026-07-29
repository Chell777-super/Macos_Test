"""
Модуль логирования.

Все события движка пишутся в файл logs/transcriber.log.
В консоль (stdout) ничего не выводим — это задача CLI/GUI.
"""

import logging
from pathlib import Path


def setup_logger(name: str = "transcriber", log_dir: str = "logs") -> logging.Logger:
    """
    Настраивает и возвращает логгер.
    
    Args:
        name: имя логгера
        log_dir: папка для лог-файлов
        
    Returns:
        Настроенный логгер
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Не добавляем хендлеры повторно
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(
        log_path / "transcriber.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Глобальный логгер
logger = setup_logger()
