"""
Статистические метрики для лексического анализа.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_juyna_coefficient(values: list[float]) -> float:
    """
    Вычисляет коэффициент Жуайна — меру стабильности частотности слова
    по корпусу текстов.

    Показывает, насколько равномерно слово распределено по учебникам:
    - значение близкое к 100 означает равномерное распределение
    - значение близкое к 0 означает высокую вариативность

    Формула:
        J = 100 × (1 − σ / (μ × √(n − 1)))

    где:
        σ — стандартное отклонение (ddof=1, выборочное)
        μ — среднее арифметическое
        n — количество значений

    Граничные случаи:
        - n == 0  → 0.0
        - n == 1  → 0.0  (нельзя вычислить выборочное σ)
        - μ == 0  → 0.0  (деление на ноль)
        - J < 0   → возвращаем 0.0 (обрезаем снизу)

    Args:
        values: список числовых значений нормализованной частотности
                по отдельным учебникам. NaN-значения игнорируются.

    Returns:
        Коэффициент Жуайна в диапазоне [0.0, 100.0], округлённый до 1 знака.

    Examples:
        >>> calculate_juyna_coefficient([10.0, 10.0, 10.0])
        100.0
        >>> calculate_juyna_coefficient([])
        0.0
        >>> calculate_juyna_coefficient([5.0])
        0.0
        >>> calculate_juyna_coefficient([1.0, 100.0])
        0.0  # или близко к 0 при высокой дисперсии
    """
    # Фильтруем NaN и приводим к float
    clean = [float(v) for v in values if pd.notna(v)]

    n = len(clean)
    if n <= 1:
        return 0.0

    mu = np.mean(clean)
    if mu == 0:
        return 0.0

    sigma = np.std(clean, ddof=1)
    juyna = 100.0 * (1.0 - sigma / (mu * np.sqrt(n - 1)))

    return round(max(0.0, juyna), 1)


def coverage_percent(
    word: str,
    coverage_counts: dict[str, int],
    total_textbooks: int,
) -> float:
    """
    Вычисляет процент учебников, в которых встречается слово.

    Args:
        word: нормализованное слово.
        coverage_counts: словарь {слово: количество учебников с этим словом}.
        total_textbooks: общее количество учебников в выборке.

    Returns:
        Процент покрытия в диапазоне [0.0, 100.0], округлённый до 1 знака.
        Возвращает 0.0, если total_textbooks == 0.

    Examples:
        >>> coverage_percent("кот", {"кот": 3}, 4)
        75.0
        >>> coverage_percent("кот", {}, 4)
        0.0
        >>> coverage_percent("кот", {"кот": 2}, 0)
        0.0
    """
    if total_textbooks == 0:
        return 0.0
    count = coverage_counts.get(word, 0)
    return round((count / total_textbooks) * 100.0, 1)


def normalized_frequency(raw_count: float, total_tokens: int, norm_base: int = 1_000_000) -> float:
    """
    Вычисляет нормализованную частотность слова.

    Позволяет сравнивать частоты слов между текстами разного объёма.

    Формула:
        НЧ = (raw_count / total_tokens) × norm_base

    Args:
        raw_count: абсолютная частота слова в тексте.
        total_tokens: общее количество токенов в тексте.
        norm_base: база нормализации (по умолчанию 1 000 000).

    Returns:
        Нормализованная частотность. 0.0, если total_tokens == 0.

    Examples:
        >>> normalized_frequency(50, 100_000)
        500.0
        >>> normalized_frequency(0, 100_000)
        0.0
        >>> normalized_frequency(10, 0)
        0.0
    """
    if total_tokens == 0:
        return 0.0
    return (raw_count / total_tokens) * norm_base