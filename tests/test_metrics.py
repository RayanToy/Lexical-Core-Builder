"""
Тесты для src/metrics.py

Покрываем:
- calculate_juyna_coefficient: граничные случаи и математическую корректность
- coverage_percent: нормальная работа и граничные случаи
- normalized_frequency: базовая логика и деление на ноль
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    calculate_juyna_coefficient,
    coverage_percent,
    normalized_frequency,
)


# ══════════════════════════════════════════════════════════════════════════════
# calculate_juyna_coefficient
# ══════════════════════════════════════════════════════════════════════════════

class TestJuynaCoefficient:
    """Тесты коэффициента Жуайна."""

    # ── Граничные случаи ──────────────────────────────────────────────────────

    def test_empty_list_returns_zero(self):
        """Пустой список — нет данных для вычисления."""
        assert calculate_juyna_coefficient([]) == 0.0

    def test_single_value_returns_zero(self):
        """Одно значение — нельзя вычислить выборочное σ."""
        assert calculate_juyna_coefficient([42.0]) == 0.0

    def test_all_zeros_returns_zero(self):
        """Все значения ноль — μ == 0, деление на ноль."""
        assert calculate_juyna_coefficient([0.0, 0.0, 0.0]) == 0.0

    def test_result_never_negative(self):
        """Результат всегда >= 0, даже при высокой дисперсии."""
        # Очень высокий разброс
        result = calculate_juyna_coefficient([1.0, 1000.0, 1.0, 1000.0])
        assert result >= 0.0

    def test_result_never_exceeds_100(self):
        """Результат всегда <= 100."""
        result = calculate_juyna_coefficient([5.0, 5.0, 5.0, 5.0])
        assert result <= 100.0

    # ── Математическая корректность ───────────────────────────────────────────

    def test_identical_values_return_100(self):
        """
        Если все значения одинаковы — σ == 0, J == 100.
        Проверяем для разных длин списка.
        """
        assert calculate_juyna_coefficient([10.0, 10.0, 10.0]) == 100.0
        assert calculate_juyna_coefficient([1.0, 1.0]) == 100.0
        assert calculate_juyna_coefficient([0.5, 0.5, 0.5, 0.5, 0.5]) == 100.0

    def test_two_values_manual_calculation(self):
        """
        Ручная проверка формулы для n=2:
        μ = (4 + 8) / 2 = 6
        σ = std([4, 8], ddof=1) = 2√2 ≈ 2.828
        √(n-1) = √1 = 1
        J = 100 * (1 - 2.828 / (6 * 1)) ≈ 100 * (1 - 0.4714) ≈ 52.9
        """
        values = [4.0, 8.0]
        mu = np.mean(values)
        sigma = np.std(values, ddof=1)
        n = len(values)
        expected = round(max(0.0, 100.0 * (1.0 - sigma / (mu * np.sqrt(n - 1)))), 1)

        result = calculate_juyna_coefficient(values)
        assert result == expected

    def test_three_values_manual_calculation(self):
        """Ручная проверка для n=3."""
        values = [10.0, 20.0, 30.0]
        mu = np.mean(values)
        sigma = np.std(values, ddof=1)
        n = len(values)
        expected = round(max(0.0, 100.0 * (1.0 - sigma / (mu * np.sqrt(n - 1)))), 1)

        result = calculate_juyna_coefficient(values)
        assert result == expected

    def test_higher_variance_gives_lower_coefficient(self):
        """Чем выше разброс — тем ниже коэффициент."""
        low_variance = calculate_juyna_coefficient([10.0, 11.0, 10.0, 11.0])
        high_variance = calculate_juyna_coefficient([1.0, 20.0, 1.0, 20.0])
        assert low_variance > high_variance

    # ── NaN-значения ──────────────────────────────────────────────────────────

    def test_nan_values_are_ignored(self):
        """NaN игнорируются — результат как будто их нет."""
        result_with_nan = calculate_juyna_coefficient([10.0, float("nan"), 10.0])
        result_without_nan = calculate_juyna_coefficient([10.0, 10.0])
        assert result_with_nan == result_without_nan

    def test_numpy_nan_ignored(self):
        """numpy.nan тоже должен игнорироваться."""
        result = calculate_juyna_coefficient([5.0, np.nan, 5.0])
        assert result == 100.0

    def test_all_nan_returns_zero(self):
        """Если все значения NaN — список после фильтрации пуст."""
        result = calculate_juyna_coefficient([float("nan"), float("nan")])
        assert result == 0.0

    # ── Округление ────────────────────────────────────────────────────────────

    def test_result_is_rounded_to_one_decimal(self):
        """Результат округлён до 1 знака после запятой."""
        result = calculate_juyna_coefficient([1.0, 2.0, 3.0])
        # Проверяем что это число с 1 знаком
        assert result == round(result, 1)

    # ── Параметризованные тесты ───────────────────────────────────────────────

    @pytest.mark.parametrize("values,expected", [
        ([10.0, 10.0], 100.0),           # одинаковые -> 100
        ([0.0, 0.0, 0.0], 0.0),          # все нули -> 0
        ([], 0.0),                        # пустой -> 0
        ([42.0], 0.0),                    # одно значение -> 0
    ])
    def test_parametrized_edge_cases(self, values, expected):
        assert calculate_juyna_coefficient(values) == expected


# ══════════════════════════════════════════════════════════════════════════════
# coverage_percent
# ══════════════════════════════════════════════════════════════════════════════

class TestCoveragePercent:
    """Тесты процента покрытия."""

    def test_full_coverage(self):
        """Слово есть во всех учебниках."""
        result = coverage_percent("кот", {"кот": 4}, total_textbooks=4)
        assert result == 100.0

    def test_partial_coverage(self):
        """Слово есть в 3 из 4 учебников."""
        result = coverage_percent("кот", {"кот": 3}, total_textbooks=4)
        assert result == 75.0

    def test_zero_coverage(self):
        """Слово не встречается ни в одном учебнике."""
        result = coverage_percent("кот", {}, total_textbooks=4)
        assert result == 0.0

    def test_word_not_in_dict(self):
        """Слова нет в словаре — считаем как 0 вхождений."""
        result = coverage_percent("пёс", {"кот": 3}, total_textbooks=4)
        assert result == 0.0

    def test_zero_total_textbooks_returns_zero(self):
        """Деление на ноль — возвращаем 0.0, не исключение."""
        result = coverage_percent("кот", {"кот": 3}, total_textbooks=0)
        assert result == 0.0

    def test_result_is_rounded_to_one_decimal(self):
        """Результат округлён до 1 знака."""
        # 1/3 ≈ 33.3%
        result = coverage_percent("слово", {"слово": 1}, total_textbooks=3)
        assert result == 33.3

    def test_result_never_exceeds_100(self):
        """Теоретически невозможно, но coverage_counts не должен давать > 100%."""
        result = coverage_percent("кот", {"кот": 4}, total_textbooks=4)
        assert result <= 100.0

    @pytest.mark.parametrize("count,total,expected", [
        (0, 10, 0.0),
        (10, 10, 100.0),
        (5, 10, 50.0),
        (1, 4, 25.0),
        (2, 3, 66.7),
    ])
    def test_parametrized(self, count, total, expected):
        result = coverage_percent("x", {"x": count}, total_textbooks=total)
        assert result == expected


# ══════════════════════════════════════════════════════════════════════════════
# normalized_frequency
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizedFrequency:
    """Тесты нормализованной частотности."""

    def test_basic_calculation(self):
        """Базовый расчёт: 50 вхождений на 100 000 слов -> 500 на миллион."""
        result = normalized_frequency(50, 100_000)
        assert result == 500.0

    def test_zero_count_returns_zero(self):
        """Слово не встречается — НЧ == 0."""
        result = normalized_frequency(0, 100_000)
        assert result == 0.0

    def test_zero_total_tokens_returns_zero(self):
        """Пустой текст — нет деления на ноль."""
        result = normalized_frequency(10, 0)
        assert result == 0.0

    def test_custom_norm_base(self):
        """Пользовательская база нормализации."""
        result = normalized_frequency(10, 1000, norm_base=10_000)
        assert result == 100.0

    def test_large_corpus(self):
        """Корпус в 1 миллион слов, 500 вхождений -> НЧ == 500."""
        result = normalized_frequency(500, 1_000_000)
        assert result == 500.0

    def test_result_is_float(self):
        """Результат всегда float."""
        result = normalized_frequency(10, 1000)
        assert isinstance(result, float)

    def test_proportional_to_raw_count(self):
        """
        При удвоении raw_count результат удваивается
        (линейная зависимость).
        """
        r1 = normalized_frequency(10, 100_000)
        r2 = normalized_frequency(20, 100_000)
        assert math.isclose(r2, r1 * 2)

    def test_inversely_proportional_to_corpus_size(self):
        """
        При удвоении корпуса НЧ уменьшается вдвое
        (обратная пропорциональность).
        """
        r1 = normalized_frequency(10, 100_000)
        r2 = normalized_frequency(10, 200_000)
        assert math.isclose(r1, r2 * 2)

    @pytest.mark.parametrize("raw,total,base,expected", [
        (100,  1_000_000, 1_000_000, 100.0),
        (1,    1_000,     1_000,     1.0),
        (0,    500_000,   1_000_000, 0.0),
        (500,  1_000_000, 1_000_000, 500.0),
        (1000, 500_000,   1_000_000, 2000.0),
    ])
    def test_parametrized(self, raw, total, base, expected):
        result = normalized_frequency(raw, total, norm_base=base)
        assert math.isclose(result, expected)