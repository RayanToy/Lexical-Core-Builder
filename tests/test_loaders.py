"""
Тесты для src/loaders.py

Покрываем:
- load_overall_subject_freq_df: валидация, нормализация, обработка ошибок
- load_common_kv_for_grade: кэширование, множественные источники
- load_textbook_freq_df: базовая загрузка учебников
- вспомогательные функции
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.loaders import (
    aggregated_common_kv_upto_grade,
    clear_common_kv_cache,
    load_common_kv_for_grade,
    load_overall_subject_freq_df,
    load_textbook_freq_df,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures: создание тестовых Excel-файлов
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_subject_freq_file(tmp_path):
    """Создаёт корректный файл общего частотного списка предмета."""
    data = {
        "Слово": ["кот", "пёс", "птица", "  РЫБА  "],
        "Сумма слов": [100, 50, 30, 20],
        "Нормализованная частотность": [25.5, 12.3, 7.8, 5.1],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "Общий частотный список Bi 5.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def subject_freq_with_alt_column_name(tmp_path):
    """Файл с альтернативным названием столбца «Нормализованная частота»."""
    data = {
        "Слово": ["кот", "пёс"],
        "Сумма слов": [100, 50],
        "Нормализованная частота": [25.5, 12.3],  # не «частотность»
    }
    df = pd.DataFrame(data)
    path = tmp_path / "список_alt.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def subject_freq_missing_columns(tmp_path):
    """Файл без обязательного столбца «Сумма слов»."""
    data = {
        "Слово": ["кот", "пёс"],
        "Частота": [100, 50],  # неправильное название
    }
    df = pd.DataFrame(data)
    path = tmp_path / "неполный.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def subject_freq_with_duplicates(tmp_path):
    """Файл с дублирующимися словами."""
    data = {
        "Слово": ["кот", "кот", "пёс"],
        "Сумма слов": [100, 200, 50],
        "Нормализованная частотность": [25.5, 30.0, 12.3],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "дубликаты.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def subject_freq_with_nans(tmp_path):
    """Файл с NaN-значениями."""
    data = {
        "Слово": ["кот", np.nan, "пёс", ""],
        "Сумма слов": [100, 50, np.nan, 10],
        "Нормализованная частотность": [25.5, 12.3, 7.8, 5.1],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "с_nan.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def empty_subject_freq_file(tmp_path):
    """Пустой Excel-файл (без данных)."""
    df = pd.DataFrame()
    path = tmp_path / "пустой.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def common_vocab_file(tmp_path):
    """Файл общеупотребительной лексики."""
    data = {
        "Слово": ["общий", "слово", "текст"],
        "Сумма слов": [500, 300, 200],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "Общий частотный список внеклассная 5.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def textbook_file(tmp_path):
    """Файл учебника."""
    data = {
        "Слово": ["клетка", "ДНК", "белок"],
        "Сумма слов": [45, 38, 27],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "учебник_биология_46852.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def config_fixture(tmp_path):
    """Минимальный Config для тестов."""
    return Config(
        base_dir=tmp_path / "base",
        common_fallback_dir=tmp_path / "fallback",
        common_books_dir=tmp_path / "books",
    )


# ══════════════════════════════════════════════════════════════════════════════
# load_overall_subject_freq_df
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadOverallSubjectFreqDF:
    """Тесты загрузки общих частотных списков предметов."""

    # ── Успешная загрузка ─────────────────────────────────────────────────────

    def test_loads_valid_file(self, valid_subject_freq_file):
        """Корректный файл загружается без ошибок."""
        df = load_overall_subject_freq_df(valid_subject_freq_file, "Bi")
        assert not df.empty
        assert len(df) == 4

    def test_has_required_columns(self, valid_subject_freq_file):
        """Результат содержит обязательные столбцы."""
        df = load_overall_subject_freq_df(valid_subject_freq_file, "Bi")
        required = ["Слово", "Сумма слов", "Нормализованная частотность"]
        assert all(col in df.columns for col in required)

    def test_words_are_normalized(self, valid_subject_freq_file):
        """Слова приведены к lowercase и без пробелов."""
        df = load_overall_subject_freq_df(valid_subject_freq_file, "Bi")
        words = df["Слово"].tolist()
        assert "кот" in words
        assert "рыба" in words  # было "  РЫБА  "
        assert "  РЫБА  " not in words

    def test_numeric_columns_are_float(self, valid_subject_freq_file):
        """Числовые столбцы имеют тип float."""
        df = load_overall_subject_freq_df(valid_subject_freq_file, "Bi")
        assert df["Сумма слов"].dtype == float
        assert df["Нормализованная частотность"].dtype == float

    def test_handles_alternative_column_name(self, subject_freq_with_alt_column_name):
        """Столбец «Нормализованная частота» переименовывается."""
        df = load_overall_subject_freq_df(
            subject_freq_with_alt_column_name, "Bi"
        )
        assert "Нормализованная частотность" in df.columns
        assert "Нормализованная частота" not in df.columns

    # ── Обработка дубликатов и NaN ────────────────────────────────────────────

    def test_removes_duplicates(self, subject_freq_with_duplicates):
        """Дублирующиеся слова удаляются (остаётся первое)."""
        df = load_overall_subject_freq_df(subject_freq_with_duplicates, "Bi")
        assert len(df[df["Слово"] == "кот"]) == 1
        # Первое вхождение: КВ=100
        row = df[df["Слово"] == "кот"].iloc[0]
        assert row["Сумма слов"] == 100.0

    def test_removes_nan_words(self, subject_freq_with_nans):
        """Строки с NaN в столбце «Слово» удаляются."""
        df = load_overall_subject_freq_df(subject_freq_with_nans, "Bi")
        assert not df["Слово"].isna().any()

    def test_removes_empty_words(self, subject_freq_with_nans):
        """Пустые строки удаляются."""
        df = load_overall_subject_freq_df(subject_freq_with_nans, "Bi")
        assert "" not in df["Слово"].tolist()

    def test_converts_invalid_numbers_to_zero(self, subject_freq_with_nans):
        """NaN в числовых столбцах заменяется на 0.0."""
        df = load_overall_subject_freq_df(subject_freq_with_nans, "Bi")
        # В строке с «пёс» было NaN в «Сумма слов»
        row = df[df["Слово"] == "пёс"]
        if not row.empty:
            assert row.iloc[0]["Сумма слов"] == 0.0

    # ── Ошибки и исключения ───────────────────────────────────────────────────

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError если файл не существует."""
        with pytest.raises(FileNotFoundError, match="не найден"):
            load_overall_subject_freq_df(tmp_path / "несуществующий.xlsx", "Bi")

    def test_raises_on_missing_columns(self, subject_freq_missing_columns):
        """ValueError если отсутствуют обязательные столбцы."""
        with pytest.raises(ValueError, match="не содержит обязательных столбцов"):
            load_overall_subject_freq_df(subject_freq_missing_columns, "Bi")

    def test_raises_on_empty_file(self, empty_subject_freq_file):
        """EmptyDataError если файл пустой."""
        with pytest.raises(pd.errors.EmptyDataError, match="не содержит данных"):
            load_overall_subject_freq_df(empty_subject_freq_file, "Bi")

    def test_error_message_includes_subject_name(self, tmp_path):
        """Сообщение об ошибке содержит название предмета."""
        with pytest.raises(FileNotFoundError, match="Bi|не найден"):
            load_overall_subject_freq_df(tmp_path / "missing.xlsx", "Bi")


# ══════════════════════════════════════════════════════════════════════════════
# load_common_kv_for_grade
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadCommonKVForGrade:
    """Тесты загрузки общеупотребительной лексики."""

    def setup_method(self):
        """Очистка кэша перед каждым тестом."""
        clear_common_kv_cache()

    # ── Источник 1: основная папка ────────────────────────────────────────────

    def test_loads_from_primary_dir(self, tmp_path, config_fixture):
        """Загружает из основной папки класса."""
        # Создаём структуру: base_dir/5 класс списки/Общеупотребительная лексика/
        base_dir = tmp_path / "base"
        common_dir = base_dir / "5 класс списки" / "Общеупотребительная лексика"
        common_dir.mkdir(parents=True)

        # Файл с данными
        data = {"Слово": ["общий", "слово"], "Сумма слов": [100, 50]}
        df = pd.DataFrame(data)
        (common_dir / "Общий частотный список внекл 5.xlsx").write_bytes(
            pd.ExcelWriter(
                common_dir / "Общий частотный список внекл 5.xlsx"
            ).book.save()
        )
        df.to_excel(common_dir / "Общий частотный список внекл 5.xlsx", index=False)

        config = Config(
            base_dir=base_dir,
            common_fallback_dir=tmp_path / "fallback",
            common_books_dir=tmp_path / "books",
        )

        result = load_common_kv_for_grade(5, config)
        assert "общий" in result
        assert result["общий"] == 100.0

    # ── Источник 2: резервная папка ───────────────────────────────────────────

    def test_loads_from_fallback_dir(self, tmp_path):
        """Если основная папка пуста — использует резервную."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["резерв"], "Сумма слов": [200]}
        df = pd.DataFrame(data)
        df.to_excel(fallback_dir / "Общий частотный список 5 класс.xlsx", index=False)

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        result = load_common_kv_for_grade(5, config)
        assert "резерв" in result
        assert result["резерв"] == 200.0

    # ── Источник 3: покнижные списки ──────────────────────────────────────────

    def test_loads_from_books_dir(self, tmp_path):
        """Если нет общих файлов — суммирует по книгам."""
        books_dir = tmp_path / "books" / "5 класс"
        books_dir.mkdir(parents=True)

        # Книга 1
        data1 = {"Слово": ["слово1", "слово2"], "Сумма слов": [10, 20]}
        pd.DataFrame(data1).to_excel(books_dir / "книга1.xlsx", index=False)

        # Книга 2 (слово1 повторяется)
        data2 = {"Слово": ["слово1", "слово3"], "Сумма слов": [5, 15]}
        pd.DataFrame(data2).to_excel(books_dir / "книга2.xlsx", index=False)

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=tmp_path / "fallback",
            common_books_dir=tmp_path / "books",
        )

        result = load_common_kv_for_grade(5, config)
        assert result["слово1"] == 15.0  # 10 + 5
        assert result["слово2"] == 20.0
        assert result["слово3"] == 15.0

    # ── Кэширование ───────────────────────────────────────────────────────────

    def test_caches_results(self, tmp_path):
        """Повторный вызов не перечитывает файлы (возвращает из кэша)."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["кэш"], "Сумма слов": [100]}
        file_path = fallback_dir / "Общий частотный список 5 класс.xlsx"
        pd.DataFrame(data).to_excel(file_path, index=False)

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        # Первый вызов
        result1 = load_common_kv_for_grade(5, config)

        # Удаляем файл
        file_path.unlink()

        # Второй вызов — должен вернуть из кэша, не упасть
        result2 = load_common_kv_for_grade(5, config)
        assert result1 == result2
        assert "кэш" in result2

    def test_cache_is_grade_specific(self, tmp_path):
        """Кэш разделён по классам."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        # Класс 5
        data5 = {"Слово": ["класс5"], "Сумма слов": [100]}
        pd.DataFrame(data5).to_excel(
            fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
        )

        # Класс 6
        data6 = {"Слово": ["класс6"], "Сумма слов": [200]}
        pd.DataFrame(data6).to_excel(
            fallback_dir / "Общий частотный список 6 класс.xlsx", index=False
        )

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        result5 = load_common_kv_for_grade(5, config)
        result6 = load_common_kv_for_grade(6, config)

        assert "класс5" in result5
        assert "класс6" in result6
        assert "класс6" not in result5

    # ── Пустой результат ──────────────────────────────────────────────────────

    def test_returns_empty_dict_when_no_data(self, tmp_path):
        """Возвращает пустой словарь, если данные не найдены."""
        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=tmp_path / "fallback",
            common_books_dir=tmp_path / "books",
        )
        result = load_common_kv_for_grade(99, config)
        assert result == {}

    # ── Нормализация ──────────────────────────────────────────────────────────

    def test_normalizes_words(self, tmp_path):
        """Слова нормализуются (lowercase, stripped)."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["  СЛОВО  ", "Кот"], "Сумма слов": [100, 50]}
        pd.DataFrame(data).to_excel(
            fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
        )

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        result = load_common_kv_for_grade(5, config)
        assert "слово" in result
        assert "кот" in result
        assert "СЛОВО" not in result


# ══════════════════════════════════════════════════════════════════════════════
# aggregated_common_kv_upto_grade
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatedCommonKV:
    """Тесты агрегации общеупотребительной лексики."""

    def setup_method(self):
        clear_common_kv_cache()

    def test_returns_dict_copy(self, tmp_path):
        """Возвращает копию словаря (не ссылку на кэш)."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["слово"], "Сумма слов": [100]}
        pd.DataFrame(data).to_excel(
            fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
        )

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        result = aggregated_common_kv_upto_grade(5, config)
        result["новое"] = 999  # модификация

        # Повторный вызов — не должен содержать «новое»
        result2 = aggregated_common_kv_upto_grade(5, config)
        assert "новое" not in result2

    def test_delegates_to_load_common_kv(self, tmp_path):
        """Функция — обёртка над load_common_kv_for_grade."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["тест"], "Сумма слов": [50]}
        pd.DataFrame(data).to_excel(
            fallback_dir / "Общий частотный список 7 класс.xlsx", index=False
        )

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        result = aggregated_common_kv_upto_grade(7, config)
        assert result == load_common_kv_for_grade(7, config)


# ══════════════════════════════════════════════════════════════════════════════
# load_textbook_freq_df
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadTextbookFreqDF:
    """Тесты загрузки частотных списков учебников."""

    def test_loads_valid_file(self, textbook_file):
        """Корректный файл загружается."""
        df = load_textbook_freq_df(textbook_file)
        assert df is not None
        assert len(df) == 3

    def test_has_required_columns(self, textbook_file):
        """Результат содержит обязательные столбцы."""
        df = load_textbook_freq_df(textbook_file)
        assert "Слово" in df.columns
        assert "Сумма слов" in df.columns

    def test_normalizes_words(self, textbook_file):
        """Слова нормализуются."""
        df = load_textbook_freq_df(textbook_file)
        words = df["Слово"].tolist()
        assert "днк" in words  # было "ДНК"

    def test_returns_none_on_missing_file(self, tmp_path):
        """None если файл не существует."""
        result = load_textbook_freq_df(tmp_path / "missing.xlsx")
        assert result is None

    def test_returns_none_on_missing_columns(self, tmp_path):
        """None если нет обязательных столбцов."""
        data = {"Столбец": [1, 2, 3]}
        path = tmp_path / "bad.xlsx"
        pd.DataFrame(data).to_excel(path, index=False)

        result = load_textbook_freq_df(path)
        assert result is None

    def test_removes_empty_words(self, tmp_path):
        """Пустые слова удаляются."""
        data = {"Слово": ["слово", "", np.nan], "Сумма слов": [10, 20, 30]}
        path = tmp_path / "с_пустыми.xlsx"
        pd.DataFrame(data).to_excel(path, index=False)

        df = load_textbook_freq_df(path)
        assert len(df) == 1
        assert df.iloc[0]["Слово"] == "слово"

    def test_converts_invalid_kv_to_zero(self, tmp_path):
        """NaN в «Сумма слов» заменяется на 0.0."""
        data = {"Слово": ["слово"], "Сумма слов": [np.nan]}
        path = tmp_path / "с_nan.xlsx"
        pd.DataFrame(data).to_excel(path, index=False)

        df = load_textbook_freq_df(path)
        assert df.iloc[0]["Сумма слов"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# clear_common_kv_cache
# ══════════════════════════════════════════════════════════════════════════════

class TestClearCache:
    """Тесты очистки кэша."""

    def test_clears_cache(self, tmp_path):
        """Кэш действительно очищается."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        data = {"Слово": ["кэш"], "Сумма слов": [100]}
        file_path = fallback_dir / "Общий частотный список 5 класс.xlsx"
        pd.DataFrame(data).to_excel(file_path, index=False)

        config = Config(
            base_dir=tmp_path / "base",
            common_fallback_dir=fallback_dir,
            common_books_dir=tmp_path / "books",
        )

        # Загружаем в кэш
        load_common_kv_for_grade(5, config)

        # Очищаем кэш
        clear_common_kv_cache()

        # Удаляем файл
        file_path.unlink()

        # Теперь должен вернуть пустой словарь (нет файла, кэш пуст)
        result = load_common_kv_for_grade(5, config)
        assert result == {}