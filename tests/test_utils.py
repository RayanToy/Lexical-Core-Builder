"""
Тесты для src/utils.py

Покрываем:
- normalize_word
- get_grade_from_dirname
- find_class_dirs
- latest_file_by_mtime
- extract_total_tokens_from_filename
- list_textbook_files
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    extract_total_tokens_from_filename,
    find_class_dirs,
    get_grade_from_dirname,
    latest_file_by_mtime,
    list_textbook_files,
    normalize_word,
)


# ══════════════════════════════════════════════════════════════════════════════
# normalize_word
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeWord:
    """Тесты нормализации слов."""

    def test_lowercase(self):
        assert normalize_word("Слово") == "слово"

    def test_all_caps(self):
        assert normalize_word("ПРИВЕТ") == "привет"

    def test_strip_leading_spaces(self):
        assert normalize_word("  слово") == "слово"

    def test_strip_trailing_spaces(self):
        assert normalize_word("слово  ") == "слово"

    def test_strip_both_sides(self):
        assert normalize_word("  Слово  ") == "слово"

    def test_already_normalized(self):
        assert normalize_word("слово") == "слово"

    def test_empty_string(self):
        assert normalize_word("") == ""

    def test_spaces_only(self):
        """Строка только из пробелов -> пустая строка."""
        assert normalize_word("   ") == ""

    def test_mixed_case(self):
        assert normalize_word("ПрИвЕт") == "привет"

    def test_numpy_nan_passthrough(self):
        """numpy.nan возвращается без изменений."""
        result = normalize_word(np.nan)
        assert pd.isna(result)

    def test_pandas_na_passthrough(self):
        """pandas.NA возвращается без изменений."""
        result = normalize_word(pd.NA)
        assert pd.isna(result)

    def test_none_passthrough(self):
        """None считается NA и возвращается без изменений."""
        result = normalize_word(None)
        assert pd.isna(result)

    def test_number_converted_to_string(self):
        """Числа конвертируются в строку."""
        assert normalize_word(42) == "42"

    @pytest.mark.parametrize("input_val,expected", [
        ("Кот", "кот"),
        ("  Пёс  ", "пёс"),
        ("БИОЛОГИЯ", "биология"),
        ("математика", "математика"),
    ])
    def test_parametrized(self, input_val, expected):
        assert normalize_word(input_val) == expected


# ══════════════════════════════════════════════════════════════════════════════
# get_grade_from_dirname
# ══════════════════════════════════════════════════════════════════════════════

class TestGetGradeFromDirname:
    """Тесты извлечения номера класса из имени директории."""

    # ── Корректные имена ──────────────────────────────────────────────────────

    def test_single_digit_grade(self):
        assert get_grade_from_dirname("5 класс списки") == 5

    def test_double_digit_grade(self):
        assert get_grade_from_dirname("10 класс списки") == 10

    def test_grade_1(self):
        assert get_grade_from_dirname("1 класс списки") == 1

    def test_grade_11(self):
        assert get_grade_from_dirname("11 класс списки") == 11

    def test_extra_spaces_around(self):
        """Пробелы вокруг всей строки допустимы."""
        assert get_grade_from_dirname("  7 класс списки  ") == 7

    def test_case_insensitive(self):
        """Регистр не важен."""
        assert get_grade_from_dirname("5 Класс Списки") == 5

    # ── Некорректные имена ────────────────────────────────────────────────────

    def test_no_match_returns_none(self):
        assert get_grade_from_dirname("общий список") is None

    def test_missing_word_spiski(self):
        """Без слова 'списки' — не совпадает."""
        assert get_grade_from_dirname("5 класс") is None

    def test_missing_word_klass(self):
        assert get_grade_from_dirname("5 списки") is None

    def test_empty_string(self):
        assert get_grade_from_dirname("") is None

    def test_just_number(self):
        assert get_grade_from_dirname("5") is None

    def test_subject_folder_name(self):
        """Имена предметных папок не должны совпадать."""
        assert get_grade_from_dirname("Bi") is None
        assert get_grade_from_dirname("Математический блок") is None

    @pytest.mark.parametrize("dirname,expected", [
        ("1 класс списки", 1),
        ("5 класс списки", 5),
        ("10 класс списки", 10),
        ("11 класс списки", 11),
        ("не класс", None),
        ("", None),
        ("класс списки", None),
    ])
    def test_parametrized(self, dirname, expected):
        assert get_grade_from_dirname(dirname) == expected


# ══════════════════════════════════════════════════════════════════════════════
# find_class_dirs
# ══════════════════════════════════════════════════════════════════════════════

class TestFindClassDirs:
    """Тесты поиска директорий классов."""

    def test_finds_all_grade_dirs(self, tmp_path):
        """Находит все директории в формате «N класс списки»."""
        (tmp_path / "5 класс списки").mkdir()
        (tmp_path / "6 класс списки").mkdir()
        (tmp_path / "7 класс списки").mkdir()

        result = find_class_dirs(tmp_path)

        assert len(result) == 3
        grades = [g for g, _ in result]
        assert grades == [5, 6, 7]

    def test_ignores_non_grade_dirs(self, tmp_path):
        """Игнорирует папки с неподходящим именем."""
        (tmp_path / "5 класс списки").mkdir()
        (tmp_path / "CombinedLexicalCores").mkdir()
        (tmp_path / "Общие списки").mkdir()

        result = find_class_dirs(tmp_path)

        assert len(result) == 1
        assert result[0][0] == 5

    def test_sorted_by_grade(self, tmp_path):
        """Результат отсортирован по возрастанию класса."""
        (tmp_path / "10 класс списки").mkdir()
        (tmp_path / "5 класс списки").mkdir()
        (tmp_path / "1 класс списки").mkdir()

        result = find_class_dirs(tmp_path)
        grades = [g for g, _ in result]

        assert grades == sorted(grades)

    def test_returns_paths(self, tmp_path):
        """Возвращает Path-объекты."""
        (tmp_path / "5 класс списки").mkdir()

        result = find_class_dirs(tmp_path)
        _, path = result[0]

        assert isinstance(path, Path)
        assert path.exists()

    def test_empty_dir_returns_empty_list(self, tmp_path):
        """Пустая директория — пустой результат."""
        result = find_class_dirs(tmp_path)
        assert result == []

    def test_files_ignored(self, tmp_path):
        """Файлы игнорируются, ищем только директории."""
        (tmp_path / "5 класс списки.txt").touch()
        result = find_class_dirs(tmp_path)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# latest_file_by_mtime
# ══════════════════════════════════════════════════════════════════════════════

class TestLatestFileByMtime:
    """Тесты поиска самого свежего файла."""

    def test_returns_none_for_empty_list(self):
        assert latest_file_by_mtime([]) is None

    def test_single_file_returned(self, tmp_path):
        f = tmp_path / "file.xlsx"
        f.touch()
        result = latest_file_by_mtime([f])
        assert result == f

    def test_returns_newest_file(self, tmp_path):
        """Возвращает файл с самым поздним mtime."""
        old = tmp_path / "old.xlsx"
        new = tmp_path / "new.xlsx"

        old.touch()
        time.sleep(0.05)   # гарантируем разницу в mtime
        new.touch()

        result = latest_file_by_mtime([old, new])
        assert result == new

    def test_accepts_strings(self, tmp_path):
        """Принимает как Path, так и строки."""
        f = tmp_path / "file.xlsx"
        f.touch()
        result = latest_file_by_mtime([str(f)])
        assert result == f

    def test_returns_path_object(self, tmp_path):
        """Всегда возвращает Path, даже если на входе строка."""
        f = tmp_path / "file.xlsx"
        f.touch()
        result = latest_file_by_mtime([str(f)])
        assert isinstance(result, Path)


# ══════════════════════════════════════════════════════════════════════════════
# extract_total_tokens_from_filename
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractTotalTokens:
    """Тесты извлечения количества токенов из имени файла."""

    def test_standard_format(self):
        path = Path("учебник_биология_46852.xlsx")
        assert extract_total_tokens_from_filename(path) == 46852

    def test_large_number(self):
        path = Path("corpus_123456789.xlsx")
        assert extract_total_tokens_from_filename(path) == 123456789

    def test_small_number(self):
        path = Path("mini_42.xlsx")
        assert extract_total_tokens_from_filename(path) == 42

    def test_no_number_returns_none(self):
        path = Path("без_числа.xlsx")
        assert extract_total_tokens_from_filename(path) is None

    def test_full_path_uses_only_filename(self):
        """Число берётся из имени файла, не из пути."""
        path = Path("/data/12345/учебник_99.xlsx")
        assert extract_total_tokens_from_filename(path) == 99

    def test_multiple_numbers_takes_last(self):
        """Берём число непосредственно перед .xlsx."""
        path = Path("5_класс_учебник_500.xlsx")
        assert extract_total_tokens_from_filename(path) == 500

    @pytest.mark.parametrize("filename,expected", [
        ("file_100.xlsx", 100),
        ("bio_5_класс_46852.xlsx", 46852),
        ("без_числа.xlsx", None),
        ("123.xlsx", 123),
    ])
    def test_parametrized(self, filename, expected):
        result = extract_total_tokens_from_filename(Path(filename))
        assert result == expected


# ══════════════════════════════════════════════════════════════════════════════
# list_textbook_files
# ══════════════════════════════════════════════════════════════════════════════

class TestListTextbookFiles:
    """Тесты фильтрации файлов учебников."""

    def test_returns_textbook_files(self, tmp_path):
        """Возвращает обычные xlsx-файлы учебников."""
        (tmp_path / "учебник_46852.xlsx").touch()
        (tmp_path / "другой_учебник_12000.xlsx").touch()

        result = list_textbook_files(tmp_path)
        names = [f.name for f in result]

        assert "учебник_46852.xlsx" in names
        assert "другой_учебник_12000.xlsx" in names

    def test_excludes_obshiy_files(self, tmp_path):
        """Исключает файлы «Общий частотный список*»."""
        (tmp_path / "Общий частотный список Bi 5 класс.xlsx").touch()
        (tmp_path / "учебник_46852.xlsx").touch()

        result = list_textbook_files(tmp_path)
        names = [f.name for f in result]

        assert "Общий частотный список Bi 5 класс.xlsx" not in names
        assert "учебник_46852.xlsx" in names

    def test_excludes_lexical_core_files(self, tmp_path):
        """Исключает файлы «Лексическое ядро*»."""
        (tmp_path / "Лексическое ядро 5 класс Bi.xlsx").touch()
        (tmp_path / "учебник_46852.xlsx").touch()

        result = list_textbook_files(tmp_path)
        names = [f.name for f in result]

        assert "Лексическое ядро 5 класс Bi.xlsx" not in names

    def test_ignores_non_xlsx(self, tmp_path):
        """Не возвращает не-xlsx файлы."""
        (tmp_path / "данные.csv").touch()
        (tmp_path / "заметки.txt").touch()
        (tmp_path / "учебник_46852.xlsx").touch()

        result = list_textbook_files(tmp_path)

        assert all(f.suffix == ".xlsx" for f in result)

    def test_result_is_sorted(self, tmp_path):
        """Результат отсортирован по имени."""
        (tmp_path / "з_учебник_100.xlsx").touch()
        (tmp_path / "а_учебник_200.xlsx").touch()

        result = list_textbook_files(tmp_path)
        names = [f.name for f in result]

        assert names == sorted(names)

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = list_textbook_files(tmp_path)
        assert result == []

    def test_only_excluded_files_returns_empty(self, tmp_path):
        """Если все файлы исключены — пустой список."""
        (tmp_path / "Общий частотный список.xlsx").touch()
        (tmp_path / "Лексическое ядро.xlsx").touch()

        result = list_textbook_files(tmp_path)
        assert result == []