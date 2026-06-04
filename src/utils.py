"""
Вспомогательные функции:
- нормализация слов
- поиск директорий классов и предметных папок
- работа с файлами (поиск, фильтрация, извлечение метаданных)
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ─── Нормализация ─────────────────────────────────────────────────────────────

def normalize_word(s: object) -> object:
    """
    Приводит слово к каноническому виду: нижний регистр, обрезка пробелов.

    Возвращает исходное значение без изменений, если оно является NaN
    (чтобы pandas-цепочки не ломались на пустых ячейках).

    Args:
        s: строка или NaN-подобное значение.

    Returns:
        Нормализованная строка или исходный NaN.

    Examples:
        >>> normalize_word("  Слово  ")
        'слово'
        >>> normalize_word("ПРИВЕТ")
        'привет'
        >>> import numpy as np
        >>> normalize_word(np.nan) is np.nan
        True
    """
    if pd.isna(s):
        return s
    return str(s).strip().lower()


# ─── Директории классов ───────────────────────────────────────────────────────

def get_grade_from_dirname(dirname: str) -> Optional[int]:
    """
    Извлекает номер класса из имени директории вида «N класс списки».

    Args:
        dirname: имя папки (не полный путь).

    Returns:
        Номер класса как int или None, если формат не совпал.

    Examples:
        >>> get_grade_from_dirname("5 класс списки")
        5
        >>> get_grade_from_dirname("10 класс списки")
        10
        >>> get_grade_from_dirname("общий список") is None
        True
    """
    match = re.match(
        r"^\s*(\d+)\s+класс\s+списки\s*$",
        dirname,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def find_class_dirs(base_dir: Path) -> list[tuple[int, Path]]:
    """
    Находит все директории классов внутри base_dir.

    Ищет папки с именем «N класс списки», возвращает список
    отсортированных пар (номер_класса, путь).

    Args:
        base_dir: корневая директория с папками классов.

    Returns:
        Список пар (grade, path), отсортированных по возрастанию класса.
        Пустой список, если подходящих папок нет.

    Examples:
        >>> # Предполагаем структуру: base_dir/5 класс списки/, base_dir/6 класс списки/
        >>> dirs = find_class_dirs(Path("/data"))
        >>> dirs[0][0]  # первый класс
        5
    """
    result = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            grade = get_grade_from_dirname(entry.name)
            if grade is not None:
                result.append((grade, Path(entry.path)))
    result.sort(key=lambda x: x[0])
    return result


# ─── Предметные папки ─────────────────────────────────────────────────────────

def find_subject_folders(
    class_dir: Path,
    allowed_subjects: set[str],
    excluded_block_names: set[str],
) -> list[Path]:
    """
    Находит предметные папки внутри директории класса.

    Папка считается предметной если:
    - её имя входит в allowed_subjects,
    - она не является блоком-агрегатором (excluded_block_names),
    - внутри есть подпапка «Общий частотный список»
      с файлом «Общий частотный список*.xlsx»
      (не начинающимся с «Лексическое ядро»).

    Args:
        class_dir: директория конкретного класса.
        allowed_subjects: множество разрешённых кодов предметов.
        excluded_block_names: множество имён блоков для исключения.

    Returns:
        Список путей к предметным папкам.
    """
    subject_folders = []

    for dirpath, dirnames, _ in os.walk(class_dir):
        base = os.path.basename(dirpath)

        # Пропускаем служебные и блочные папки
        if base in {"Общий частотный список", "Общеупотребительная лексика"}:
            continue
        if base in excluded_block_names:
            continue

        # Проверяем наличие нужного подкаталога и файла
        freq_dir = os.path.join(dirpath, "Общий частотный список")
        if not os.path.isdir(freq_dir):
            continue

        candidates = [
            c for c in glob.glob(
                os.path.join(freq_dir, "Общий частотный список*.xlsx")
            )
            if not os.path.basename(c).startswith("Лексическое ядро")
        ]

        if candidates and base in allowed_subjects:
            subject_folders.append(Path(dirpath))

    return subject_folders


# ─── Работа с файлами ─────────────────────────────────────────────────────────

def latest_file_by_mtime(paths: list[str | Path]) -> Optional[Path]:
    """
    Возвращает путь к файлу с максимальным временем модификации.

    Args:
        paths: список путей к файлам.

    Returns:
        Path к самому свежему файлу или None, если список пуст.

    Examples:
        >>> latest_file_by_mtime([])  # None
        >>> latest_file_by_mtime(["/a/old.xlsx", "/a/new.xlsx"])
        PosixPath('/a/new.xlsx')  # зависит от mtime
    """
    if not paths:
        return None
    return Path(max(paths, key=lambda p: os.path.getmtime(p)))


def find_overall_subject_freq_file(subject_folder: Path) -> Optional[Path]:
    """
    Ищет общий частотный список предмета.

    Путь: subject_folder/Общий частотный список/Общий частотный список*.xlsx
    Из кандидатов исключаются файлы, начинающиеся с «Лексическое ядро».
    Возвращает самый свежий файл.

    Args:
        subject_folder: папка предмета.

    Returns:
        Path к файлу или None, если файл не найден.
    """
    freq_dir = subject_folder / "Общий частотный список"
    if not freq_dir.is_dir():
        return None

    candidates = [
        c for c in freq_dir.glob("Общий частотный список*.xlsx")
        if not c.name.startswith("Лексическое ядро")
    ]
    return latest_file_by_mtime(candidates)


def list_textbook_files(subject_folder: Path) -> list[Path]:
    """
    Возвращает файлы учебников из корня предметной папки.

    Исключает:
    - файлы, начинающиеся с «Общий частотный список»
    - файлы, начинающиеся с «Лексическое ядро»

    Args:
        subject_folder: папка предмета.

    Returns:
        Отсортированный список путей к файлам учебников.
    """
    result = []
    for f in subject_folder.glob("*.xlsx"):
        if f.name.startswith("Общий частотный список"):
            continue
        if f.name.startswith("Лексическое ядро"):
            continue
        result.append(f)
    return sorted(result)


def extract_total_tokens_from_filename(path: Path) -> Optional[int]:
    """
    Извлекает общее количество слов из имени файла учебника.

    Предполагаемый формат имени: «...46852.xlsx»
    Число непосредственно перед расширением — количество токенов.

    Args:
        path: путь к файлу учебника.

    Returns:
        Количество токенов как int или None, если не удалось извлечь.

    Examples:
        >>> extract_total_tokens_from_filename(Path("учебник_биология_46852.xlsx"))
        46852
        >>> extract_total_tokens_from_filename(Path("без_числа.xlsx")) is None
        True
    """
    match = re.search(r"(\d+)(?=\.xlsx$)", path.name)
    if match:
        return int(match.group(1))
    return None