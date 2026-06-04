"""
Загрузка и валидация данных из Excel-файлов.

Модуль отвечает за:
- загрузку общих частотных списков предметов
- загрузку общеупотребительной лексики (с кэшированием)
- валидацию структуры данных
- нормализацию типов и дедупликацию
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config
from .utils import latest_file_by_mtime, normalize_word


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка общих частотных списков предметов
# ══════════════════════════════════════════════════════════════════════════════

def load_overall_subject_freq_df(path: Path, subject_name: str = "") -> pd.DataFrame:
    """
    Загружает общий частотный список предмета и нормализует столбцы.

    Обязательные столбцы (после нормализации):
    - «Слово»
    - «Сумма слов»
    - «Нормализованная частотность»

    Поддерживается вариант названия «Нормализованная частота» (автозамена).

    Args:
        path: путь к Excel-файлу с общим списком.
        subject_name: имя предмета (для информативных ошибок).

    Returns:
        DataFrame с нормализованными столбцами:
        - «Слово» (str, lowercase, stripped, без дубликатов)
        - «Сумма слов» (float)
        - «Нормализованная частотность» (float)

    Raises:
        FileNotFoundError: если файл не существует.
        ValueError: если отсутствуют обязательные столбцы.
        pd.errors.EmptyDataError: если файл пустой.

    Examples:
        >>> df = load_overall_subject_freq_df(Path("Bi_5.xlsx"), "Bi")
        >>> "Слово" in df.columns
        True
        >>> df["Слово"].dtype
        dtype('O')  # object (string)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Файл общего частотного списка не найден: {path}"
        )

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise ValueError(
            f"Ошибка чтения файла {path} ({subject_name}): {e}"
        ) from e

    if df.empty:
        raise pd.errors.EmptyDataError(
            f"Файл {path} ({subject_name}) не содержит данных."
        )

    # Нормализация названий столбцов
    rename_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower == "нормализованная частота":
            rename_map[col] = "Нормализованная частотность"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Проверка обязательных столбцов
    required = ["Слово", "Сумма слов", "Нормализованная частотность"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Файл {path} ({subject_name}) не содержит обязательных столбцов: {missing}.\n"
            f"Доступные столбцы: {list(df.columns)}"
        )

    # Нормализация данных
    df = df.copy()
    df["Слово"] = df["Слово"].map(normalize_word)
    df["Сумма слов"] = pd.to_numeric(df["Сумма слов"], errors="coerce").fillna(0.0)
    df["Нормализованная частотность"] = pd.to_numeric(
        df["Нормализованная частотность"], errors="coerce"
    ).fillna(0.0)

    # Удаление пустых и дублирующихся слов
    df = df.dropna(subset=["Слово"])
    df = df[df["Слово"] != ""]
    df = df.drop_duplicates(subset=["Слово"], keep="first")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка общеупотребительной лексики (с кэшированием)
# ══════════════════════════════════════════════════════════════════════════════

# Кэш: {класс: {слово: КВ}}
_common_kv_cache: dict[int, dict[str, float]] = {}


def load_common_kv_for_grade(grade: int, config: Config) -> dict[str, float]:
    """
    Загружает общеупотребительную лексику (внеклассное чтение) для класса.

    Возвращает словарь {слово: «Сумма слов»}.
    Результат кэшируется — повторные вызовы для того же класса
    не перечитывают файлы.

    Источники данных (по приоритету):
    1. base_dir/<класс> класс списки/Общеупотребительная лексика/
       Общий частотный список*.xlsx
    2. common_fallback_dir/Общий частотный список <класс> класс.xlsx
    3. common_books_dir/<класс> класс/*.xlsx — суммирование по книгам

    Args:
        grade: номер класса (1–11).
        config: объект конфигурации.

    Returns:
        Словарь {слово: КВ}, где слова нормализованы (lowercase, stripped).
        Пустой словарь, если данные не найдены.

    Examples:
        >>> kv_map = load_common_kv_for_grade(5, config)
        >>> kv_map.get("кот", 0.0)
        15.0  # пример значения
    """
    # Проверка кэша
    if grade in _common_kv_cache:
        return _common_kv_cache[grade]

    kv_map = {}

    # ── Источник 1: основная папка класса ────────────────────────────────────
    primary_dir = config.base_dir / f"{grade} класс списки" / "Общеупотребительная лексика"
    if primary_dir.is_dir():
        candidates = list(primary_dir.glob("Общий частотный список*.xlsx"))
        if candidates:
            path = latest_file_by_mtime(candidates)
            try:
                kv_map = _load_common_kv_from_file(path)
                if kv_map:
                    _common_kv_cache[grade] = kv_map
                    return kv_map
            except Exception as e:
                print(f"[Общеупотр. {grade}] Ошибка чтения {path}: {e}")

    # ── Источник 2: резервная папка ───────────────────────────────────────────
    fallback_exact = config.common_fallback_dir / f"Общий частотный список {grade} класс.xlsx"
    candidates = []

    if fallback_exact.exists():
        candidates = [fallback_exact]
    else:
        pattern = f"Общий частотный список *{grade}*класс*.xlsx"
        candidates = list(config.common_fallback_dir.glob(pattern))

    if candidates:
        path = latest_file_by_mtime(candidates)
        try:
            kv_map = _load_common_kv_from_file(path)
            if kv_map:
                _common_kv_cache[grade] = kv_map
                return kv_map
        except Exception as e:
            print(f"[Общеупотр. {grade} резерв] Ошибка чтения {path}: {e}")

    # ── Источник 3: покнижные списки ──────────────────────────────────────────
    books_dir = config.common_books_dir / f"{grade} класс"
    if books_dir.is_dir():
        kv_map = _load_common_kv_from_books(books_dir, grade)

    # Кэшируем даже пустой результат (чтобы не повторять поиск)
    _common_kv_cache[grade] = kv_map
    return kv_map


def _load_common_kv_from_file(path: Path) -> dict[str, float]:
    """
    Читает один общий частотный список внеклассной литературы.

    Ожидается структура: столбцы «Слово» и «Сумма слов».

    Args:
        path: путь к Excel-файлу.

    Returns:
        Словарь {слово: КВ} с нормализованными ключами.
        Пустой словарь при ошибке чтения или отсутствии нужных столбцов.
    """
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[Общеупотр.] Не удалось прочитать {path}: {e}")
        return {}

    if "Слово" not in df.columns or "Сумма слов" not in df.columns:
        print(
            f"[Общеупотр.] Файл {path} не содержит столбцов «Слово» и/или «Сумма слов». "
            f"Доступные: {list(df.columns)}"
        )
        return {}

    df = df[["Слово", "Сумма слов"]].dropna(subset=["Слово"])
    df["Слово"] = df["Слово"].map(normalize_word)
    df["Сумма слов"] = pd.to_numeric(df["Сумма слов"], errors="coerce").fillna(0.0)

    # Отфильтровываем пустые слова
    df = df[df["Слово"] != ""]

    return {w: float(kv) for w, kv in zip(df["Слово"], df["Сумма слов"])}


def _load_common_kv_from_books(books_dir: Path, grade: int) -> dict[str, float]:
    """
    Читает все покнижные списки из директории и суммирует «Сумма слов».

    Args:
        books_dir: директория с файлами книг (напр., «5 класс/»).
        grade: номер класса (для логирования).

    Returns:
        Словарь {слово: суммарный КВ}.
    """
    book_files = list(books_dir.glob("*.xlsx"))
    if not book_files:
        return {}

    accumulator = defaultdict(float)

    for book_path in book_files:
        try:
            df = pd.read_excel(book_path)
        except Exception as e:
            print(f"[Общеупотр. {grade} книги] Ошибка чтения {book_path}: {e}")
            continue

        if "Слово" not in df.columns or "Сумма слов" not in df.columns:
            continue

        df = df[["Слово", "Сумма слов"]].dropna(subset=["Слово"])
        df["Слово"] = df["Слово"].map(normalize_word)
        df["Сумма слов"] = pd.to_numeric(df["Сумма слов"], errors="coerce").fillna(0.0)
        df = df[df["Слово"] != ""]

        for word, kv in zip(df["Слово"], df["Сумма слов"]):
            accumulator[word] += float(kv)

    return dict(accumulator)


def aggregated_common_kv_upto_grade(grade: int, config: Config) -> dict[str, float]:
    """
    Возвращает общеупотребительную лексику для класса.

    ВАЖНО: общие списки по внеклассной литературе уже кумулятивны
    внутри класса (включают предыдущие классы). Поэтому не суммируем 1..grade,
    а берём только текущий класс.

    Args:
        grade: номер класса.
        config: объект конфигурации.

    Returns:
        Словарь {слово: КВ} для текущего класса.

    Examples:
        >>> kv_map = aggregated_common_kv_upto_grade(5, config)
        >>> len(kv_map) > 0
        True
    """
    return dict(load_common_kv_for_grade(grade, config))


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка учебников
# ══════════════════════════════════════════════════════════════════════════════

def load_textbook_freq_df(path: Path) -> Optional[pd.DataFrame]:
    """
    Загружает частотный список конкретного учебника.

    Обязательные столбцы: «Слово», «Сумма слов».

    Args:
        path: путь к Excel-файлу учебника.

    Returns:
        DataFrame с нормализованными данными или None при ошибке.
        Столбцы:
        - «Слово» (str, нормализовано)
        - «Сумма слов» (float)

    Examples:
        >>> df = load_textbook_freq_df(Path("учебник_46852.xlsx"))
        >>> df is not None
        True
        >>> "Слово" in df.columns
        True
    """
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[Учебник] Ошибка чтения {path}: {e}")
        return None

    if "Слово" not in df.columns or "Сумма слов" not in df.columns:
        print(
            f"[Учебник] {path} не содержит столбцов «Слово» и/или «Сумма слов». "
            f"Доступные: {list(df.columns)}"
        )
        return None

    df = df.copy()
    df["Слово"] = df["Слово"].map(normalize_word)
    df["Сумма слов"] = pd.to_numeric(df["Сумма слов"], errors="coerce").fillna(0.0)

    # Удаляем пустые слова
    df = df.dropna(subset=["Слово"])
    df = df[df["Слово"] != ""]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Очистка кэша (для тестирования)
# ══════════════════════════════════════════════════════════════════════════════

def clear_common_kv_cache():
    """
    Очищает кэш общеупотребительной лексики.

    Полезно для тестов, когда нужно гарантировать перечитывание данных.
    """
    global _common_kv_cache
    _common_kv_cache = {}