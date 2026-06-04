"""
Построение кумулятивных словарей ALL.

Модуль отвечает за:
- агрегацию лексических ядер всех предметов в единый словарь
- межклассное накопление (класс N включает данные классов 1..N)
- интеграцию общеупотребительной лексики
- сохранение сводных таблиц ALL по классам
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .loaders import load_common_kv_for_grade
from .utils import normalize_word


# ══════════════════════════════════════════════════════════════════════════════
# Структура кумулятивного состояния
# ══════════════════════════════════════════════════════════════════════════════

CumulativeState = dict[str, any]
"""
Тип кумулятивного состояния:
{
    "words": {
        word: {
            "kv_total": float,                  # сумма КВ по предметам (без внеклассной)
            "per_subject": dict[str, float],    # КВ по каждому предмету
            "textbooks": set[str],              # уникальные учебники "subject|tbname"
            "first_grade": int,                 # класс первого появления
            "domains": set[str],                # предметные области
        }
    },
    "subjects": set[str],  # все встреченные предметы
}
"""


def init_cumulative_all_state() -> CumulativeState:
    """
    Инициализирует пустое кумулятивное состояние.

    Returns:
        Словарь с пустыми структурами данных.

    Examples:
        >>> state = init_cumulative_all_state()
        >>> len(state["words"])
        0
        >>> len(state["subjects"])
        0
    """
    return {
        "words": {},
        "subjects": set(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Обновление состояния из предметных ядер
# ══════════════════════════════════════════════════════════════════════════════

def update_cumulative_all_state(
    state: CumulativeState,
    subject_core_paths: list[Path],
    grade: int,
    config: Config,
) -> None:
    """
    Обновляет кумулятивное состояние данными предметных ядер текущего класса.

    Читает Excel-файлы лексических ядер, извлекает:
    - слова и их частоты
    - предметные области
    - покнижные частоты (для подсчёта уникальных учебников)

    Args:
        state: кумулятивное состояние (изменяется in-place).
        subject_core_paths: список путей к файлам ядер предметов.
        grade: номер текущего класса.
        config: объект конфигурации.

    Examples:
        >>> state = init_cumulative_all_state()
        >>> update_cumulative_all_state(state, [Path("Bi_core.xlsx")], 5, config)
        >>> len(state["words"]) > 0
        True
    """
    for core_path in subject_core_paths:
        try:
            df = pd.read_excel(core_path)
        except Exception as e:
            print(f"  [ALL {grade}] Ошибка чтения {core_path.name}: {e}")
            continue

        # Извлечение названия предмета из пути
        # Структура: .../Subject/Общий частотный список/Лексическое ядро...
        subject_name = core_path.parent.parent.name
        state["subjects"].add(subject_name)

        # Проверка обязательных столбцов
        if "Слово" not in df.columns or "КВ" not in df.columns:
            print(
                f"  [ALL {grade}] В {core_path.name} нет столбцов 'Слово'/'КВ'. Пропуск."
            )
            continue

        # Нормализация данных
        df = df.copy()
        df["Слово"] = df["Слово"].map(normalize_word)
        df["КВ"] = pd.to_numeric(df["КВ"], errors="coerce").fillna(0.0)

        # Столбец предметной области (если есть)
        domain_col = "Предметная область"
        if domain_col in df.columns:
            df[domain_col] = df[domain_col].astype(str).str.strip()
        else:
            df[domain_col] = subject_name

        # Покнижные столбцы КВ_* (для подсчёта учебников)
        tb_cols = [c for c in df.columns if c.startswith("КВ_") and c != "КВ"]
        # Исключаем служебные файлы
        bad_markers = ("общий частотный список", "лексическое ядро")
        tb_cols = [
            c for c in tb_cols
            if not any(m in c[len("КВ_"):].lower() for m in bad_markers)
        ]

        for col in tb_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Обработка каждого слова
        for _, row in df.iterrows():
            word = row["Слово"]
            if not isinstance(word, str) or not word:
                continue

            # Получение или создание записи слова
            if word not in state["words"]:
                state["words"][word] = {
                    "kv_total": 0.0,
                    "per_subject": defaultdict(float),
                    "textbooks": set(),
                    "first_grade": grade,
                    "domains": set(),
                }

            entry = state["words"][word]

            # Обновление минимального класса
            if entry["first_grade"] is None or grade < entry["first_grade"]:
                entry["first_grade"] = grade

            # КВ по предмету
            kv = float(row["КВ"])
            entry["kv_total"] += kv
            entry["per_subject"][subject_name] += kv

            # Предметная область
            domain = str(row.get(domain_col, subject_name))
            entry["domains"].add(domain)

            # Уникальные учебники
            for col in tb_cols:
                if row[col] > 0:
                    tb_name = col[len("КВ_"):]
                    entry["textbooks"].add(f"{subject_name}|{tb_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Обновление состояния общеупотребительной лексикой
# ══════════════════════════════════════════════════════════════════════════════

def update_cumulative_all_with_extracurricular(
    state: CumulativeState,
    grade: int,
    config: Config,
) -> None:
    """
    Добавляет общеупотребительную лексику (внеклассное чтение) в кумулятивное состояние.

    ВАЖНО: общие списки внеклассной литературы уже кумулятивны (включают предыдущие классы).
    Поэтому НЕ суммируем между классами — просто обновляем значение для текущего класса.

    Args:
        state: кумулятивное состояние (изменяется in-place).
        grade: номер текущего класса.
        config: объект конфигурации.

    Examples:
        >>> state = init_cumulative_all_state()
        >>> update_cumulative_all_with_extracurricular(state, 5, config)
        >>> config.extracurricular_tag in state["subjects"]
        True
    """
    kv_map = load_common_kv_for_grade(grade, config)
    if not kv_map:
        return

    state["subjects"].add(config.extracurricular_tag)

    for word, kv in kv_map.items():
        word_norm = normalize_word(word)
        if not word_norm:
            continue

        # Получение или создание записи
        if word_norm not in state["words"]:
            state["words"][word_norm] = {
                "kv_total": 0.0,  # внеклассная НЕ входит в kv_total
                "per_subject": defaultdict(float),
                "textbooks": set(),
                "first_grade": grade,
                "domains": set(),
            }

        entry = state["words"][word_norm]

        # Обновление минимального класса
        if entry["first_grade"] is None or grade < entry["first_grade"]:
            entry["first_grade"] = grade

        # Перезаписываем КВ внеклассной литературы (не суммируем!)
        kv_f = float(kv)
        entry["per_subject"][config.extracurricular_tag] = kv_f

        if kv_f > 0:
            entry["domains"].add(config.extracurricular_tag)


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение кумулятивного ALL
# ══════════════════════════════════════════════════════════════════════════════

def save_cumulative_all_for_grade(
    grade: int,
    state: CumulativeState,
    config: Config,
) -> Optional[Path]:
    """
    Сохраняет кумулятивный ALL-словарь для класса.

    Создаёт Excel-файл с колонками:
    - Слово
    - Предметные области (через запятую)
    - КВ (сумма слов) — суммарный КВ по предметам (без внеклассной)
    - Количество учебников
    - Появился в классе
    - КВ (<предмет>) — для каждого предмета и внеклассной литературы

    Файл сортируется по убыванию КВ.

    Args:
        grade: номер класса.
        state: кумулятивное состояние.
        config: объект конфигурации.

    Returns:
        Путь к сохранённому файлу или None при ошибке.

    Examples:
        >>> state = init_cumulative_all_state()
        >>> path = save_cumulative_all_for_grade(5, state, config)
        >>> path.exists() if path else True
        True
    """
    words_state = state["words"]
    subjects = sorted(state["subjects"])

    # Формирование строк таблицы
    rows = []
    for word, entry in words_state.items():
        row = {
            "Слово": word,
            "Предметные области": ", ".join(sorted(entry["domains"])),
            "КВ (сумма слов)": float(entry["kv_total"]),
            "Количество учебников": int(len(entry["textbooks"])),
            "Появился в классе": (
                int(entry["first_grade"])
                if entry["first_grade"] is not None
                else np.nan
            ),
        }

        # КВ по каждому предмету
        for subj in subjects:
            row[f"КВ ({subj})"] = float(entry["per_subject"].get(subj, 0.0))

        rows.append(row)

    # Создание DataFrame
    if not rows:
        print(f"  [ALL {grade}] Нет данных для сохранения (кумулятив пуст).")
        result = pd.DataFrame(
            columns=[
                "Слово",
                "Предметные области",
                "КВ (сумма слов)",
                "Количество учебников",
                "Появился в классе",
            ]
        )
    else:
        result = pd.DataFrame(rows)
        result = result.sort_values(
            by="КВ (сумма слов)", ascending=False
        ).reset_index(drop=True)

    # Сохранение
    out_dir = config.combined_output_root / f"{grade} класс"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"Лексическое ядро {grade} класс ALL.xlsx"

    try:
        result.to_excel(out_path, index=False)
        print(f"  ✓ [ALL {grade}] Сохранён: {out_path}")
        return out_path
    except Exception as e:
        print(f"  ✗ [ALL {grade}] Ошибка сохранения {out_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def get_cumulative_stats(state: CumulativeState) -> dict[str, any]:
    """
    Возвращает статистику кумулятивного состояния.

    Args:
        state: кумулятивное состояние.

    Returns:
        Словарь со статистикой:
        - total_words: общее количество слов
        - total_subjects: количество предметов
        - words_by_grade: словарь {класс: количество слов}

    Examples:
        >>> state = init_cumulative_all_state()
        >>> stats = get_cumulative_stats(state)
        >>> stats["total_words"]
        0
    """
    words_by_grade = defaultdict(int)
    for entry in state["words"].values():
        grade = entry.get("first_grade")
        if grade is not None:
            words_by_grade[grade] += 1

    return {
        "total_words": len(state["words"]),
        "total_subjects": len(state["subjects"]),
        "words_by_grade": dict(words_by_grade),
    }


def print_cumulative_summary(state: CumulativeState, grade: int) -> None:
    """
    Выводит краткую сводку кумулятивного состояния.

    Args:
        state: кумулятивное состояние.
        grade: текущий класс.

    Examples:
        >>> state = init_cumulative_all_state()
        >>> print_cumulative_summary(state, 5)
        [ALL 5] Всего слов: 0 | Предметов: 0
    """
    stats = get_cumulative_stats(state)
    print(
        f"  [ALL {grade}] Всего слов: {stats['total_words']} | "
        f"Предметов: {stats['total_subjects']}"
    )

    if stats["words_by_grade"]:
        grade_dist = ", ".join(
            f"{g}кл: {count}" for g, count in sorted(stats["words_by_grade"].items())
        )
        print(f"  [ALL {grade}] Распределение по классам: {grade_dist}")