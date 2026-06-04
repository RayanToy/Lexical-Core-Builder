"""
Построение лексического ядра по предмету.

Основной модуль, реализующий логику:
- отбор слов по порогам покрытия и частотности
- наследование ядер младших классов
- интеграцию общеупотребительной лексики
- вычисление метрик (покрытие, НЧ по учебникам, коэффициент Жуайна)
- формирование и сохранение итоговой таблицы
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .loaders import (
    aggregated_common_kv_upto_grade,
    load_overall_subject_freq_df,
    load_textbook_freq_df,
)
from .metrics import (
    calculate_juyna_coefficient,
    coverage_percent,
    normalized_frequency,
)
from .utils import (
    extract_total_tokens_from_filename,
    find_overall_subject_freq_file,
    list_textbook_files,
)


# ══════════════════════════════════════════════════════════════════════════════
# Структуры данных для хранения частотности по учебникам
# ══════════════════════════════════════════════════════════════════════════════

class TextbookFrequencies:
    """
    Хранит частотные данные слов по учебникам.

    Attributes:
        kv: dict[слово][учебник] = абсолютная частота
        nf: dict[слово][учебник] = нормализованная частотность
        coverage: dict[слово] = количество учебников, где слово встречается
        textbook_names: список названий учебников
    """

    def __init__(self):
        self.kv: dict[str, dict[str, float]] = defaultdict(dict)
        self.nf: dict[str, dict[str, float]] = defaultdict(dict)
        self.coverage: dict[str, int] = defaultdict(int)
        self.textbook_names: list[str] = []

    def add_textbook(
        self,
        textbook_name: str,
        words_freq: dict[str, float],
        total_tokens: int,
        norm_base: int,
    ):
        """
        Добавляет данные одного учебника.

        Args:
            textbook_name: название учебника (без расширения).
            words_freq: словарь {слово: абсолютная частота}.
            total_tokens: общее количество слов в учебнике.
            norm_base: база нормализации (обычно 1_000_000).
        """
        self.textbook_names.append(textbook_name)

        for word, freq in words_freq.items():
            if freq <= 0:
                continue

            # Абсолютная частота
            self.kv[word][textbook_name] = freq

            # Нормализованная частотность
            nf = normalized_frequency(freq, total_tokens, norm_base)
            self.nf[word][textbook_name] = nf

            # Покрытие
            self.coverage[word] += 1

    def get_kv(self, word: str, textbook: str) -> float:
        """Возвращает абсолютную частоту слова в учебнике."""
        return self.kv.get(word, {}).get(textbook, 0.0)

    def get_nf(self, word: str, textbook: str) -> float:
        """Возвращает нормализованную частотность слова в учебнике."""
        return self.nf.get(word, {}).get(textbook, 0.0)

    def get_coverage_count(self, word: str) -> int:
        """Возвращает количество учебников, содержащих слово."""
        return self.coverage.get(word, 0)

    def get_nf_values(self, word: str) -> list[float]:
        """Возвращает список НЧ слова по всем учебникам (для Жуайна)."""
        return [
            self.nf.get(word, {}).get(tb, 0.0)
            for tb in self.textbook_names
        ]


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка учебников предмета
# ══════════════════════════════════════════════════════════════════════════════

def load_textbooks_for_subject(
    subject_folder: Path,
    config: Config,
) -> tuple[TextbookFrequencies, list[Path], list[Path]]:
    """
    Загружает все учебники из папки предмета.

    Args:
        subject_folder: путь к папке предмета.
        config: объект конфигурации.

    Returns:
        Кортеж:
        - TextbookFrequencies: агрегированные данные по учебникам
        - список использованных файлов
        - список пропущенных файлов

    Examples:
        >>> freqs, used, skipped = load_textbooks_for_subject(Path("5 класс/Bi"), config)
        >>> len(freqs.textbook_names) > 0
        True
    """
    textbook_files = list_textbook_files(subject_folder)
    frequencies = TextbookFrequencies()
    used_files = []
    skipped_files = []

    for tb_path in textbook_files:
        # Извлечение количества токенов из имени файла
        total_tokens = extract_total_tokens_from_filename(tb_path)
        if not total_tokens or total_tokens <= 0:
            print(f"  [SKIP] Не удалось извлечь количество слов: {tb_path.name}")
            skipped_files.append(tb_path)
            continue

        # Загрузка данных учебника
        df = load_textbook_freq_df(tb_path)
        if df is None:
            skipped_files.append(tb_path)
            continue

        # Формирование словаря {слово: частота}
        words_freq = {}
        for _, row in df.iterrows():
            word = row["Слово"]
            freq = float(row["Сумма слов"])
            if freq < 0:
                continue
            words_freq[word] = words_freq.get(word, 0.0) + freq

        # Добавление в агрегированную структуру
        textbook_name = tb_path.stem  # имя без расширения
        frequencies.add_textbook(
            textbook_name=textbook_name,
            words_freq=words_freq,
            total_tokens=total_tokens,
            norm_base=config.norm_base,
        )

        used_files.append(tb_path)

    return frequencies, used_files, skipped_files


# ══════════════════════════════════════════════════════════════════════════════
# Построение базового ядра по порогам
# ══════════════════════════════════════════════════════════════════════════════

def build_base_core(
    freq_df: pd.DataFrame,
    textbook_freqs: TextbookFrequencies,
    config: Config,
) -> set[str]:
    """
    Строит базовое ядро предмета по порогам покрытия и частотности.

    Слово включается в базовое ядро если одновременно:
    - Покрытие >= coverage_threshold (%)
    - НЧ >= norm_freq_threshold

    Args:
        freq_df: общий частотный список предмета (из load_overall_subject_freq_df).
        textbook_freqs: частоты по учебникам.
        config: объект конфигурации.

    Returns:
        Множество слов базового ядра.

    Examples:
        >>> base = build_base_core(freq_df, textbook_freqs, config)
        >>> len(base) > 0
        True
    """
    base_core = set()
    total_textbooks = len(textbook_freqs.textbook_names)

    if total_textbooks == 0:
        return base_core

    for _, row in freq_df.iterrows():
        word = row["Слово"]
        if pd.isna(word) or word == "":
            continue

        # Покрытие
        cov_count = textbook_freqs.get_coverage_count(word)
        cov_pct = coverage_percent(word, textbook_freqs.coverage, total_textbooks)

        # Нормализованная частотность из общего списка
        nf = float(row["Нормализованная частотность"])

        # Проверка порогов
        if cov_pct >= config.coverage_threshold and nf >= config.norm_freq_threshold:
            base_core.add(word)

    return base_core


# ══════════════════════════════════════════════════════════════════════════════
# Фильтрация общеупотребительной лексики
# ══════════════════════════════════════════════════════════════════════════════

def filter_common_vocabulary(
    grade: int,
    config: Config,
) -> set[str]:
    """
    Возвращает слова общеупотребительной лексики, удовлетворяющие порогу.

    Фильтр: КВ > common_kv_min_threshold.

    Args:
        grade: номер класса.
        config: объект конфигурации.

    Returns:
        Множество слов.

    Examples:
        >>> common = filter_common_vocabulary(5, config)
        >>> isinstance(common, set)
        True
    """
    common_kv = aggregated_common_kv_upto_grade(grade, config)
    return {
        word for word, kv in common_kv.items()
        if word and kv > config.common_kv_min_threshold
    }


# ══════════════════════════════════════════════════════════════════════════════
# Формирование итоговой таблицы ядра
# ══════════════════════════════════════════════════════════════════════════════

def build_core_dataframe(
    final_core: set[str],
    subject_name: str,
    freq_df: pd.DataFrame,
    textbook_freqs: TextbookFrequencies,
    common_kv_map: dict[str, float],
    base_core: set[str],
    inherited_words: set[str],
    extra_words: set[str],
    common_words: set[str],
    config: Config,
) -> pd.DataFrame:
    """
    Создаёт DataFrame итогового лексического ядра.

    Args:
        final_core: итоговое множество слов ядра.
        subject_name: название предмета.
        freq_df: общий частотный список предмета.
        textbook_freqs: частоты по учебникам.
        common_kv_map: словарь {слово: КВ} общеупотребительной лексики.
        base_core: базовое ядро (для определения источника слова).
        inherited_words: унаследованные слова.
        extra_words: дополнительные слова (например, для Bi).
        common_words: слова из общеупотребительной лексики.
        config: объект конфигурации.

    Returns:
        DataFrame с колонками:
        - Слово
        - Предметная область
        - КВ
        - НЧ
        - Покрытие (%)
        - Коэффициент Жуайна
        - КВ (Внеклассная литература)
        - КВ_<учебник> (для каждого учебника)
        - НЧ_<учебник> (для каждого учебника)
    """
    # Справочники из общего списка предмета
    subj_kv = dict(zip(freq_df["Слово"], freq_df["Сумма слов"]))
    subj_nf = dict(zip(freq_df["Слово"], freq_df["Нормализованная частотность"]))

    total_textbooks = len(textbook_freqs.textbook_names)

    rows = []
    for word in final_core:
        if pd.isna(word) or word == "":
            continue

        # Определение предметной области
        subject_sources = base_core | inherited_words | extra_words
        if word in common_words and word not in subject_sources:
            domain = config.extracurricular_tag
        else:
            domain = subject_name

        # Основные метрики
        row = {
            "Слово": word,
            "Предметная область": domain,
            "КВ": float(subj_kv.get(word, 0.0)),
            "НЧ": float(subj_nf.get(word, 0.0)),
            "Покрытие (%)": coverage_percent(
                word, textbook_freqs.coverage, total_textbooks
            ),
            config.extracurricular_kv_col: float(common_kv_map.get(word, 0.0)),
        }

        # Частоты по учебникам
        nf_values = []
        for tb_name in textbook_freqs.textbook_names:
            kv_tb = textbook_freqs.get_kv(word, tb_name)
            nf_tb = textbook_freqs.get_nf(word, tb_name)

            row[f"КВ_{tb_name}"] = kv_tb
            row[f"НЧ_{tb_name}"] = round(nf_tb, 1)
            nf_values.append(nf_tb)

        # Коэффициент Жуайна
        row["Коэффициент Жуайна"] = calculate_juyna_coefficient(nf_values)

        # Опциональный столбец НЧ общеупотр.
        if config.add_common_nf_column:
            row["НЧ (общеупотр.)"] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Округление
    df["НЧ"] = df["НЧ"].astype(float).round(1)
    df["Покрытие (%)"] = df["Покрытие (%)"].astype(float).round(1)
    df["Коэффициент Жуайна"] = df["Коэффициент Жуайна"].astype(float).round(1)

    # Упорядочивание столбцов
    ordered_cols = [
        "Слово",
        "Предметная область",
        "КВ",
        "НЧ",
        "Покрытие (%)",
        "Коэффициент Жуайна",
        config.extracurricular_kv_col,
    ]
    ordered_cols += [f"КВ_{tb}" for tb in textbook_freqs.textbook_names]
    ordered_cols += [f"НЧ_{tb}" for tb in textbook_freqs.textbook_names]

    if config.add_common_nf_column:
        ordered_cols.append("НЧ (общеупотр.)")

    # Добавляем недостающие столбцы
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0.0 if ("КВ_" in col or "НЧ_" in col) else np.nan

    df = df[ordered_cols]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение ядра
# ══════════════════════════════════════════════════════════════════════════════

def save_lexical_core(
    df: pd.DataFrame,
    subject_folder: Path,
    grade: int,
    subject_name: str,
    total_unique_words: int,
    core_size: int,
) -> Optional[Path]:
    """
    Сохраняет лексическое ядро в Excel-файл.

    Удаляет старые файлы ядра перед сохранением.
    Имя файла: «Лексическое ядро <класс> <предмет> <всего_слов>_<размер_ядра>.xlsx»

    Args:
        df: таблица ядра.
        subject_folder: папка предмета.
        grade: номер класса.
        subject_name: название предмета.
        total_unique_words: общее количество уникальных слов в учебниках.
        core_size: размер итогового ядра.

    Returns:
        Путь к сохранённому файлу или None при ошибке.
    """
    output_folder = subject_folder / "Общий частотный список"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Удаление старых файлов ядра
    old_cores = list(output_folder.glob("Лексическое ядро*.xlsx"))
    for old_file in old_cores:
        try:
            old_file.unlink()
        except Exception as e:
            print(f"  [Предупреждение] Не удалось удалить старый файл {old_file}: {e}")

    # Формирование имени
    filename = (
        f"Лексическое ядро {grade} класс {subject_name} "
        f"{total_unique_words}_{core_size}.xlsx"
    )
    output_path = output_folder / filename

    try:
        df.to_excel(output_path, index=False)
        print(f"  ✓ Сохранено: {output_path}")
        return output_path
    except Exception as e:
        print(f"  ✗ Ошибка сохранения {output_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция построения ядра
# ══════════════════════════════════════════════════════════════════════════════

def create_lexical_core_for_subject(
    grade: int,
    subject_folder: Path,
    lower_cores_accumulator: dict[str, set[str]],
    config: Config,
    extra_known_words: Optional[set[str]] = None,
) -> set[str]:
    """
    Создаёт лексическое ядро для одного предмета в конкретном классе.

    Алгоритм:
    1. Загрузка общего частотного списка предмета
    2. Загрузка учебников и вычисление НЧ/покрытия
    3. Построение базового ядра по порогам
    4. Наследование ядер младших классов
    5. Добавление общеупотребительной лексики
    6. Добавление дополнительных слов (extra_known_words, например для Bi)
    7. Формирование итоговой таблицы с метриками
    8. Сохранение в Excel

    Args:
        grade: номер класса.
        subject_folder: путь к папке предмета.
        lower_cores_accumulator: словарь {предмет: множество слов из младших классов}.
        config: объект конфигурации.
        extra_known_words: дополнительные слова для включения (например, OkM/Ec для Bi).

    Returns:
        Множество слов итогового ядра (для наследования следующим классом).

    Examples:
        >>> accumulator = {}
        >>> core = create_lexical_core_for_subject(5, Path("5 класс/Bi"), accumulator, config)
        >>> len(core) > 0
        True
    """
    subject_name = subject_folder.name

    print(f"\n{'='*70}")
    print(f"Класс {grade} — {subject_name}")
    print(f"{'='*70}")

    # ── Шаг 1: Загрузка общего частотного списка предмета ────────────────────
    overall_freq_path = find_overall_subject_freq_file(subject_folder)
    if not overall_freq_path:
        print(f"  ✗ Не найден общий частотный список предмета → пропуск")
        return set()

    try:
        freq_df = load_overall_subject_freq_df(overall_freq_path, subject_name)
    except Exception as e:
        print(f"  ✗ Ошибка чтения общего списка: {e}")
        return set()

    print(f"  → Общий список: {overall_freq_path.name}")
    print(f"  → Всего слов в списке: {len(freq_df)}")

    # ── Шаг 2: Загрузка учебников ─────────────────────────────────────────────
    textbook_freqs, used_files, skipped_files = load_textbooks_for_subject(
        subject_folder, config
    )

    total_textbooks = len(textbook_freqs.textbook_names)
    print(f"  → Учебников: найдено {len(used_files) + len(skipped_files)}, "
          f"использовано {total_textbooks}, пропущено {len(skipped_files)}")

    if total_textbooks == 0:
        print(f"  ✗ Нет пригодных учебников → пропуск")
        return set()

    # ── Шаг 3: Базовое ядро ───────────────────────────────────────────────────
    base_core = build_base_core(freq_df, textbook_freqs, config)
    print(f"  → Базовое ядро (покрытие ≥{config.coverage_threshold}%, "
          f"НЧ ≥{config.norm_freq_threshold}): {len(base_core)} слов")

    # ── Шаг 4: Наследование ───────────────────────────────────────────────────
    inherited_words = lower_cores_accumulator.get(subject_name, set())
    print(f"  → Унаследовано из младших классов: {len(inherited_words)} слов")

    # ── Шаг 5: Общеупотребительная лексика ────────────────────────────────────
    common_words = filter_common_vocabulary(grade, config)
    common_kv_map = aggregated_common_kv_upto_grade(grade, config)
    print(f"  → Общеупотребительная лексика (КВ >{config.common_kv_min_threshold}): "
          f"{len(common_words)} слов")

    # ── Шаг 6: Дополнительные слова ───────────────────────────────────────────
    extra_words = set(extra_known_words) if extra_known_words else set()
    if extra_words:
        print(f"  → Дополнительные слова (OkM/Ec): {len(extra_words)} слов")

    # ── Шаг 7: Итоговое ядро ──────────────────────────────────────────────────
    final_core = base_core | inherited_words | common_words | extra_words
    print(f"  → ИТОГО в ядре: {len(final_core)} слов")

    # ── Шаг 8: Формирование таблицы ───────────────────────────────────────────
    core_df = build_core_dataframe(
        final_core=final_core,
        subject_name=subject_name,
        freq_df=freq_df,
        textbook_freqs=textbook_freqs,
        common_kv_map=common_kv_map,
        base_core=base_core,
        inherited_words=inherited_words,
        extra_words=extra_words,
        common_words=common_words,
        config=config,
    )

    if core_df.empty:
        print(f"  ✗ Итоговая таблица пуста → пропуск сохранения")
        return set()

    # ── Шаг 9: Сохранение ─────────────────────────────────────────────────────
    unique_words_in_textbooks = len(set(textbook_freqs.kv.keys()))
    save_lexical_core(
        df=core_df,
        subject_folder=subject_folder,
        grade=grade,
        subject_name=subject_name,
        total_unique_words=unique_words_in_textbooks,
        core_size=len(final_core),
    )

    # ── Диагностика ───────────────────────────────────────────────────────────
    high_nf = core_df[core_df["НЧ"] >= config.norm_freq_threshold]
    mid_nf = core_df[(core_df["НЧ"] >= 5) & (core_df["НЧ"] < 10)]
    print(f"  → В ядре: НЧ ≥{config.norm_freq_threshold} → {len(high_nf)}, "
          f"5 ≤ НЧ < 10 → {len(mid_nf)}")

    if len(high_nf) > 0:
        print("\n  Примеры слов с высокой частотностью:")
        sample = high_nf[["Слово", "Предметная область", "НЧ", "Покрытие (%)"]].head(5)
        for _, row in sample.iterrows():
            print(f"    • {row['Слово']:20s} | НЧ={row['НЧ']:6.1f} | "
                  f"Покрытие={row['Покрытие (%)']:5.1f}%")

    return final_core