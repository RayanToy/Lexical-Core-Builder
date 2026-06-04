"""
Конфигурация проекта.
Загружается из YAML-файла, все пути — через pathlib.Path.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ─── Обязательные пути ────────────────────────────────────────────────────
    base_dir: Path
    common_fallback_dir: Path
    common_books_dir: Path

    # ─── Пороги базового ядра ─────────────────────────────────────────────────
    coverage_threshold: int = 70        # покрытие в %
    norm_freq_threshold: int = 20       # порог НЧ из общего списка предмета

    # ─── Настройки общеупотребительной лексики ────────────────────────────────
    common_kv_min_threshold: int = 7    # фильтр: "Сумма слов" > 7
    add_common_nf_column: bool = False  # доп. столбец "НЧ (общеупотр.)"

    # ─── Нормализация ─────────────────────────────────────────────────────────
    norm_base: int = 1_000_000

    # ─── Тег внеклассной литературы ───────────────────────────────────────────
    extracurricular_tag: str = "Внеклассная литература"

    # ─── Разрешённые предметы ─────────────────────────────────────────────────
    allowed_subjects: set[str] = field(default_factory=lambda: {
        "Al", "Bi", "Bt", "Ch", "Cy", "Ec", "Ge", "Geo", "Gm", "Hs",
        "Inf", "Iz", "Lt", "Mt", "Mu", "Ob", "OkM", "Ph", "R", "Sc",
        "Sm", "T", "inf",
    })

    # ─── Исключённые блоки ────────────────────────────────────────────────────
    excluded_block_names: set[str] = field(default_factory=lambda: {
        "Гуманитарный", "Гуманитарный блок",
        "Естественно научный", "Естественно-научный",
        "Естественно-научный блок", "Естественно научный блок",
        "Математический", "Математический блок",
        "Технологический", "Технология",
        "Филологический", "Филологический блок",
        "Эстетический",
    })

    # ─── Производные свойства ─────────────────────────────────────────────────
    @property
    def combined_output_root(self) -> Path:
        """Папка для кумулятивных ALL-файлов."""
        return self.base_dir / "CombinedLexicalCores"

    @property
    def extracurricular_kv_col(self) -> str:
        """Название столбца КВ внеклассной литературы."""
        return f"КВ ({self.extracurricular_tag})"


# ─── Загрузка из YAML ─────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> Config:
    """
    Читает YAML-файл и возвращает объект Config.

    Пример config.yaml:
        base_dir: "/data/lexical_lists"
        common_fallback_dir: "/data/fallback"
        common_books_dir: "/data/books"
        coverage_threshold: 70
        norm_freq_threshold: 20

    Raises:
        FileNotFoundError: если файл конфига не найден.
        KeyError: если отсутствует обязательное поле.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {config_path}\n"
            f"Скопируйте config.yaml.example -> config.yaml и заполните пути."
        )

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Обязательные поля
    required = ("base_dir", "common_fallback_dir", "common_books_dir")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"В конфиге отсутствуют обязательные поля: {missing}")

    # Конвертируем строки-пути в Path
    path_fields = ("base_dir", "common_fallback_dir", "common_books_dir")
    for key in path_fields:
        if key in raw:
            raw[key] = Path(raw[key])

    # set-поля из YAML (если пользователь переопределил)
    set_fields = ("allowed_subjects", "excluded_block_names")
    for key in set_fields:
        if key in raw and isinstance(raw[key], list):
            raw[key] = set(raw[key])

    return Config(**raw)