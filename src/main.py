"""
Главный модуль проекта Lexical Core Builder.

Запускает полный pipeline построения лексических ядер:
1. Поиск директорий классов
2. Для каждого класса:
   - построение ядер по предметам
   - наследование от младших классов
   - интеграция общеупотребительной лексики
   - сохранение предметных ядер
3. Агрегация в кумулятивные ALL-словари
4. Специальная обработка для биологии (дополнение из OkM/Ec)

Использование:
    python -m src.main --config config.yaml
    python -m src.main  # использует config.yaml по умолчанию
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from .config import load_config
from .core_builder import create_lexical_core_for_subject
from .cumulative import (
    init_cumulative_all_state,
    print_cumulative_summary,
    save_cumulative_all_for_grade,
    update_cumulative_all_state,
    update_cumulative_all_with_extracurricular,
)
from .utils import find_class_dirs, find_subject_folders, latest_file_by_mtime


# ══════════════════════════════════════════════════════════════════════════════
# Построение дополнительной лексики для Bi
# ══════════════════════════════════════════════════════════════════════════════

def build_aux_known_for_bi(config) -> set[str]:
    """
    Собирает дополнительную лексику для биологии из OkM и Ec за 2-4 классы.

    Эта лексика добавляется в ядро биологии как базовая, поскольку
    предполагается, что учащиеся уже знакомы с соответствующими понятиями
    из окружающего мира и экологии.

    Args:
        config: объект конфигурации.

    Returns:
        Множество слов из ядер OkM и Ec за 2-4 классы.

    Examples:
        >>> aux = build_aux_known_for_bi(config)
        >>> len(aux) > 0
        True
    """
    aux_subjects = {"OkM", "Ec"}
    target_grades = [2, 3, 4]
    aux_known = set()

    print(f"\n{'='*70}")
    print("Подготовка дополнительной лексики для биологии")
    print(f"Предметы: {', '.join(aux_subjects)}, классы: {target_grades}")
    print(f"{'='*70}")

    class_dirs = dict(find_class_dirs(config.base_dir))
    local_accumulator = {}  # наследование только в рамках OkM/Ec 2-4

    for grade in target_grades:
        class_dir = class_dirs.get(grade)
        if not class_dir:
            print(f"[Bi-aux] Класс {grade}: директория не найдена")
            continue

        subject_folders = find_subject_folders(
            class_dir, config.allowed_subjects, config.excluded_block_names
        )

        for subject_folder in subject_folders:
            subject_name = subject_folder.name
            if subject_name not in aux_subjects:
                continue

            print(f"[Bi-aux] Обрабатываем {grade} класс, {subject_name}")

            # Создаём ядро через основную функцию
            subject_core = create_lexical_core_for_subject(
                grade=grade,
                subject_folder=subject_folder,
                lower_cores_accumulator=local_accumulator,
                config=config,
                extra_known_words=None,  # только Bi будет получать extra
            )

            if subject_core:
                aux_known |= subject_core
                # Обновляем локальный аккумулятор
                acc_set = local_accumulator.get(subject_name, set())
                local_accumulator[subject_name] = acc_set | subject_core

    print(f"[Bi-aux] Собрано слов: {len(aux_known)}")
    return aux_known


# ══════════════════════════════════════════════════════════════════════════════
# Главный pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_lexical_core_pipeline(config) -> None:
    """
    Запускает полный pipeline построения лексических ядер.

    Алгоритм:
    1. Поиск всех классов в base_dir
    2. Подготовка дополнительной лексики для биологии
    3. Для каждого класса по порядку:
       - поиск предметных папок
       - создание ядер по предметам
       - обновление кумулятивного ALL
    4. Вывод итоговой статистики

    Args:
        config: объект конфигурации.
    """
    start_time = time.time()

    # ── Поиск классов ─────────────────────────────────────────────────────────
    class_dirs = find_class_dirs(config.base_dir)
    if not class_dirs:
        print("❌ Не найдено ни одной директории классов в base_dir.")
        print(f"   Проверьте путь: {config.base_dir}")
        print("   Ожидаемые папки: '5 класс списки', '6 класс списки', и т.д.")
        return

    print(f"🔍 Найдено классов: {len(class_dirs)}")
    for grade, path in class_dirs:
        print(f"   {grade} класс: {path}")

    # ── Подготовка Bi-дополнения ──────────────────────────────────────────────
    aux_known_for_bi = build_aux_known_for_bi(config)

    # ── Инициализация состояния ───────────────────────────────────────────────
    lower_cores_accumulator = {}  # {предмет: множество слов из младших классов}
    cumulative_all_state = init_cumulative_all_state()

    # ── Обработка по классам ──────────────────────────────────────────────────
    total_subjects_processed = 0

    for grade, class_dir in class_dirs:
        print(f"\n{'█'*70}")
        print(f"КЛАСС {grade}")
        print(f"{'█'*70}")

        # Поиск предметных папок
        subject_folders = find_subject_folders(
            class_dir, config.allowed_subjects, config.excluded_block_names
        )

        if not subject_folders:
            print(f"⚠️  Предметные папки не найдены в {class_dir}")
            continue

        print(f"📚 Найдено предметов: {len(subject_folders)}")
        for folder in subject_folders:
            print(f"   • {folder.name}")

        # Список сохранённых ядер для обновления кумулятивного ALL
        saved_core_paths = []

        # Обработка каждого предмета
        for subject_folder in subject_folders:
            subject_name = subject_folder.name

            # Определение дополнительных слов для Bi
            extra_words = aux_known_for_bi if subject_name == "Bi" else None

            # Создание ядра предмета
            final_core = create_lexical_core_for_subject(
                grade=grade,
                subject_folder=subject_folder,
                lower_cores_accumulator=lower_cores_accumulator,
                config=config,
                extra_known_words=extra_words,
            )

            # Обновление аккумулятора наследования
            if final_core:
                acc_set = lower_cores_accumulator.get(subject_name, set())
                lower_cores_accumulator[subject_name] = acc_set | final_core
                total_subjects_processed += 1

            # Поиск сохранённого файла ядра
            output_dir = subject_folder / "Общий частотный список"
            if output_dir.exists():
                candidates = list(output_dir.glob("Лексическое ядро *.xlsx"))
                latest_core = latest_file_by_mtime(candidates)
                if latest_core:
                    saved_core_paths.append(latest_core)

        # ── Обновление кумулятивного ALL ──────────────────────────────────────
        if saved_core_paths:
            print(f"\n🔄 Обновление кумулятивного ALL для {grade} класса...")

            # Предметные ядра
            update_cumulative_all_state(
                cumulative_all_state, saved_core_paths, grade, config
            )

            # Общеупотребительная лексика
            update_cumulative_all_with_extracurricular(
                cumulative_all_state, grade, config
            )

            # Сохранение ALL
            save_cumulative_all_for_grade(grade, cumulative_all_state, config)

            # Статистика
            print_cumulative_summary(cumulative_all_state, grade)
        else:
            print(f"⚠️  Нет данных для ALL {grade} класса")

    # ── Итоговая статистика ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    print(f"\n{'🏁'*70}")
    print("ЗАВЕРШЕНО")
    print(f"{'🏁'*70}")
    print(f"📊 Обработано предметов: {total_subjects_processed}")
    print(f"⏱️  Время выполнения: {minutes}:{seconds:02d}")

    # Статистика кумулятивного состояния
    if cumulative_all_state["words"]:
        from .cumulative import get_cumulative_stats
        stats = get_cumulative_stats(cumulative_all_state)
        print(f"📖 Всего слов в ALL: {stats['total_words']}")
        print(f"📚 Всего предметов: {stats['total_subjects']}")

    print(f"💾 Результаты сохранены в: {config.combined_output_root}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI интерфейс
# ══════════════════════════════════════════════════════════════════════════════

def create_parser() -> argparse.ArgumentParser:
    """
    Создаёт парсер аргументов командной строки.

    Returns:
        Настроенный ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Lexical Core Builder — построение лексических ядер учебных текстов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m src.main                          # config.yaml
  python -m src.main --config config.local.yaml
  python -m src.main --config /path/to/config.yaml

Файл конфигурации должен содержать пути к данным:
  base_dir: "/data/lexical_lists"
  common_fallback_dir: "/data/fallback"
  common_books_dir: "/data/books"

Подробнее в README.md
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Путь к файлу конфигурации (по умолчанию: config.yaml)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Lexical Core Builder 1.0.0",
    )

    return parser


def main() -> None:
    """
    Точка входа приложения.

    Парсит аргументы командной строки, загружает конфигурацию
    и запускает основной pipeline.
    """
    parser = create_parser()
    args = parser.parse_args()

    print("🚀 Lexical Core Builder")
    print("=" * 50)

    try:
        # Загрузка конфигурации
        print(f"📄 Загрузка конфигурации: {args.config}")
        config = load_config(args.config)

        # Проверка доступности базовых путей
        if not config.base_dir.exists():
            print(f"❌ Базовая директория не найдена: {config.base_dir}")
            sys.exit(1)

        print(f"✅ Базовая директория: {config.base_dir}")
        print(f"✅ Резервная директория: {config.common_fallback_dir}")
        print(f"✅ Директория книг: {config.common_books_dir}")

        # Запуск pipeline
        run_lexical_core_pipeline(config)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Создайте файл конфигурации:")
        print("   cp config.yaml.example config.yaml")
        print("   # отредактируйте пути в config.yaml")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Остановлено пользователем")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        print("\n📍 Трассировка:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()