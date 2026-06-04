"""
Тесты для src/cumulative.py

Покрываем:
- init_cumulative_all_state: инициализация
- update_cumulative_all_state: обновление из предметных ядер
- update_cumulative_all_with_extracurricular: добавление внеклассной
- save_cumulative_all_for_grade: сохранение файла
- вспомогательные функции
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.cumulative import (
    get_cumulative_stats,
    init_cumulative_all_state,
    print_cumulative_summary,
    save_cumulative_all_for_grade,
    update_cumulative_all_state,
    update_cumulative_all_with_extracurricular,
)
from src.loaders import clear_common_kv_cache


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config_fixture(tmp_path):
    """Базовая конфигурация."""
    return Config(
        base_dir=tmp_path / "base",
        common_fallback_dir=tmp_path / "fallback",
        common_books_dir=tmp_path / "books",
    )


@pytest.fixture
def sample_core_file(tmp_path):
    """Тестовый файл лексического ядра предмета."""
    subject_dir = tmp_path / "Bi" / "Общий частотный список"
    subject_dir.mkdir(parents=True)

    data = {
        "Слово": ["клетка", "днк", "белок"],
        "Предметная область": ["Bi", "Bi", "Bi"],
        "КВ": [150.0, 120.0, 80.0],
        "НЧ": [35.0, 28.0, 18.0],
        "Покрытие (%)": [100.0, 100.0, 75.0],
        "Коэффициент Жуайна": [95.0, 92.0, 88.0],
        "КВ (Внеклассная литература)": [0.0, 0.0, 0.0],
        "КВ_учебник1": [50.0, 40.0, 30.0],
        "КВ_учебник2": [60.0, 50.0, 25.0],
        "НЧ_учебник1": [5000.0, 4000.0, 3000.0],
        "НЧ_учебник2": [7500.0, 6250.0, 3125.0],
    }
    df = pd.DataFrame(data)
    path = subject_dir / "Лексическое ядро 5 класс Bi 100_3.xlsx"
    df.to_excel(path, index=False)

    return path


@pytest.fixture
def common_vocab_fixture(tmp_path, config_fixture):
    """Тестовая общеупотребительная лексика."""
    clear_common_kv_cache()

    fallback_dir = config_fixture.common_fallback_dir
    fallback_dir.mkdir(parents=True)

    data = {
        "Слово": ["общий", "текст", "слово"],
        "Сумма слов": [500.0, 300.0, 200.0],
    }
    pd.DataFrame(data).to_excel(
        fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# init_cumulative_all_state
# ══════════════════════════════════════════════════════════════════════════════

class TestInitCumulativeAllState:
    """Тесты инициализации кумулятивного состояния."""

    def test_creates_empty_state(self):
        """Создаёт пустое состояние с правильной структурой."""
        state = init_cumulative_all_state()

        assert "words" in state
        assert "subjects" in state
        assert isinstance(state["words"], dict)
        assert isinstance(state["subjects"], set)

    def test_initial_state_is_empty(self):
        """Начальное состояние пусто."""
        state = init_cumulative_all_state()

        assert len(state["words"]) == 0
        assert len(state["subjects"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# update_cumulative_all_state
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateCumulativeAllState:
    """Тесты обновления кумулятивного состояния из предметных ядер."""

    def test_updates_state_from_core_file(
        self, sample_core_file, config_fixture
    ):
        """Обновляет состояние данными из файла ядра."""
        state = init_cumulative_all_state()

        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        assert len(state["words"]) == 3
        assert "клетка" in state["words"]
        assert "днк" in state["words"]
        assert "белок" in state["words"]

    def test_adds_subject_to_subjects_set(
        self, sample_core_file, config_fixture
    ):
        """Добавляет предмет в множество subjects."""
        state = init_cumulative_all_state()

        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        assert "Bi" in state["subjects"]

    def test_records_first_grade(self, sample_core_file, config_fixture):
        """Записывает класс первого появления слова."""
        state = init_cumulative_all_state()

        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        assert state["words"]["клетка"]["first_grade"] == 5

    def test_updates_first_grade_to_minimum(
        self, sample_core_file, config_fixture
    ):
        """При повторном появлении сохраняет минимальный класс."""
        state = init_cumulative_all_state()

        # Слово появилось в 3 классе
        state["words"]["клетка"] = {
            "kv_total": 50.0,
            "per_subject": {"Bi": 50.0},
            "textbooks": set(),
            "first_grade": 3,
            "domains": {"Bi"},
        }

        # Обновляем из 5 класса
        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        # first_grade должен остаться 3
        assert state["words"]["клетка"]["first_grade"] == 3

    def test_accumulates_kv_across_grades(
        self, sample_core_file, config_fixture
    ):
        """КВ суммируется по классам."""
        state = init_cumulative_all_state()

        # КВ в 4 классе
        state["words"]["клетка"] = {
            "kv_total": 100.0,
            "per_subject": {"Bi": 100.0},
            "textbooks": set(),
            "first_grade": 4,
            "domains": {"Bi"},
        }

        # Обновляем из 5 класса (КВ=150)
        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        # Должно быть 100 + 150 = 250
        assert state["words"]["клетка"]["kv_total"] == 250.0
        assert state["words"]["клетка"]["per_subject"]["Bi"] == 250.0

    def test_counts_unique_textbooks(self, sample_core_file, config_fixture):
        """Подсчитывает уникальные учебники."""
        state = init_cumulative_all_state()

        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        # «клетка» есть в двух учебниках: учебник1 и учебник2
        textbooks = state["words"]["клетка"]["textbooks"]
        assert "Bi|учебник1" in textbooks
        assert "Bi|учебник2" in textbooks
        assert len(textbooks) == 2

    def test_records_domains(self, sample_core_file, config_fixture):
        """Записывает предметные области."""
        state = init_cumulative_all_state()

        update_cumulative_all_state(
            state, [sample_core_file], grade=5, config=config_fixture
        )

        assert "Bi" in state["words"]["клетка"]["domains"]

    def test_handles_missing_columns_gracefully(
        self, tmp_path, config_fixture
    ):
        """Пропускает файлы без обязательных столбцов."""
        subject_dir = tmp_path / "Mt" / "Общий частотный список"
        subject_dir.mkdir(parents=True)

        # Файл без столбца КВ
        data = {"Слово": ["слово"], "Частота": [100]}
        path = subject_dir / "Лексическое ядро 5 класс Mt.xlsx"
        pd.DataFrame(data).to_excel(path, index=False)

        state = init_cumulative_all_state()
        update_cumulative_all_state(state, [path], grade=5, config=config_fixture)

        # Состояние должно остаться пустым
        assert len(state["words"]) == 0

    def test_handles_corrupted_files_gracefully(
        self, tmp_path, config_fixture
    ):
        """Пропускает нечитаемые файлы."""
        bad_file = tmp_path / "bad.xlsx"
        bad_file.write_bytes(b"not an excel file")

        state = init_cumulative_all_state()
        update_cumulative_all_state(state, [bad_file], grade=5, config=config_fixture)

        assert len(state["words"]) == 0

    def test_processes_multiple_core_files(self, tmp_path, config_fixture):
        """Обрабатывает несколько файлов ядер."""
        # Ядро Bi
        bi_dir = tmp_path / "Bi" / "Общий частотный список"
        bi_dir.mkdir(parents=True)
        bi_data = {
            "Слово": ["клетка"],
            "Предметная область": ["Bi"],
            "КВ": [100.0],
            "КВ_учебник1": [100.0],
        }
        bi_path = bi_dir / "Лексическое ядро 5 класс Bi.xlsx"
        pd.DataFrame(bi_data).to_excel(bi_path, index=False)

        # Ядро Mt
        mt_dir = tmp_path / "Mt" / "Общий частотный список"
        mt_dir.mkdir(parents=True)
        mt_data = {
            "Слово": ["число"],
            "Предметная область": ["Mt"],
            "КВ": [200.0],
            "КВ_учебник1": [200.0],
        }
        mt_path = mt_dir / "Лексическое ядро 5 класс Mt.xlsx"
        pd.DataFrame(mt_data).to_excel(mt_path, index=False)

        state = init_cumulative_all_state()
        update_cumulative_all_state(
            state, [bi_path, mt_path], grade=5, config=config_fixture
        )

        assert len(state["words"]) == 2
        assert "Bi" in state["subjects"]
        assert "Mt" in state["subjects"]


# ══════════════════════════════════════════════════════════════════════════════
# update_cumulative_all_with_extracurricular
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateCumulativeAllWithExtracurricular:
    """Тесты добавления общеупотребительной лексики."""

    def test_adds_extracurricular_words(
        self, common_vocab_fixture, config_fixture
    ):
        """Добавляет слова из внеклассной литературы."""
        state = init_cumulative_all_state()

        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        assert "общий" in state["words"]
        assert "текст" in state["words"]
        assert "слово" in state["words"]

    def test_adds_extracurricular_tag_to_subjects(
        self, common_vocab_fixture, config_fixture
    ):
        """Добавляет тег внеклассной литературы в subjects."""
        state = init_cumulative_all_state()

        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        assert config_fixture.extracurricular_tag in state["subjects"]

    def test_does_not_add_to_kv_total(
        self, common_vocab_fixture, config_fixture
    ):
        """КВ внеклассной НЕ включается в kv_total."""
        state = init_cumulative_all_state()

        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        # kv_total должен остаться 0
        assert state["words"]["общий"]["kv_total"] == 0.0

    def test_records_kv_in_per_subject(
        self, common_vocab_fixture, config_fixture
    ):
        """КВ записывается в per_subject под тегом внеклассной."""
        state = init_cumulative_all_state()

        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        tag = config_fixture.extracurricular_tag
        assert state["words"]["общий"]["per_subject"][tag] == 500.0

    def test_overwrites_extracurricular_kv(
        self, common_vocab_fixture, config_fixture
    ):
        """
        При обновлении КВ внеклассной перезаписывается (не суммируется).
        Т.к. общие списки внеклассной уже кумулятивны.
        """
        state = init_cumulative_all_state()
        tag = config_fixture.extracurricular_tag

        # Слово уже было с КВ=100
        state["words"]["общий"] = {
            "kv_total": 0.0,
            "per_subject": {tag: 100.0},
            "textbooks": set(),
            "first_grade": 4,
            "domains": {tag},
        }

        # Обновляем из 5 класса (КВ=500)
        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        # Должно быть 500, а не 600
        assert state["words"]["общий"]["per_subject"][tag] == 500.0

    def test_updates_first_grade_to_minimum(
        self, common_vocab_fixture, config_fixture
    ):
        """Обновляет first_grade до минимального."""
        state = init_cumulative_all_state()

        # Слово уже было в 6 классе
        state["words"]["общий"] = {
            "kv_total": 0.0,
            "per_subject": {},
            "textbooks": set(),
            "first_grade": 6,
            "domains": set(),
        }

        # Обновляем из 5 класса
        update_cumulative_all_with_extracurricular(
            state, grade=5, config=config_fixture
        )

        assert state["words"]["общий"]["first_grade"] == 5

    def test_handles_empty_common_vocab(self, config_fixture):
        """Корректно работает при отсутствии общеупотр. лексики."""
        clear_common_kv_cache()
        state = init_cumulative_all_state()

        update_cumulative_all_with_extracurricular(
            state, grade=99, config=config_fixture
        )

        # Должно остаться пустым
        assert len(state["words"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# save_cumulative_all_for_grade
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveCumulativeAllForGrade:
    """Тесты сохранения кумулятивного ALL."""

    def test_saves_file_with_correct_name(self, config_fixture):
        """Сохраняет файл с правильным именем."""
        state = init_cumulative_all_state()
        state["words"]["слово"] = {
            "kv_total": 100.0,
            "per_subject": {"Bi": 100.0},
            "textbooks": {"Bi|учебник1"},
            "first_grade": 5,
            "domains": {"Bi"},
        }
        state["subjects"].add("Bi")

        path = save_cumulative_all_for_grade(5, state, config_fixture)

        assert path is not None
        assert path.exists()
        assert "Лексическое ядро 5 класс ALL.xlsx" in path.name

    def test_creates_output_directory(self, config_fixture):
        """Создаёт директорию для вывода если её нет."""
        state = init_cumulative_all_state()
        state["words"]["слово"] = {
            "kv_total": 100.0,
            "per_subject": {"Bi": 100.0},
            "textbooks": set(),
            "first_grade": 5,
            "domains": {"Bi"},
        }
        state["subjects"].add("Bi")

        path = save_cumulative_all_for_grade(5, state, config_fixture)

        assert path.parent.exists()
        assert "5 класс" in path.parent.name

    def test_dataframe_has_required_columns(self, config_fixture):
        """Таблица содержит обязательные столбцы."""
        state = init_cumulative_all_state()
        state["words"]["слово"] = {
            "kv_total": 100.0,
            "per_subject": {"Bi": 100.0},
            "textbooks": set(),
            "first_grade": 5,
            "domains": {"Bi"},
        }
        state["subjects"].add("Bi")

        path = save_cumulative_all_for_grade(5, state, config_fixture)
        df = pd.read_excel(path)

        required = [
            "Слово",
            "Предметные области",
            "КВ (сумма слов)",
            "Количество учебников",
            "Появился в классе",
            "КВ (Bi)",
        ]
        assert all(col in df.columns for col in required)

    def test_sorts_by_kv_descending(self, config_fixture):
        """Таблица отсортирована по убыванию КВ."""
        state = init_cumulative_all_state()
        state["words"]["слово1"] = {
            "kv_total": 50.0,
            "per_subject": {},
            "textbooks": set(),
            "first_grade": 5,
            "domains": set(),
        }
        state["words"]["слово2"] = {
            "kv_total": 200.0,
            "per_subject": {},
            "textbooks": set(),
            "first_grade": 5,
            "domains": set(),
        }
        state["words"]["слово3"] = {
            "kv_total": 100.0,
            "per_subject": {},
            "textbooks": set(),
            "first_grade": 5,
            "domains": set(),
        }

        path = save_cumulative_all_for_grade(5, state, config_fixture)
        df = pd.read_excel(path)

        words = df["Слово"].tolist()
        assert words == ["слово2", "слово3", "слово1"]

    def test_handles_empty_state(self, config_fixture):
        """Корректно сохраняет пустое состояние."""
        state = init_cumulative_all_state()

        path = save_cumulative_all_for_grade(5, state, config_fixture)

        assert path is not None
        df = pd.read_excel(path)
        assert len(df) == 0

    def test_includes_all_subjects_columns(self, config_fixture):
        """Включает столбцы для всех предметов."""
        state = init_cumulative_all_state()
        state["words"]["слово"] = {
            "kv_total": 100.0,
            "per_subject": {"Bi": 50.0, "Mt": 50.0},
            "textbooks": set(),
            "first_grade": 5,
            "domains": {"Bi", "Mt"},
        }
        state["subjects"] = {"Bi", "Mt", config_fixture.extracurricular_tag}

        path = save_cumulative_all_for_grade(5, state, config_fixture)
        df = pd.read_excel(path)

        assert "КВ (Bi)" in df.columns
        assert "КВ (Mt)" in df.columns
        assert f"КВ ({config_fixture.extracurricular_tag})" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCumulativeStats:
    """Тесты получения статистики."""

    def test_returns_correct_stats(self):
        """Возвращает корректную статистику."""
        state = init_cumulative_all_state()
        state["words"]["слово1"] = {
            "first_grade": 5,
            "kv_total": 100.0,
            "per_subject": {},
            "textbooks": set(),
            "domains": set(),
        }
        state["words"]["слово2"] = {
            "first_grade": 5,
            "kv_total": 50.0,
            "per_subject": {},
            "textbooks": set(),
            "domains": set(),
        }
        state["words"]["слово3"] = {
            "first_grade": 6,
            "kv_total": 30.0,
            "per_subject": {},
            "textbooks": set(),
            "domains": set(),
        }
        state["subjects"] = {"Bi", "Mt"}

        stats = get_cumulative_stats(state)

        assert stats["total_words"] == 3
        assert stats["total_subjects"] == 2
        assert stats["words_by_grade"][5] == 2
        assert stats["words_by_grade"][6] == 1


class TestPrintCumulativeSummary:
    """Тесты вывода сводки."""

    def test_prints_without_errors(self, capsys):
        """Выводит сводку без ошибок."""
        state = init_cumulative_all_state()
        state["words"]["слово"] = {
            "first_grade": 5,
            "kv_total": 100.0,
            "per_subject": {},
            "textbooks": set(),
            "domains": set(),
        }
        state["subjects"] = {"Bi"}

        print_cumulative_summary(state, 5)

        captured = capsys.readouterr()
        assert "Всего слов: 1" in captured.out
        assert "Предметов: 1" in captured.out