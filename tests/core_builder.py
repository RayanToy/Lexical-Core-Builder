"""
Тесты для src/core_builder.py

Покрываем:
- TextbookFrequencies: добавление учебников, вычисление метрик
- load_textbooks_for_subject: загрузка и агрегация
- build_base_core: отбор по порогам
- filter_common_vocabulary: фильтрация общеупотр. лексики
- build_core_dataframe: формирование итоговой таблицы
- save_lexical_core: сохранение файла
- create_lexical_core_for_subject: интеграция (end-to-end)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.core_builder import (
    TextbookFrequencies,
    build_base_core,
    build_core_dataframe,
    create_lexical_core_for_subject,
    filter_common_vocabulary,
    load_textbooks_for_subject,
    save_lexical_core,
)
from src.loaders import clear_common_kv_cache


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config_fixture(tmp_path):
    """Базовая конфигурация для тестов."""
    return Config(
        base_dir=tmp_path / "base",
        common_fallback_dir=tmp_path / "fallback",
        common_books_dir=tmp_path / "books",
        coverage_threshold=70,
        norm_freq_threshold=20,
        common_kv_min_threshold=7,
        norm_base=1_000_000,
    )


@pytest.fixture
def subject_freq_df():
    """Тестовый общий частотный список предмета."""
    data = {
        "Слово": ["клетка", "днк", "белок", "рибосома", "митохондрия"],
        "Сумма слов": [150, 120, 80, 50, 30],
        "Нормализованная частотность": [35.0, 28.0, 18.0, 12.0, 7.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def textbook_files_fixture(tmp_path):
    """Создаёт три тестовых учебника."""
    subject_dir = tmp_path / "Bi"
    subject_dir.mkdir()

    # Учебник 1: 10000 слов
    data1 = {
        "Слово": ["клетка", "днк", "белок"],
        "Сумма слов": [50, 40, 30],
    }
    pd.DataFrame(data1).to_excel(subject_dir / "учебник1_10000.xlsx", index=False)

    # Учебник 2: 8000 слов
    data2 = {
        "Слово": ["клетка", "днк", "рибосома"],
        "Сумма слов": [60, 50, 25],
    }
    pd.DataFrame(data2).to_excel(subject_dir / "учебник2_8000.xlsx", index=False)

    # Учебник 3: 12000 слов
    data3 = {
        "Слово": ["клетка", "белок", "митохондрия"],
        "Сумма слов": [40, 20, 15],
    }
    pd.DataFrame(data3).to_excel(subject_dir / "учебник3_12000.xlsx", index=False)

    return subject_dir


@pytest.fixture
def common_vocab_fixture(tmp_path, config_fixture):
    """Создаёт файл общеупотребительной лексики."""
    clear_common_kv_cache()
    
    fallback_dir = config_fixture.common_fallback_dir
    fallback_dir.mkdir(parents=True)

    data = {
        "Слово": ["общий", "текст", "слово", "редкое"],
        "Сумма слов": [100, 50, 20, 5],
    }
    pd.DataFrame(data).to_excel(
        fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# TextbookFrequencies
# ══════════════════════════════════════════════════════════════════════════════

class TestTextbookFrequencies:
    """Тесты класса TextbookFrequencies."""

    def test_initial_state_is_empty(self):
        """При создании все структуры пусты."""
        tf = TextbookFrequencies()
        assert len(tf.kv) == 0
        assert len(tf.nf) == 0
        assert len(tf.coverage) == 0
        assert len(tf.textbook_names) == 0

    def test_add_single_textbook(self):
        """Добавление одного учебника."""
        tf = TextbookFrequencies()
        words_freq = {"кот": 10.0, "пёс": 5.0}
        
        tf.add_textbook(
            textbook_name="учебник1",
            words_freq=words_freq,
            total_tokens=1000,
            norm_base=1_000_000,
        )

        assert "учебник1" in tf.textbook_names
        assert tf.get_kv("кот", "учебник1") == 10.0
        assert tf.get_coverage_count("кот") == 1

    def test_add_multiple_textbooks(self):
        """Добавление нескольких учебников."""
        tf = TextbookFrequencies()

        tf.add_textbook("учебник1", {"кот": 10.0}, 1000, 1_000_000)
        tf.add_textbook("учебник2", {"кот": 20.0}, 2000, 1_000_000)

        assert len(tf.textbook_names) == 2
        assert tf.get_coverage_count("кот") == 2

    def test_coverage_counts_unique_textbooks(self):
        """Покрытие считает уникальные учебники, где слово встречается."""
        tf = TextbookFrequencies()

        tf.add_textbook("tb1", {"слово": 10.0}, 1000, 1_000_000)
        tf.add_textbook("tb2", {"слово": 5.0}, 1000, 1_000_000)
        tf.add_textbook("tb3", {"другое": 3.0}, 1000, 1_000_000)

        assert tf.get_coverage_count("слово") == 2
        assert tf.get_coverage_count("другое") == 1

    def test_ignores_zero_frequency(self):
        """Слова с нулевой частотой не учитываются в покрытии."""
        tf = TextbookFrequencies()
        
        tf.add_textbook("tb1", {"слово": 0.0}, 1000, 1_000_000)

        assert tf.get_coverage_count("слово") == 0

    def test_ignores_negative_frequency(self):
        """Отрицательные частоты игнорируются."""
        tf = TextbookFrequencies()
        
        tf.add_textbook("tb1", {"слово": -5.0}, 1000, 1_000_000)

        assert tf.get_coverage_count("слово") == 0

    def test_calculates_normalized_frequency(self):
        """Нормализованная частотность вычисляется корректно."""
        tf = TextbookFrequencies()
        
        # 10 вхождений на 1000 слов = 10_000 на миллион
        tf.add_textbook("tb1", {"слово": 10.0}, 1000, 1_000_000)

        nf = tf.get_nf("слово", "tb1")
        assert nf == 10_000.0

    def test_get_nf_values_returns_list(self):
        """get_nf_values возвращает список НЧ по всем учебникам."""
        tf = TextbookFrequencies()
        
        tf.add_textbook("tb1", {"слово": 10.0}, 1000, 1_000_000)
        tf.add_textbook("tb2", {"слово": 5.0}, 1000, 1_000_000)
        tf.add_textbook("tb3", {}, 1000, 1_000_000)

        nf_values = tf.get_nf_values("слово")
        assert len(nf_values) == 3
        assert nf_values[0] == 10_000.0
        assert nf_values[1] == 5_000.0
        assert nf_values[2] == 0.0  # нет в tb3

    def test_get_kv_returns_zero_for_missing_word(self):
        """get_kv возвращает 0.0 для отсутствующего слова."""
        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"кот": 10.0}, 1000, 1_000_000)

        assert tf.get_kv("пёс", "tb1") == 0.0

    def test_get_nf_returns_zero_for_missing_textbook(self):
        """get_nf возвращает 0.0 для несуществующего учебника."""
        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"кот": 10.0}, 1000, 1_000_000)

        assert tf.get_nf("кот", "tb999") == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# load_textbooks_for_subject
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadTextbooksForSubject:
    """Тесты загрузки учебников предмета."""

    def test_loads_all_valid_textbooks(self, textbook_files_fixture, config_fixture):
        """Загружает все корректные учебники."""
        freqs, used, skipped = load_textbooks_for_subject(
            textbook_files_fixture, config_fixture
        )

        assert len(freqs.textbook_names) == 3
        assert len(used) == 3
        assert len(skipped) == 0

    def test_aggregates_word_frequencies(self, textbook_files_fixture, config_fixture):
        """Частоты слов агрегируются корректно."""
        freqs, _, _ = load_textbooks_for_subject(
            textbook_files_fixture, config_fixture
        )

        # «клетка» есть во всех трёх учебниках
        assert freqs.get_coverage_count("клетка") == 3

        # «рибосома» только в учебнике 2
        assert freqs.get_coverage_count("рибосома") == 1

    def test_skips_file_without_token_count(self, tmp_path, config_fixture):
        """Пропускает файлы без числа токенов в имени."""
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        data = {"Слово": ["слово"], "Сумма слов": [10]}
        pd.DataFrame(data).to_excel(subject_dir / "без_числа.xlsx", index=False)

        freqs, used, skipped = load_textbooks_for_subject(subject_dir, config_fixture)

        assert len(used) == 0
        assert len(skipped) == 1

    def test_skips_corrupted_file(self, tmp_path, config_fixture):
        """Пропускает нечитаемые файлы."""
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        # Создаём пустой файл
        (subject_dir / "поломанный_1000.xlsx").write_bytes(b"not an excel file")

        freqs, used, skipped = load_textbooks_for_subject(subject_dir, config_fixture)

        assert len(used) == 0
        assert len(skipped) == 1

    def test_returns_empty_for_empty_folder(self, tmp_path, config_fixture):
        """Пустая папка — пустой результат."""
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        freqs, used, skipped = load_textbooks_for_subject(subject_dir, config_fixture)

        assert len(freqs.textbook_names) == 0
        assert len(used) == 0
        assert len(skipped) == 0


# ══════════════════════════════════════════════════════════════════════════════
# build_base_core
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildBaseCore:
    """Тесты построения базового ядра."""

    def test_filters_by_coverage_and_nf(self, subject_freq_df, config_fixture):
        """Отбирает только слова, удовлетворяющие обоим порогам."""
        # Создаём учебники так, чтобы «клетка» и «днк» были в 100% учебников
        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"клетка": 10, "днк": 8}, 1000, 1_000_000)
        tf.add_textbook("tb2", {"клетка": 12, "днк": 9}, 1000, 1_000_000)

        # coverage_threshold = 70%, norm_freq_threshold = 20
        # «клетка»: НЧ=35 (✓), покрытие=100% (✓) → в ядре
        # «днк»:    НЧ=28 (✓), покрытие=100% (✓) → в ядре
        # «белок»:  НЧ=18 (✗), покрытие=0%   (✗) → НЕ в ядре

        base = build_base_core(subject_freq_df, tf, config_fixture)

        assert "клетка" in base
        assert "днк" in base
        assert "белок" not in base

    def test_excludes_low_coverage(self, subject_freq_df, config_fixture):
        """Исключает слова с низким покрытием."""
        tf = TextbookFrequencies()
        # «клетка» только в 1 из 3 учебников = 33% < 70%
        tf.add_textbook("tb1", {"клетка": 10}, 1000, 1_000_000)
        tf.add_textbook("tb2", {}, 1000, 1_000_000)
        tf.add_textbook("tb3", {}, 1000, 1_000_000)

        base = build_base_core(subject_freq_df, tf, config_fixture)

        assert "клетка" not in base

    def test_excludes_low_nf(self, subject_freq_df, config_fixture):
        """Исключает слова с низкой НЧ."""
        tf = TextbookFrequencies()
        # «рибосома»: НЧ=12 < 20 (порог)
        tf.add_textbook("tb1", {"рибосома": 10}, 1000, 1_000_000)

        base = build_base_core(subject_freq_df, tf, config_fixture)

        assert "рибосома" not in base

    def test_returns_empty_for_zero_textbooks(self, subject_freq_df, config_fixture):
        """Если учебников нет — пустое ядро."""
        tf = TextbookFrequencies()
        base = build_base_core(subject_freq_df, tf, config_fixture)

        assert len(base) == 0

    def test_ignores_empty_words(self, config_fixture):
        """Пропускает пустые и NaN слова."""
        import numpy as np
        
        freq_df = pd.DataFrame({
            "Слово": ["", np.nan, "слово"],
            "Сумма слов": [100, 100, 100],
            "Нормализованная частотность": [50, 50, 50],
        })

        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"слово": 10}, 1000, 1_000_000)

        base = build_base_core(freq_df, tf, config_fixture)

        assert "" not in base
        assert "слово" in base


# ══════════════════════════════════════════════════════════════════════════════
# filter_common_vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterCommonVocabulary:
    """Тесты фильтрации общеупотребительной лексики."""

    def test_filters_by_threshold(self, common_vocab_fixture, config_fixture):
        """Отбирает слова с КВ > порога."""
        # common_kv_min_threshold = 7
        # «общий»: 100 (✓), «текст»: 50 (✓), «слово»: 20 (✓), «редкое»: 5 (✗)
        
        result = filter_common_vocabulary(5, config_fixture)

        assert "общий" in result
        assert "текст" in result
        assert "слово" in result
        assert "редкое" not in result

    def test_returns_empty_when_no_data(self, config_fixture):
        """Если данных нет — пустое множество."""
        clear_common_kv_cache()
        result = filter_common_vocabulary(99, config_fixture)
        assert len(result) == 0

    def test_ignores_empty_words(self, tmp_path, config_fixture):
        """Пустые ключи не включаются."""
        clear_common_kv_cache()
        
        fallback_dir = config_fixture.common_fallback_dir
        fallback_dir.mkdir(parents=True)

        data = {"Слово": ["слово", ""], "Сумма слов": [100, 100]}
        pd.DataFrame(data).to_excel(
            fallback_dir / "Общий частотный список 5 класс.xlsx", index=False
        )

        result = filter_common_vocabulary(5, config_fixture)

        assert "слово" in result
        assert "" not in result


# ══════════════════════════════════════════════════════════════════════════════
# build_core_dataframe
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildCoreDataframe:
    """Тесты формирования итоговой таблицы ядра."""

    def test_creates_dataframe_with_correct_columns(
        self, subject_freq_df, config_fixture
    ):
        """Таблица содержит все обязательные столбцы."""
        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"клетка": 10}, 1000, 1_000_000)

        df = build_core_dataframe(
            final_core={"клетка"},
            subject_name="Bi",
            freq_df=subject_freq_df,
            textbook_freqs=tf,
            common_kv_map={},
            base_core={"клетка"},
            inherited_words=set(),
            extra_words=set(),
            common_words=set(),
            config=config_fixture,
        )

        required = [
            "Слово", "Предметная область", "КВ", "НЧ",
            "Покрытие (%)", "Коэффициент Жуайна"
        ]
        assert all(col in df.columns for col in required)

    def test_assigns_subject_domain_for_base_words(
        self, subject_freq_df, config_fixture
    ):
        """Слова из базового ядра получают предметную область."""
        tf = TextbookFrequencies()
        tf.add_textbook("tb1", {"клетка": 10}, 1000, 1_000_000)

        df = build_core_dataframe(
            final_core={"клетка"},
            subject_name="Bi",
            freq_df=subject_freq_df,
            textbook_freqs=tf,
            common_kv_map={},
            base_core={"клетка"},
            inherited_words=set(),
            extra_words=set(),
            common_words=set(),
            config=config_fixture,
        )

        assert df.iloc[0]["Предметная область"] == "Bi"

    def test_assigns_extracurricular_domain_for_common_only(
        self, subject_freq_df, config_fixture
    ):
        """Слова только из общеупотр. лексики → область «Внеклассная литература»."""
        tf = TextbookFrequencies()

        df = build_core_dataframe(
            final_core={"общее_слово"},
            subject_name="Bi",
            freq_df=subject_freq_df,
            textbook_freqs=tf,
            common_kv_map={"общее_слово": 100},
            base_core=set(),
            inherited_words=set(),
            extra_words=set(),
            common_words={"общее_слово"},
            config=config_fixture,
        )

        assert df.iloc[0]["Предметная область"] == config_fixture.extracurricular_tag

    def test_includes_textbook_columns(self, subject_freq_df, config_fixture):
        """Таблица содержит КВ_/НЧ_ для каждого учебника."""
        tf = TextbookFrequencies()
        tf.add_textbook("учебник1", {"клетка": 10}, 1000, 1_000_000)
        tf.add_textbook("учебник2", {"клетка": 5}, 1000, 1_000_000)

        df = build_core_dataframe(
            final_core={"клетка"},
            subject_name="Bi",
            freq_df=subject_freq_df,
            textbook_freqs=tf,
            common_kv_map={},
            base_core={"клетка"},
            inherited_words=set(),
            extra_words=set(),
            common_words=set(),
            config=config_fixture,
        )

        assert "КВ_учебник1" in df.columns
        assert "НЧ_учебник1" in df.columns
        assert "КВ_учебник2" in df.columns
        assert "НЧ_учебник2" in df.columns

    def test_returns_empty_dataframe_for_empty_core(
        self, subject_freq_df, config_fixture
    ):
        """Пустое ядро → пустая таблица."""
        tf = TextbookFrequencies()

        df = build_core_dataframe(
            final_core=set(),
            subject_name="Bi",
            freq_df=subject_freq_df,
            textbook_freqs=tf,
            common_kv_map={},
            base_core=set(),
            inherited_words=set(),
            extra_words=set(),
            common_words=set(),
            config=config_fixture,
        )

        assert df.empty


# ══════════════════════════════════════════════════════════════════════════════
# save_lexical_core
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveLexicalCore:
    """Тесты сохранения лексического ядра."""

    def test_saves_file_with_correct_name(self, tmp_path, config_fixture):
        """Файл сохраняется с корректным именем."""
        df = pd.DataFrame({"Слово": ["кот", "пёс"], "КВ": [10, 5]})
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        path = save_lexical_core(
            df=df,
            subject_folder=subject_dir,
            grade=5,
            subject_name="Bi",
            total_unique_words=100,
            core_size=2,
        )

        assert path is not None
        assert path.exists()
        assert "Лексическое ядро 5 класс Bi 100_2.xlsx" in path.name

    def test_creates_output_folder_if_missing(self, tmp_path):
        """Создаёт папку «Общий частотный список», если её нет."""
        df = pd.DataFrame({"Слово": ["кот"]})
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        save_lexical_core(
            df, subject_dir, 5, "Bi", 100, 1
        )

        assert (subject_dir / "Общий частотный список").exists()

    def test_deletes_old_core_files(self, tmp_path):
        """Удаляет старые файлы ядра перед сохранением."""
        subject_dir = tmp_path / "Bi"
        output_dir = subject_dir / "Общий частотный список"
        output_dir.mkdir(parents=True)

        # Старый файл
        old_file = output_dir / "Лексическое ядро 4 класс Bi 50_10.xlsx"
        pd.DataFrame({"Слово": ["старое"]}).to_excel(old_file, index=False)

        # Новый файл
        df = pd.DataFrame({"Слово": ["новое"]})
        save_lexical_core(df, subject_dir, 5, "Bi", 100, 1)

        # Старый файл должен быть удалён
        assert not old_file.exists()

    def test_returns_none_on_error(self, tmp_path, monkeypatch):
        """Возвращает None при ошибке сохранения."""
        df = pd.DataFrame({"Слово": ["кот"]})
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        # Монкей-патчим to_excel чтобы вызвать ошибку
        def mock_to_excel(*args, **kwargs):
            raise PermissionError("Denied")

        monkeypatch.setattr(pd.DataFrame, "to_excel", mock_to_excel)

        result = save_lexical_core(df, subject_dir, 5, "Bi", 100, 1)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# create_lexical_core_for_subject (интеграционный тест)
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateLexicalCoreForSubject:
    """Интеграционные тесты главной функции."""

    def test_creates_core_for_valid_subject(
        self, tmp_path, config_fixture, common_vocab_fixture
    ):
        """Создаёт ядро для корректного предмета."""
        # Структура данных
        base_dir = tmp_path / "base"
        class_dir = base_dir / "5 класс списки"
        subject_dir = class_dir / "Bi"
        freq_dir = subject_dir / "Общий частотный список"
        freq_dir.mkdir(parents=True)

        # Общий список предмета
        freq_data = {
            "Слово": ["клетка", "днк"],
            "Сумма слов": [150, 120],
            "Нормализованная частотность": [35.0, 28.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq_dir / "Общий частотный список Bi 5.xlsx", index=False
        )

        # Учебники
        tb_data = {"Слово": ["клетка", "днк"], "Сумма слов": [50, 40]}
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник1_10000.xlsx", index=False)
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник2_10000.xlsx", index=False)

        # Обновляем config
        config = Config(
            base_dir=base_dir,
            common_fallback_dir=config_fixture.common_fallback_dir,
            common_books_dir=config_fixture.common_books_dir,
        )

        accumulator = {}
        core = create_lexical_core_for_subject(
            grade=5,
            subject_folder=subject_dir,
            lower_cores_accumulator=accumulator,
            config=config,
        )

        assert len(core) > 0
        assert "клетка" in core
        assert "днк" in core

    def test_returns_empty_for_missing_freq_file(self, tmp_path, config_fixture):
        """Возвращает пустое множество, если нет общего списка."""
        subject_dir = tmp_path / "Bi"
        subject_dir.mkdir()

        accumulator = {}
        core = create_lexical_core_for_subject(
            5, subject_dir, accumulator, config_fixture
        )

        assert len(core) == 0

    def test_returns_empty_for_no_textbooks(self, tmp_path, config_fixture):
        """Возвращает пустое множество, если нет учебников."""
        subject_dir = tmp_path / "Bi"
        freq_dir = subject_dir / "Общий частотный список"
        freq_dir.mkdir(parents=True)

        freq_data = {
            "Слово": ["клетка"],
            "Сумма слов": [150],
            "Нормализованная частотность": [35.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq_dir / "Общий частотный список Bi.xlsx", index=False
        )

        accumulator = {}
        core = create_lexical_core_for_subject(
            5, subject_dir, accumulator, config_fixture
        )

        assert len(core) == 0

    def test_inherits_from_lower_grades(self, tmp_path, config_fixture):
        """Наследует слова из младших классов."""
        subject_dir = tmp_path / "Bi"
        freq_dir = subject_dir / "Общий частотный список"
        freq_dir.mkdir(parents=True)

        freq_data = {
            "Слово": ["новое"],
            "Сумма слов": [10],
            "Нормализованная частотность": [5.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq_dir / "Общий частотный список Bi.xlsx", index=False
        )

        tb_data = {"Слово": ["новое"], "Сумма слов": [5]}
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник_10000.xlsx", index=False)

        # В аккумуляторе уже есть «старое» из 4 класса
        accumulator = {"Bi": {"старое"}}

        core = create_lexical_core_for_subject(
            5, subject_dir, accumulator, config_fixture
        )

        assert "старое" in core  # унаследовано

    def test_includes_extra_known_words(self, tmp_path, config_fixture):
        """Включает дополнительные слова (extra_known_words)."""
        subject_dir = tmp_path / "Bi"
        freq_dir = subject_dir / "Общий частотный список"
        freq_dir.mkdir(parents=True)

        freq_data = {
            "Слово": ["слово"],
            "Сумма слов": [10],
            "Нормализованная частотность": [5.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq_dir / "Общий частотный список Bi.xlsx", index=False
        )

        tb_data = {"Слово": ["слово"], "Сумма слов": [5]}
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник_10000.xlsx", index=False)

        accumulator = {}
        extra = {"дополнительное"}

        core = create_lexical_core_for_subject(
            5, subject_dir, accumulator, config_fixture, extra_known_words=extra
        )

        assert "дополнительное" in core

    def test_saves_file_on_success(self, tmp_path, config_fixture):
        """Сохраняет файл ядра при успешной обработке."""
        base_dir = tmp_path / "base"
        class_dir = base_dir / "5 класс списки"
        subject_dir = class_dir / "Bi"
        freq_dir = subject_dir / "Общий частотный список"
        freq_dir.mkdir(parents=True)

        freq_data = {
            "Слово": ["клетка"],
            "Сумма слов": [150],
            "Нормализованная частотность": [35.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq_dir / "Общий частотный список Bi.xlsx", index=False
        )

        tb_data = {"Слово": ["клетка"], "Сумма слов": [50]}
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник1_10000.xlsx", index=False)
        pd.DataFrame(tb_data).to_excel(subject_dir / "учебник2_10000.xlsx", index=False)

        config = Config(
            base_dir=base_dir,
            common_fallback_dir=config_fixture.common_fallback_dir,
            common_books_dir=config_fixture.common_books_dir,
        )

        accumulator = {}
        create_lexical_core_for_subject(5, subject_dir, accumulator, config)

        # Проверяем что файл создан
        saved_files = list(freq_dir.glob("Лексическое ядро*.xlsx"))
        assert len(saved_files) == 1
        assert "Лексическое ядро 5 класс Bi" in saved_files[0].name