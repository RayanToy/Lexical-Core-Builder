"""
Тесты для src/main.py

Покрываем основные сценарии:
- CLI парсер
- build_aux_known_for_bi
- валидация конфигурации
- обработку ошибок
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.main import build_aux_known_for_bi, create_parser, main


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config_with_aux_data(tmp_path):
    """Конфигурация с тестовыми данными для OkM/Ec."""
    base_dir = tmp_path / "base"

    # Создаём структуру для 2-4 классов с OkM и Ec
    for grade in [2, 3]:
        for subject in ["OkM", "Ec"]:
            class_dir = base_dir / f"{grade} класс списки"
            subject_dir = class_dir / subject
            freq_dir = subject_dir / "Общий частотный список"
            freq_dir.mkdir(parents=True)

            # Общий список предмета
            freq_data = {
                "Слово": [f"слово{grade}", f"понятие{grade}"],
                "Сумма слов": [150, 120],
                "Нормализованная частотность": [35.0, 28.0],
            }
            pd.DataFrame(freq_data).to_excel(
                freq_dir / f"Общий частотный список {subject}.xlsx", index=False
            )

            # Учебник
            tb_data = {
                "Слово": [f"слово{grade}", f"понятие{grade}"],
                "Сумма слов": [50, 40],
            }
            pd.DataFrame(tb_data).to_excel(
                subject_dir / f"учебник_{subject}_{grade}_10000.xlsx", index=False
            )

    return Config(
        base_dir=base_dir,
        common_fallback_dir=tmp_path / "fallback",
        common_books_dir=tmp_path / "books",
        coverage_threshold=50,  # снижаем для тестов
        norm_freq_threshold=10,  # снижаем для тестов
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI парсер
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateParser:
    """Тесты CLI парсера."""

    def test_parser_has_config_argument(self):
        """Парсер принимает аргумент --config."""
        parser = create_parser()
        args = parser.parse_args(["--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_parser_has_default_config(self):
        """По умолчанию используется config.yaml."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.config == "config.yaml"

    def test_parser_has_version(self):
        """Парсер поддерживает --version."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_accepts_short_config(self):
        """Парсер принимает короткий вариант -c."""
        parser = create_parser()
        args = parser.parse_args(["-c", "test.yaml"])
        assert args.config == "test.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# build_aux_known_for_bi
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildAuxKnownForBi:
    """Тесты построения дополнительной лексики для биологии."""

    def test_returns_set_of_words(self, config_with_aux_data):
        """Возвращает множество слов."""
        result = build_aux_known_for_bi(config_with_aux_data)
        assert isinstance(result, set)

    def test_includes_words_from_okm_and_ec(self, config_with_aux_data):
        """Включает слова из OkM и Ec."""
        result = build_aux_known_for_bi(config_with_aux_data)

        # Должны быть слова из 2 и 3 классов
        expected_words = {"слово2", "понятие2", "слово3", "понятие3"}
        assert expected_words.issubset(result)

    def test_returns_empty_for_missing_data(self, tmp_path):
        """Возвращает пустое множество при отсутствии данных."""
        config = Config(
            base_dir=tmp_path / "empty",
            common_fallback_dir=tmp_path / "fallback",
            common_books_dir=tmp_path / "books",
        )

        result = build_aux_known_for_bi(config)
        assert len(result) == 0

    def test_processes_grades_2_to_4_only(self, config_with_aux_data):
        """Обрабатывает только классы 2-4."""
        # Добавим данные для 5 класса
        base_dir = config_with_aux_data.base_dir
        class5_dir = base_dir / "5 класс списки" / "OkM"
        freq5_dir = class5_dir / "Общий частотный список"
        freq5_dir.mkdir(parents=True)

        freq_data = {
            "Слово": ["слово5"],
            "Сумма слов": [150],
            "Нормализованная частотность": [35.0],
        }
        pd.DataFrame(freq_data).to_excel(
            freq5_dir / "Общий частотный список OkM.xlsx", index=False
        )

        tb_data = {"Слово": ["слово5"], "Сумма слов": [50]}
        pd.DataFrame(tb_data).to_excel(
            class5_dir / "учебник_OkM_5_10000.xlsx", index=False
        )

        result = build_aux_known_for_bi(config_with_aux_data)

        # «слово5» не должно быть включено
        assert "слово5" not in result


# ══════════════════════════════════════════════════════════════════════════════
# Интеграционные тесты main()
# ══════════════════════════════════════════════════════════════════════════════

class TestMain:
    """Интеграционные тесты главной функции."""

    def test_main_exits_on_missing_config(self, monkeypatch, capsys):
        """main() завершается с ошибкой при отсутствии конфига."""
        # Имитируем аргументы
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", "missing.yaml"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "не найден" in captured.out.lower()

    def test_main_exits_on_missing_base_dir(self, tmp_path, monkeypatch, capsys):
        """main() завершается с ошибкой если base_dir не существует."""
        # Создаём конфиг с несуществующей базовой директорией
        config_file = tmp_path / "test_config.yaml"
        config_content = f"""
base_dir: "{tmp_path / 'nonexistent'}"
common_fallback_dir: "{tmp_path / 'fallback'}"
common_books_dir: "{tmp_path / 'books'}"
"""
        config_file.write_text(config_content)

        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "не найдена" in captured.out.lower()

    def test_main_handles_keyboard_interrupt(self, tmp_path, monkeypatch, capsys):
        """main() корректно обрабатывает Ctrl+C."""
        # Создаём валидный конфиг
        config_file = tmp_path / "test_config.yaml"
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        config_content = f"""
base_dir: "{base_dir}"
common_fallback_dir: "{tmp_path / 'fallback'}"
common_books_dir: "{tmp_path / 'books'}"
"""
        config_file.write_text(config_content)

        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

        # Эмулируем KeyboardInterrupt в run_lexical_core_pipeline
        def mock_pipeline(config):
            raise KeyboardInterrupt()

        monkeypatch.setattr("src.main.run_lexical_core_pipeline", mock_pipeline)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Остановлено пользователем" in captured.out

    def test_main_prints_version_info(self, tmp_path, monkeypatch, capsys):
        """main() выводит информацию о запуске."""
        config_file = tmp_path / "test_config.yaml"
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        config_content = f"""
base_dir: "{base_dir}"
common_fallback_dir: "{tmp_path / 'fallback'}"
common_books_dir: "{tmp_path / 'books'}"
"""
        config_file.write_text(config_content)

        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

        # Мокаем pipeline чтобы не выполнять реальную обработку
        monkeypatch.setattr("src.main.run_lexical_core_pipeline", lambda config: None)

        main()

        captured = capsys.readouterr()
        assert "Lexical Core Builder" in captured.out
        assert "Загрузка конфигурации" in captured.out