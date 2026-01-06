"""Tests for language mapping functionality."""

import pytest

from app.languages import map_whisper_to_omnilingual


class TestMapWhisperToOmnilingual:
    """Tests for the map_whisper_to_omnilingual function."""

    @pytest.mark.parametrize(
        "whisper_code,expected",
        [
            ("en", "eng_Latn"),
            ("es", "spa_Latn"),
            ("fr", "fra_Latn"),
            ("de", "deu_Latn"),
            ("zh", "cmn_Hans"),
            ("ja", "jpn_Jpan"),
            ("ko", "kor_Hang"),
            ("ar", "arb_Arab"),
            ("hi", "hin_Deva"),
            ("pt", "por_Latn"),
            ("ru", "rus_Cyrl"),
            ("it", "ita_Latn"),
        ],
    )
    def test_two_letter_codes(self, whisper_code: str, expected: str):
        """2-letter Whisper codes should map correctly."""
        assert map_whisper_to_omnilingual(whisper_code) == expected

    @pytest.mark.parametrize(
        "language_name,expected",
        [
            ("english", "eng_Latn"),
            ("spanish", "spa_Latn"),
            ("french", "fra_Latn"),
            ("german", "deu_Latn"),
            ("chinese", "cmn_Hans"),
            ("japanese", "jpn_Jpan"),
            ("korean", "kor_Hang"),
            ("arabic", "arb_Arab"),
            ("hindi", "hin_Deva"),
            ("portuguese", "por_Latn"),
        ],
    )
    def test_full_language_names(self, language_name: str, expected: str):
        """Full language names should map correctly."""
        assert map_whisper_to_omnilingual(language_name) == expected

    @pytest.mark.parametrize(
        "language_input,expected",
        [
            ("EN", "eng_Latn"),
            ("En", "eng_Latn"),
            ("ENGLISH", "eng_Latn"),
            ("English", "eng_Latn"),
            ("SPANISH", "spa_Latn"),
            ("Spanish", "spa_Latn"),
        ],
    )
    def test_case_insensitivity(self, language_input: str, expected: str):
        """Language mapping should be case-insensitive."""
        assert map_whisper_to_omnilingual(language_input) == expected

    @pytest.mark.parametrize(
        "omnilingual_code",
        [
            "eng_Latn",
            "spa_Latn",
            "fra_Latn",
            "deu_Latn",
            "cmn_Hans",
            "jpn_Jpan",
        ],
    )
    def test_passthrough_omnilingual_codes(self, omnilingual_code: str):
        """Already-valid Omnilingual-ASR codes should pass through unchanged."""
        assert map_whisper_to_omnilingual(omnilingual_code) == omnilingual_code

    def test_unknown_language_defaults_to_english(self, caplog):
        """Unknown languages should default to 'eng_Latn' and log a warning."""
        result = map_whisper_to_omnilingual("unknown_language")
        assert result == "eng_Latn"
        assert "Unknown language: unknown_language" in caplog.text

    def test_empty_string_defaults_to_english(self, caplog):
        """Empty string should default to 'eng_Latn' and log a warning."""
        result = map_whisper_to_omnilingual("")
        assert result == "eng_Latn"

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("castilian", "spa_Latn"),  # Spanish alias
            ("mandarin", "cmn_Hans"),  # Chinese alias
            ("flemish", "nld_Latn"),  # Dutch alias
            ("moldavian", "ron_Latn"),  # Romanian alias
            ("moldovan", "ron_Latn"),  # Romanian alias
            ("valencian", "cat_Latn"),  # Catalan alias
            ("panjabi", "pan_Guru"),  # Punjabi alias
            ("sinhalese", "sin_Sinh"),  # Sinhala alias
            ("pushto", "pbt_Arab"),  # Pashto alias
            ("burmese", "mya_Mymr"),  # Myanmar alias
        ],
    )
    def test_language_aliases(self, alias: str, expected: str):
        """Language aliases should map correctly."""
        assert map_whisper_to_omnilingual(alias) == expected

    @pytest.mark.parametrize(
        "whisper_code,expected",
        [
            ("yue", "yue_Hant"),  # Cantonese (3-letter)
            ("haw", "haw_Latn"),  # Hawaiian (3-letter)
        ],
    )
    def test_three_letter_codes(self, whisper_code: str, expected: str):
        """3-letter Whisper codes should map correctly."""
        assert map_whisper_to_omnilingual(whisper_code) == expected
