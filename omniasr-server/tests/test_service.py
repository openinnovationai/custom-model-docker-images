"""Tests for language mapping functionality."""

from unittest.mock import patch

import pytest

from app.service import OmnilingualASRService


class TestOmnilingualASRService:
    """Tests for the OmnilingualASRService class."""

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("omniASR_W2V_300M", False),
            ("omniASR_W2V_1B", False),
            ("omniASR_W2V_3B", False),
            ("omniASR_W2V_7B", False),
            ("omniASR_CTC_300M", False),
            ("omniASR_CTC_1B", False),
            ("omniASR_CTC_3B", False),
            ("omniASR_CTC_7B", False),
            ("omniASR_CTC_300M_v2", False),
            ("omniASR_CTC_1B_v2", False),
            ("omniASR_CTC_3B_v2", False),
            ("omniASR_CTC_7B_v2", False),
            ("omniASR_LLM_300M", True),
            ("omniASR_LLM_1B", True),
            ("omniASR_LLM_3B", True),
            ("omniASR_LLM_7B", True),
            ("omniASR_LLM_300M_v2", True),
            ("omniASR_LLM_1B_v2", True),
            ("omniASR_LLM_3B_v2", True),
            ("omniASR_LLM_7B_v2", True),
            ("omniASR_LLM_Unlimited_300M_v2", True),
            ("omniASR_LLM_Unlimited_1B_v2", True),
            ("omniASR_LLM_Unlimited_3B_v2", True),
            ("omniASR_LLM_Unlimited_7B_v2", True),
            ("omniASR_LLM_7B_ZS", True),
        ],
    )
    def test_is_llm_model(self, model_name: str, expected: bool):
        """Test that the is_llm_model property returns the correct value."""
        with patch("app.service.MODEL_NAME", model_name):
            service = OmnilingualASRService()
            assert service.is_llm_model == expected
