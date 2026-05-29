"""
Tests for GroqKeyManager and rate-limit retry plumbing.

All tests are unit-level — no real API calls, no env vars required
(keys are passed directly via the `keys` constructor parameter).
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ===========================================================================
# GroqKeyManager — core behaviour
# ===========================================================================

class TestGroqKeyManager:
    def _km(self, *keys: str):
        from app.key_manager import GroqKeyManager
        return GroqKeyManager(keys=list(keys))

    # --- get_current_key ---

    def test_get_current_key_returns_first(self):
        km = self._km("key1", "key2", "key3")
        assert km.get_current_key() == "key1"

    def test_get_current_key_single_key(self):
        km = self._km("only_key")
        assert km.get_current_key() == "only_key"

    # --- key_count ---

    def test_key_count_three(self):
        km = self._km("a", "b", "c")
        assert km.key_count == 3

    def test_key_count_one(self):
        km = self._km("sole")
        assert km.key_count == 1

    # --- rotate ---

    def test_rotate_returns_second_key(self):
        km = self._km("key1", "key2", "key3")
        result = km.rotate()
        assert result == "key2"

    def test_rotate_advances_current(self):
        km = self._km("key1", "key2", "key3")
        km.rotate()
        assert km.get_current_key() == "key2"

    def test_rotate_twice_reaches_third(self):
        km = self._km("key1", "key2", "key3")
        km.rotate()
        km.rotate()
        assert km.get_current_key() == "key3"

    def test_rotate_past_last_raises(self):
        from app.key_manager import AllKeysExhaustedError
        km = self._km("key1", "key2")
        km.rotate()  # key2
        with pytest.raises(AllKeysExhaustedError):
            km.rotate()  # no more keys

    def test_rotate_single_key_raises_immediately(self):
        from app.key_manager import AllKeysExhaustedError
        km = self._km("only")
        with pytest.raises(AllKeysExhaustedError):
            km.rotate()

    def test_rotate_exhausted_error_message_mentions_count(self):
        from app.key_manager import AllKeysExhaustedError
        km = self._km("k1", "k2")
        km.rotate()
        with pytest.raises(AllKeysExhaustedError, match="2"):
            km.rotate()

    def test_get_current_key_stable_after_exhaustion(self):
        """Index must not go out of bounds after AllKeysExhaustedError."""
        from app.key_manager import AllKeysExhaustedError
        km = self._km("k1", "k2")
        km.rotate()
        try:
            km.rotate()
        except AllKeysExhaustedError:
            pass
        # Should not raise IndexError
        assert km.get_current_key() == "k2"

    # --- reset ---

    def test_reset_returns_to_first_key(self):
        km = self._km("key1", "key2", "key3")
        km.rotate()
        km.rotate()
        km.reset()
        assert km.get_current_key() == "key1"

    def test_reset_allows_rotating_again(self):
        from app.key_manager import AllKeysExhaustedError
        km = self._km("k1", "k2")
        km.rotate()
        try:
            km.rotate()
        except AllKeysExhaustedError:
            pass
        km.reset()
        # Should not raise now
        result = km.rotate()
        assert result == "k2"


# ===========================================================================
# _load_keys — environment variable loading
# ===========================================================================

class TestGroqKeyManagerEnvLoading:
    def test_loads_numbered_keys(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY_1", "env_key1")
        monkeypatch.setenv("GROQ_API_KEY_2", "env_key2")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY_3", raising=False)

        # Import fresh after monkeypatching
        from app.key_manager import GroqKeyManager
        km = GroqKeyManager()
        assert km.key_count == 2
        assert km.get_current_key() == "env_key1"

    def test_falls_back_to_groq_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "legacy_key")
        monkeypatch.delenv("GROQ_API_KEY_1", raising=False)

        from app.key_manager import GroqKeyManager
        km = GroqKeyManager()
        assert km.get_current_key() == "legacy_key"

    def test_numbered_keys_take_priority_over_legacy(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "legacy")
        monkeypatch.setenv("GROQ_API_KEY_1", "numbered1")
        monkeypatch.delenv("GROQ_API_KEY_2", raising=False)

        from app.key_manager import GroqKeyManager
        km = GroqKeyManager()
        assert km.get_current_key() == "numbered1"
        assert km.key_count == 1

    def test_raises_value_error_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        for i in range(1, 5):
            monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)

        from app.key_manager import GroqKeyManager
        with pytest.raises(ValueError, match="No Groq API keys"):
            GroqKeyManager()


# ===========================================================================
# _is_rate_limit helper
# ===========================================================================

class TestIsRateLimit:
    def _check(self, exc: BaseException) -> bool:
        from app.key_manager import _is_rate_limit
        return _is_rate_limit(exc)

    def test_exception_named_ratelimit(self):
        class RateLimitError(Exception): pass
        assert self._check(RateLimitError("oops"))

    def test_exception_named_rate_limit_underscored(self):
        class Rate_LimitError(Exception): pass
        assert self._check(Rate_LimitError("oops"))

    def test_message_contains_429(self):
        assert self._check(Exception("HTTP 429 Too Many Requests"))

    def test_regular_exception_is_not_rate_limit(self):
        assert not self._check(ValueError("bad value"))

    def test_cause_chain_detected(self):
        class RateLimitError(Exception): pass
        outer = RuntimeError("wrapper")
        outer.__cause__ = RateLimitError("inner 429")
        assert self._check(outer)

    def test_non_rate_limit_cause(self):
        outer = RuntimeError("wrapper")
        outer.__cause__ = ValueError("unrelated")
        assert not self._check(outer)


# ===========================================================================
# Evaluator retry — mocked LLM, simulated rate limit
# ===========================================================================

class TestEvaluatorRateLimit:
    def _make_structured_llm(self, side_effects: list):
        """
        Return a mock structured LLM whose ainvoke() raises/returns
        the items in *side_effects* in order.
        """
        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=side_effects)
        return mock

    def _make_base_llm(self, structured_mock):
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured_mock)
        return llm

    async def test_evaluator_retries_on_rate_limit_and_succeeds(self):
        from app.key_manager import GroqKeyManager
        from app.evaluator import Evaluator, EvaluationResult

        good_result = EvaluationResult(
            relevance=8, groundedness=7, sales_effectiveness=6, reasoning="ok"
        )

        class FakeRateLimitError(Exception): pass
        # Patch _is_rate_limit so our fake exception is treated as a rate limit
        with patch("app.evaluator._is_rate_limit", side_effect=lambda e: isinstance(e, FakeRateLimitError)):
            structured_mock = self._make_structured_llm(
                [FakeRateLimitError("429"), good_result]
            )
            base_llm = self._make_base_llm(structured_mock)

            km = GroqKeyManager(keys=["key1", "key2"])
            new_llm = MagicMock()
            new_llm.with_structured_output = MagicMock(return_value=structured_mock)

            evaluator = Evaluator(
                llm=base_llm,
                key_manager=km,
                rebuild_llm=lambda: new_llm,
            )
            result = await evaluator.evaluate("q", "a", session_id="s1")

        assert result.relevance == 8
        assert km.get_current_key() == "key2"

    async def test_evaluator_returns_zero_when_all_keys_exhausted(self):
        from app.key_manager import GroqKeyManager
        from app.evaluator import Evaluator

        class FakeRateLimitError(Exception): pass

        with patch("app.evaluator._is_rate_limit", side_effect=lambda e: isinstance(e, FakeRateLimitError)):
            structured_mock = self._make_structured_llm(
                [FakeRateLimitError("429"), FakeRateLimitError("429")]
            )
            base_llm = self._make_base_llm(structured_mock)

            km = GroqKeyManager(keys=["key1", "key2"])
            new_llm = MagicMock()
            new_llm.with_structured_output = MagicMock(return_value=structured_mock)

            evaluator = Evaluator(
                llm=base_llm,
                key_manager=km,
                rebuild_llm=lambda: new_llm,
            )
            result = await evaluator.evaluate("q", "a", session_id="s2")

        assert result.relevance == 0
        assert "исчерпан" in result.reasoning.lower() or "ошибка" in result.reasoning.lower()

    async def test_evaluator_without_key_manager_no_retry(self):
        """Without a key_manager, a rate-limit error returns zeros immediately."""
        from app.evaluator import Evaluator

        class FakeRateLimitError(Exception): pass

        with patch("app.evaluator._is_rate_limit", return_value=True):
            structured_mock = self._make_structured_llm([FakeRateLimitError("429")])
            base_llm = self._make_base_llm(structured_mock)

            evaluator = Evaluator(llm=base_llm)
            result = await evaluator.evaluate("q", "a", session_id="s3")

        assert result.relevance == 0
        assert structured_mock.ainvoke.call_count == 1
