"""
Prompt quality and evaluator tests.

Unit tests (no API key): check prompt files, evaluator plumbing, memory module.
Integration tests (require OPENAI_API_KEY): marked with @pytest.mark.integration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROMPTS_DIR = Path(__file__).parent.parent / "app" / "prompts"


# ===========================================================================
# Prompt file sanity checks
# ===========================================================================

class TestPromptFiles:
    def test_system_prompt_exists(self):
        assert (PROMPTS_DIR / "system_prompt.md").exists()

    def test_rag_prompt_exists(self):
        assert (PROMPTS_DIR / "rag_prompt.md").exists()

    def test_judge_prompt_exists(self):
        assert (PROMPTS_DIR / "judge_prompt.md").exists()

    def test_system_prompt_mentions_tools(self):
        text = (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")
        assert "search_knowledge_base" in text
        assert "calculate_tariff" in text
        assert "create_lead" in text

    def test_system_prompt_mentions_qualification(self):
        text = (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8").lower()
        assert "квалификац" in text

    def test_system_prompt_has_guardrail_against_hallucination(self):
        text = (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8").lower()
        assert "не придумыв" in text or "не выдумыв" in text or "не называй" in text

    def test_rag_prompt_has_context_placeholder(self):
        text = (PROMPTS_DIR / "rag_prompt.md").read_text(encoding="utf-8")
        assert "{context}" in text

    def test_rag_prompt_has_question_placeholder(self):
        text = (PROMPTS_DIR / "rag_prompt.md").read_text(encoding="utf-8")
        assert "{question}" in text

    def test_rag_prompt_instructs_no_hallucination(self):
        text = (PROMPTS_DIR / "rag_prompt.md").read_text(encoding="utf-8").lower()
        assert "только" in text or "строго" in text

    def test_judge_prompt_has_all_placeholders(self):
        text = (PROMPTS_DIR / "judge_prompt.md").read_text(encoding="utf-8")
        for placeholder in ["{conversation_history}", "{user_message}", "{agent_response}"]:
            assert placeholder in text, f"Missing placeholder: {placeholder}"

    def test_judge_prompt_defines_all_three_metrics(self):
        text = (PROMPTS_DIR / "judge_prompt.md").read_text(encoding="utf-8").lower()
        assert "relevance" in text
        assert "groundedness" in text
        assert "sales_effectiveness" in text

    def test_judge_prompt_defines_score_range(self):
        text = (PROMPTS_DIR / "judge_prompt.md").read_text(encoding="utf-8")
        assert "0" in text and "10" in text

    def test_judge_prompt_mentions_reasoning(self):
        text = (PROMPTS_DIR / "judge_prompt.md").read_text(encoding="utf-8").lower()
        assert "reasoning" in text or "объяснение" in text or "пояснение" in text


# ===========================================================================
# EvaluationResult model
# ===========================================================================

class TestEvaluationResult:
    def test_model_has_required_fields(self):
        from app.evaluator import EvaluationResult
        result = EvaluationResult(
            relevance=8,
            groundedness=7,
            sales_effectiveness=6,
            reasoning="Тест",
        )
        assert result.relevance == 8
        assert result.groundedness == 7
        assert result.sales_effectiveness == 6
        assert result.reasoning == "Тест"

    def test_average_property(self):
        from app.evaluator import EvaluationResult
        result = EvaluationResult(
            relevance=9, groundedness=6, sales_effectiveness=3, reasoning=""
        )
        assert abs(result.average - 6.0) < 0.01

    def test_scores_out_of_range_raise_validation_error(self):
        from pydantic import ValidationError
        from app.evaluator import EvaluationResult
        with pytest.raises(ValidationError):
            EvaluationResult(relevance=11, groundedness=5, sales_effectiveness=5, reasoning="x")

    def test_negative_scores_raise_validation_error(self):
        from pydantic import ValidationError
        from app.evaluator import EvaluationResult
        with pytest.raises(ValidationError):
            EvaluationResult(relevance=-1, groundedness=5, sales_effectiveness=5, reasoning="x")

    def test_model_dump_produces_dict(self):
        from app.evaluator import EvaluationResult
        result = EvaluationResult(relevance=8, groundedness=7, sales_effectiveness=6, reasoning="ok")
        d = result.model_dump()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"relevance", "groundedness", "sales_effectiveness", "reasoning"}


# ===========================================================================
# Evaluator — unit tests (mocked LLM)
# ===========================================================================

class TestEvaluator:
    def _make_evaluator(self, scores: dict | None = None):
        from app.evaluator import Evaluator, EvaluationResult
        scores = scores or {"relevance": 8, "groundedness": 7, "sales_effectiveness": 6}
        mock_result = EvaluationResult(**scores, reasoning="Тестовое объяснение.")
        # Evaluator calls self._structured_llm.ainvoke(...), so mock .ainvoke explicitly.
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
        return Evaluator(llm=mock_llm), mock_structured

    async def test_evaluate_returns_evaluation_result(self):
        from app.evaluator import EvaluationResult
        evaluator, _ = self._make_evaluator()
        result = await evaluator.evaluate(
            user_message="Сколько стоит тариф Бизнес?",
            agent_response="Тариф «Бизнес» стоит 990 ₽/мес.",
            session_id="test_session",
        )
        assert isinstance(result, EvaluationResult)

    async def test_evaluate_scores_within_range(self):
        evaluator, _ = self._make_evaluator()
        result = await evaluator.evaluate(
            user_message="Тест",
            agent_response="Ответ",
            session_id="s1",
        )
        for score in [result.relevance, result.groundedness, result.sales_effectiveness]:
            assert 0 <= score <= 10

    async def test_evaluate_calls_structured_llm(self):
        evaluator, mock_structured = self._make_evaluator()
        await evaluator.evaluate("q", "a", session_id="s2")
        mock_structured.ainvoke.assert_awaited_once()

    async def test_evaluate_logs_to_jsonl(self, tmp_path, monkeypatch):
        import app.evaluator as ev_mod
        log_file = tmp_path / "eval.jsonl"
        monkeypatch.setattr(ev_mod, "LOG_PATH", log_file)

        evaluator, _ = self._make_evaluator()
        await evaluator.evaluate(
            user_message="Вопрос о тарифах",
            agent_response="Тариф Бизнес — 990 ₽/мес",
            session_id="log_test",
        )

        assert log_file.exists()
        entries = [json.loads(l) for l in log_file.read_text().strip().splitlines() if l]
        assert len(entries) == 1
        assert entries[0]["session_id"] == "log_test"
        assert "scores" in entries[0]
        assert entries[0]["scores"]["relevance"] == 8

    async def test_evaluate_log_entry_has_timestamp(self, tmp_path, monkeypatch):
        import app.evaluator as ev_mod
        log_file = tmp_path / "eval.jsonl"
        monkeypatch.setattr(ev_mod, "LOG_PATH", log_file)

        evaluator, _ = self._make_evaluator()
        await evaluator.evaluate("q", "a", session_id="ts_test")

        entry = json.loads(log_file.read_text().strip())
        assert "timestamp" in entry

    async def test_evaluate_graceful_on_llm_failure(self):
        from app.evaluator import Evaluator, EvaluationResult
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=Exception("API down"))
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.evaluate("q", "a", session_id="err")

        assert isinstance(result, EvaluationResult)
        assert result.relevance == 0
        assert "ошибка" in result.reasoning.lower() or "не выполнена" in result.reasoning.lower()


# ===========================================================================
# Memory module
# ===========================================================================

class TestMemory:
    def test_session_memory_creates_fresh_profile(self):
        from app.memory import SessionMemory
        session = SessionMemory("s1")
        assert session.client_profile.business_type is None
        assert session.client_profile.turnover is None
        assert session.client_profile.needs == []

    def test_client_profile_update_business_type(self):
        from app.memory import SessionMemory
        session = SessionMemory("s2")
        session.update_profile(business_type="ИП")
        assert session.client_profile.business_type == "ИП"

    def test_client_profile_update_turnover(self):
        from app.memory import SessionMemory
        session = SessionMemory("s3")
        session.update_profile(turnover=1_500_000)
        assert session.client_profile.turnover == 1_500_000

    def test_client_profile_needs_appended(self):
        from app.memory import SessionMemory
        session = SessionMemory("s4")
        session.update_profile(needs="РКО")
        session.update_profile(needs="Эквайринг")
        assert "РКО" in session.client_profile.needs
        assert "Эквайринг" in session.client_profile.needs

    def test_client_profile_needs_no_duplicate(self):
        from app.memory import SessionMemory
        session = SessionMemory("s5")
        session.update_profile(needs="РКО")
        session.update_profile(needs="РКО")
        assert session.client_profile.needs.count("РКО") == 1

    def test_is_qualified_false_without_data(self):
        from app.memory import SessionMemory
        session = SessionMemory("s6")
        assert not session.client_profile.is_qualified()

    def test_is_qualified_true_with_type_and_turnover(self):
        from app.memory import SessionMemory
        session = SessionMemory("s7")
        session.update_profile(business_type="ООО", turnover=500_000)
        assert session.client_profile.is_qualified()

    def test_can_create_lead_false_without_name_phone(self):
        from app.memory import SessionMemory
        session = SessionMemory("s8")
        assert not session.client_profile.can_create_lead()

    def test_can_create_lead_true_with_name_and_phone(self):
        from app.memory import SessionMemory
        session = SessionMemory("s9")
        session.update_profile(name="Иван", phone="79161234567")
        assert session.client_profile.can_create_lead()

    def test_memory_manager_creates_new_session(self):
        from app.memory import MemoryManager
        mgr = MemoryManager()
        session = mgr.get_or_create("new_session")
        assert session.session_id == "new_session"

    def test_memory_manager_returns_same_session(self):
        from app.memory import MemoryManager
        mgr = MemoryManager()
        s1 = mgr.get_or_create("shared")
        s2 = mgr.get_or_create("shared")
        assert s1 is s2

    def test_memory_manager_delete_removes_session(self):
        from app.memory import MemoryManager
        mgr = MemoryManager()
        mgr.get_or_create("del_me")
        assert mgr.session_count() == 1
        mgr.delete("del_me")
        assert mgr.session_count() == 0

    def test_get_history_text_empty_on_new_session(self):
        from app.memory import SessionMemory
        session = SessionMemory("hist")
        text = session.get_history_text()
        assert isinstance(text, str)


# ===========================================================================
# Integration tests (require OPENAI_API_KEY)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)
class TestEvaluatorIntegration:
    async def test_evaluator_returns_valid_scores_for_good_answer(self):
        from langchain_groq import ChatGroq
        from app.evaluator import Evaluator
        from app.agent import DEFAULT_GROQ_MODEL

        llm = ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)
        evaluator = Evaluator(llm=llm)

        result = await evaluator.evaluate(
            user_message="Сколько стоит открыть счёт для ИП?",
            agent_response=(
                "Открытие расчётного счёта бесплатно для всех тарифов. "
                "Для ИП нужны паспорт и ОГРНИП. "
                "Счёт можно открыть онлайн — реквизиты выдаются за 5 минут."
            ),
            conversation_history="",
            session_id="integration_good",
        )
        assert result.relevance >= 7
        assert result.groundedness >= 6
        assert isinstance(result.reasoning, str) and len(result.reasoning) > 20

    async def test_evaluator_scores_low_for_irrelevant_answer(self):
        from langchain_groq import ChatGroq
        from app.evaluator import Evaluator
        from app.agent import DEFAULT_GROQ_MODEL

        llm = ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)
        evaluator = Evaluator(llm=llm)

        result = await evaluator.evaluate(
            user_message="Сколько стоит эквайринг?",
            agent_response="Сегодня прекрасная погода и солнечный день!",
            conversation_history="",
            session_id="integration_bad",
        )
        assert result.relevance <= 3
