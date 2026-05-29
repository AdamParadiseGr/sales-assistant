from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from app.key_manager import AllKeysExhaustedError, GroqKeyManager, _is_rate_limit

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
LOG_PATH = Path(__file__).parent.parent / "logs" / "evaluation_log.jsonl"


class EvaluationResult(BaseModel):
    relevance: int = Field(..., ge=0, le=10, description="Релевантность ответа вопросу (0–10)")
    groundedness: int = Field(..., ge=0, le=10, description="Обоснованность данными из базы (0–10)")
    sales_effectiveness: int = Field(..., ge=0, le=10, description="Эффективность продаж (0–10)")
    reasoning: str = Field(..., description="Краткое объяснение всех трёх оценок")

    @property
    def average(self) -> float:
        return (self.relevance + self.groundedness + self.sales_effectiveness) / 3


class Evaluator:
    """
    LLM-судья: оценивает каждый ответ агента по трём метрикам.

    Supports automatic key rotation on Groq rate-limit errors when
    *key_manager* and *rebuild_llm* are provided.
    """

    def __init__(
        self,
        llm: Any,
        key_manager: Optional[GroqKeyManager] = None,
        rebuild_llm: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._key_manager = key_manager
        self._rebuild_llm_fn = rebuild_llm
        self._structured_llm = llm.with_structured_output(EvaluationResult)
        self._judge_template = self._load_judge_prompt()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _load_judge_prompt(self) -> str:
        return (PROMPTS_DIR / "judge_prompt.md").read_text(encoding="utf-8")

    def _build_prompt(
        self,
        user_message: str,
        agent_response: str,
        conversation_history: str,
    ) -> str:
        return self._judge_template.format(
            conversation_history=conversation_history or "— (начало диалога)",
            user_message=user_message,
            agent_response=agent_response,
        )

    # ------------------------------------------------------------------
    # Rate-limit retry
    # ------------------------------------------------------------------

    async def _invoke_judge_with_retry(self, prompt: str) -> EvaluationResult:
        max_attempts = self._key_manager.key_count if self._key_manager else 1

        for attempt in range(max_attempts):
            try:
                return await self._structured_llm.ainvoke(prompt)

            except Exception as exc:
                can_rotate = (
                    _is_rate_limit(exc)
                    and self._key_manager is not None
                    and self._rebuild_llm_fn is not None
                    and attempt < max_attempts - 1
                )
                if can_rotate:
                    logger.warning(
                        "Evaluator rate limit on key %d/%d — rotating key. Error: %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    try:
                        self._key_manager.rotate()  # type: ignore[union-attr]
                        new_llm = self._rebuild_llm_fn()  # type: ignore[misc]
                        self._structured_llm = new_llm.with_structured_output(EvaluationResult)
                    except AllKeysExhaustedError as ke:
                        logger.error("Evaluator: all Groq keys exhausted: %s", ke)
                        return EvaluationResult(
                            relevance=0,
                            groundedness=0,
                            sales_effectiveness=0,
                            reasoning=f"Оценка не выполнена (все ключи исчерпаны): {ke}",
                        )
                else:
                    logger.warning("Evaluator LLM call failed: %s", exc)
                    return EvaluationResult(
                        relevance=0,
                        groundedness=0,
                        sales_effectiveness=0,
                        reasoning=f"Оценка не выполнена (ошибка): {exc}",
                    )

        # Should only be reached if max_attempts == 0 (guard)
        return EvaluationResult(
            relevance=0,
            groundedness=0,
            sales_effectiveness=0,
            reasoning="Оценка не выполнена (все попытки исчерпаны)",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        user_message: str,
        agent_response: str,
        conversation_history: str = "",
        session_id: str = "",
    ) -> EvaluationResult:
        prompt = self._build_prompt(user_message, agent_response, conversation_history)
        result = await self._invoke_judge_with_retry(prompt)

        self._log(user_message, agent_response, result, session_id)
        logger.info(
            "[%s] Eval scores — R:%d G:%d S:%d avg:%.1f",
            session_id,
            result.relevance,
            result.groundedness,
            result.sales_effectiveness,
            result.average,
        )
        return result

    def _log(
        self,
        user_message: str,
        agent_response: str,
        result: EvaluationResult,
        session_id: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_message": user_message,
            "agent_response": agent_response[:600],
            "scores": result.model_dump(),
        }
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not write evaluation log: %s", exc)
