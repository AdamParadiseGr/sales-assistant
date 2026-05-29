from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent

from app.evaluator import Evaluator
from app.key_manager import AllKeysExhaustedError, GroqKeyManager, _is_rate_limit
from app.memory import MemoryManager
from app.tools.lead_tool import create_lead_tool
from app.tools.rag_tool import create_rag_tool
from app.tools.tariff_tool import create_tariff_tool

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


class _GroqNoParallel(ChatGroq):
    """ChatGroq that always disables parallel tool calls.

    Groq + LLaMA generates malformed tool-call JSON when parallel_tool_calls
    is enabled. create_react_agent calls bind_tools internally, so we inject
    the flag here to fix it at the source.
    """

    def bind_tools(self, tools, **kwargs):
        kwargs.setdefault("parallel_tool_calls", False)
        return super().bind_tools(tools, **kwargs)


class SalesAgent:
    """
    LangGraph ReAct agent with automatic Groq key rotation on rate-limit (429).

    Key rotation:
    - Uses GroqKeyManager to cycle through GROQ_API_KEY_1/2/3 on rate limits.
    - On each 429, rotates to the next key and rebuilds the LangGraph + tools.
    - Raises AllKeysExhaustedError (translated to a user-friendly message) when
      all keys are exhausted.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        leads_db_path: Optional[str] = None,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.3,
        key_manager: Optional[GroqKeyManager] = None,
    ) -> None:
        # Key manager: explicit > single key > env
        if key_manager is not None:
            self._key_manager = key_manager
        elif groq_api_key is not None:
            self._key_manager = GroqKeyManager(keys=[groq_api_key])
        else:
            self._key_manager = GroqKeyManager()

        self._model = model
        self._temperature = temperature
        self._db_path = leads_db_path or os.environ.get("LEADS_DB_PATH", "./leads.db")
        chroma_dir = chroma_persist_dir or os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

        # Embeddings + vectorstore — expensive, built only once
        self._vectorstore = self._build_vectorstore(chroma_dir)

        # LLMs + graph — cheap to rebuild, recreated on key rotation
        self._main_llm = self._make_main_llm()
        self._judge_llm = self._make_judge_llm()

        self._tools = self._build_tools()
        self._system_prompt = _load_prompt("system_prompt")
        self._memory_manager = MemoryManager()

        self.evaluator = Evaluator(
            llm=self._judge_llm,
            key_manager=self._key_manager,
            rebuild_llm=self._make_judge_llm,
        )

        # No checkpointer — history is passed explicitly per call
        self._graph = create_react_agent(
            model=self._main_llm,
            tools=self._tools,
        )

    # ------------------------------------------------------------------
    # LLM / graph construction helpers
    # ------------------------------------------------------------------

    def _make_main_llm(self) -> _GroqNoParallel:
        return _GroqNoParallel(
            model=self._model,
            temperature=self._temperature,
            groq_api_key=self._key_manager.get_current_key(),
        )

    def _make_judge_llm(self) -> _GroqNoParallel:
        return _GroqNoParallel(
            model=self._model,
            temperature=0,
            groq_api_key=self._key_manager.get_current_key(),
        )

    def _build_vectorstore(self, chroma_dir: str) -> Chroma:
        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )

    def _build_tools(self) -> list:
        return [
            create_rag_tool(vectorstore=self._vectorstore, llm=self._main_llm),
            create_tariff_tool(),
            create_lead_tool(db_path=self._db_path),
        ]

    def _rebuild_agent(self) -> None:
        """Recreate main LLM, tools (with new LLM), and graph after key rotation."""
        logger.info("Rebuilding agent with key index %d", self._key_manager._index)
        self._main_llm = self._make_main_llm()
        self._tools = self._build_tools()
        self._graph = create_react_agent(
            model=self._main_llm,
            tools=self._tools,
        )

    # ------------------------------------------------------------------
    # Rate-limit retry
    # ------------------------------------------------------------------

    async def _invoke_with_key_rotation(
        self,
        messages: list,
        config: dict,
        session_id: str,
    ) -> str:
        max_attempts = self._key_manager.key_count

        for attempt in range(max_attempts):
            try:
                result = await self._graph.ainvoke(
                    {"messages": messages}, config=config
                )
                return result["messages"][-1].content

            except Exception as exc:
                if _is_rate_limit(exc) and attempt < max_attempts - 1:
                    logger.warning(
                        "[%s] Groq rate limit on key %d/%d — rotating key. Error: %s",
                        session_id,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    self._key_manager.rotate()
                    self._rebuild_agent()
                else:
                    raise

        raise AllKeysExhaustedError(
            f"All {max_attempts} Groq API key(s) exhausted for session {session_id}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(self, session_id: str, user_message: str) -> str:
        session = self._memory_manager.get_or_create(session_id)
        logger.info("[%s] >>> %s", session_id, user_message[:120])

        messages = (
            [SystemMessage(content=self._system_prompt)]
            + session.get_messages()
            + [HumanMessage(content=user_message)]
        )
        config = {"configurable": {"thread_id": session_id}}

        try:
            response = await self._invoke_with_key_rotation(messages, config, session_id)
        except AllKeysExhaustedError as exc:
            logger.error("[%s] All Groq keys exhausted: %s", session_id, exc)
            response = "Извините, превышен лимит запросов. Попробуйте повторить через несколько минут."
        except Exception as exc:
            logger.exception("[%s] Agent error: %s", session_id, exc)
            response = "Извините, произошла техническая ошибка. Попробуйте повторить запрос."

        session.add_user_message(user_message)
        session.add_ai_message(response)

        logger.info("[%s] <<< %s", session_id, response[:120])

        history = session.get_history_text(last_n=6)
        asyncio.create_task(
            self.evaluator.evaluate(
                user_message=user_message,
                agent_response=response,
                conversation_history=history,
                session_id=session_id,
            )
        )

        return response

    def get_client_profile(self, session_id: str) -> dict:
        session = self._memory_manager.get_or_create(session_id)
        return session.client_profile.to_dict()

    def reset_session(self, session_id: str) -> None:
        self._memory_manager.delete(session_id)
        logger.info("Session reset: %s", session_id)

    @property
    def active_sessions(self) -> int:
        return self._memory_manager.session_count()
