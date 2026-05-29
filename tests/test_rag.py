"""
Tests for the RAG tool.

Unit tests mock ChromaDB and the LLM — no API keys or ChromaDB required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_doc_factory():
    def _make(content: str, source: str = "rko"):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source}
        return doc
    return _make


@pytest.fixture
def mock_vectorstore(mock_doc_factory):
    store = MagicMock()
    store.similarity_search.return_value = [
        mock_doc_factory(
            "Тариф «Бизнес»: обслуживание 990 ₽/мес. До 50 платёжных поручений бесплатно.",
            source="rko",
        ),
        mock_doc_factory(
            "Тариф «Старт»: 0 ₽/мес при обороте от 10 000 ₽. Платежи в другие банки — 19 ₽.",
            source="rko",
        ),
        mock_doc_factory(
            "Открытие счёта бесплатно для всех тарифов. Онлайн за 5 минут.",
            source="rko",
        ),
    ]
    return store


@pytest.fixture
def mock_llm_sync():
    """LLM whose .invoke() returns a fixed content object."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="Тариф «Бизнес» стоит 990 ₽/мес. До 50 платежей в другие банки бесплатно."
    )
    return llm


@pytest.fixture
def mock_llm_async():
    """LLM whose .ainvoke() returns a fixed content object."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Тариф «Бизнес» стоит 990 ₽/мес.")
    )
    return llm


# ---------------------------------------------------------------------------
# Sync behaviour
# ---------------------------------------------------------------------------

def test_rag_tool_returns_non_empty_string(mock_vectorstore, mock_llm_sync):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    result = tool.invoke("Сколько стоит тариф Бизнес?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_rag_tool_calls_similarity_search(mock_vectorstore, mock_llm_sync):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    tool.invoke("Какие тарифы есть?")
    mock_vectorstore.similarity_search.assert_called_once()


def test_rag_tool_requests_top_3_chunks(mock_vectorstore, mock_llm_sync):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    tool.invoke("тестовый запрос")
    call_args = mock_vectorstore.similarity_search.call_args
    assert call_args[1].get("k", call_args[0][1] if len(call_args[0]) > 1 else None) == 3


def test_rag_tool_empty_results_returns_not_found_message():
    from app.tools.rag_tool import create_rag_tool
    empty_store = MagicMock()
    empty_store.similarity_search.return_value = []
    llm = MagicMock()

    tool = create_rag_tool(vectorstore=empty_store, llm=llm)
    result = tool.invoke("запрос без результата")

    assert "не найдена" in result.lower() or "не найден" in result.lower()
    llm.__or__.assert_not_called()  # LLM must NOT be called when no docs


def test_rag_tool_includes_source_metadata(mock_vectorstore, mock_llm_sync):
    """The formatted prompt fed to the LLM should mention the document source."""
    captured: list[str] = []

    def capture_invoke(formatted_input):
        captured.append(str(formatted_input))
        return MagicMock(content="ответ")

    mock_llm_sync.invoke.side_effect = capture_invoke

    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    tool.invoke("тарифы")

    assert captured, "llm.invoke was never called"
    assert "rko" in captured[0].lower()


# ---------------------------------------------------------------------------
# Async behaviour
# ---------------------------------------------------------------------------

async def test_rag_tool_async_returns_string(mock_vectorstore, mock_llm_async):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_async)
    result = await tool.arun("Сколько стоит тариф Бизнес?")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_rag_tool_async_no_results_message():
    from app.tools.rag_tool import create_rag_tool
    empty_store = MagicMock()
    empty_store.similarity_search.return_value = []
    llm = MagicMock()

    tool = create_rag_tool(vectorstore=empty_store, llm=llm)
    result = await tool.arun("несуществующий запрос")
    assert "не найдена" in result.lower() or "не найден" in result.lower()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

def test_rag_tool_has_correct_name(mock_vectorstore, mock_llm_sync):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    assert tool.name == "search_knowledge_base"


def test_rag_tool_description_is_informative(mock_vectorstore, mock_llm_sync):
    from app.tools.rag_tool import create_rag_tool
    tool = create_rag_tool(vectorstore=mock_vectorstore, llm=mock_llm_sync)
    desc = tool.description.lower()
    assert "знан" in desc or "база" in desc  # "базе знаний" somewhere
