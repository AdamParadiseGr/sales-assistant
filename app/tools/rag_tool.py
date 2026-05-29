import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_rag_prompt() -> str:
    return (PROMPTS_DIR / "rag_prompt.md").read_text(encoding="utf-8")


def _build_context(docs: list) -> str:
    parts = [f"[Источник: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(parts)


def _search_and_synthesize(query: str, vectorstore: Any, llm: Any) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "Информация по данному запросу не найдена в базе знаний."

    context = _build_context(docs)
    logger.info("RAG retrieved %d chunks for: %s", len(docs), query[:60])

    prompt_template = _load_rag_prompt()
    prompt = PromptTemplate.from_template(prompt_template)
    formatted = prompt.format(context=context, question=query)
    response = llm.invoke(formatted)
    return response.content


async def _search_and_synthesize_async(query: str, vectorstore: Any, llm: Any) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "Информация по данному запросу не найдена в базе знаний."

    context = _build_context(docs)
    logger.info("RAG retrieved %d chunks for: %s", len(docs), query[:60])

    prompt_template = _load_rag_prompt()
    prompt = PromptTemplate.from_template(prompt_template)
    formatted = prompt.format(context=context, question=query)
    response = await llm.ainvoke(formatted)
    return response.content


def create_rag_tool(vectorstore: Any, llm: Any) -> StructuredTool:
    def search_knowledge_base(query: str) -> str:
        return _search_and_synthesize(query, vectorstore, llm)

    async def search_knowledge_base_async(query: str) -> str:
        return await _search_and_synthesize_async(query, vectorstore, llm)

    return StructuredTool.from_function(
        func=search_knowledge_base,
        coroutine=search_knowledge_base_async,
        name="search_knowledge_base",
        description=(
            "Поиск информации по продуктам банка: РКО, эквайринг, кредиты. "
            "Используй ВСЕГДА перед ответом на вопросы о тарифах, условиях, "
            "требованиях, документах и сроках. "
            "Вход: поисковый запрос на русском языке. "
            "Выход: синтезированный ответ из базы знаний."
        ),
    )
