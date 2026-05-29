# AI Sales Assistant — Project Specification

## Контекст
Этот проект — портфельная работа для позиции AI/Prompt-инженер в Точка Банк.
Цель: показать владение RAG, function calling, multi-turn диалогом, structured output и LLM-оценкой качества.

---

## Что строим
**AI-ассистент для продажи банковских продуктов МСБ** (расчётно-кассовое обслуживание, эквайринг, кредиты).

Ассистент помогает менеджеру по продажам или клиенту:
1. Узнать про продукты банка (RAG по базе знаний)
2. Подобрать подходящий тариф под параметры бизнеса
3. Рассчитать стоимость (function calling)
4. Оставить заявку (function calling → CRM)
5. Вести связный многоходовой диалог (memory)
6. Каждый ответ автоматически оценивается LLM-судьёй (structured output)

---

## Архитектура

```
User (Telegram)
      │
      ▼
FastAPI (main.py)
      │
      ▼
LangChain Agent (agent.py)
      ├── RAG Tool        → ChromaDB (knowledge base)
      ├── Tariff Tool     → tariffs.json (расчёт стоимости)
      ├── Lead Tool       → leads.db (SQLite CRM)
      └── Memory          → ConversationBufferMemory
      │
      ▼
LLM Judge (evaluator.py)  → evaluation_log.jsonl
```

---

## Стек
- **Python 3.11+**
- **FastAPI** — REST API + webhook для Telegram
- **python-telegram-bot** — Telegram интеграция
- **LangChain** — агент, memory, tools
- **OpenAI API** (`gpt-4o-mini`) — основная LLM (дешевле для разработки)
- **ChromaDB** — векторная БД для RAG
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **SQLite** — хранение лидов
- **Pydantic v2** — structured output / валидация
- **pytest** — тесты качества промптов
- **python-dotenv** — управление секретами

---

## Структура проекта

```
sales-assistant/
├── CLAUDE.md                   # этот файл
├── README.md                   # документация для GitHub
├── .env.example                # шаблон переменных окружения
├── .gitignore
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI приложение + Telegram webhook
│   ├── agent.py                # LangChain агент
│   ├── memory.py               # управление памятью диалога
│   ├── evaluator.py            # LLM-судья (structured output)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py         # поиск по базе знаний
│   │   ├── tariff_tool.py      # расчёт тарифов
│   │   └── lead_tool.py        # создание заявки в CRM
│   │
│   ├── prompts/
│   │   ├── system_prompt.md    # основной системный промпт агента
│   │   ├── rag_prompt.md       # промпт для RAG-синтеза ответа
│   │   └── judge_prompt.md     # промпт LLM-судьи
│   │
│   └── data/
│       ├── knowledge_base/     # .md файлы с описанием продуктов
│       │   ├── rko.md
│       │   ├── acquiring.md
│       │   └── credit.md
│       └── tariffs.json        # структурированные тарифы
│
├── scripts/
│   ├── ingest.py               # загрузка базы знаний в ChromaDB
│   └── run_evals.py            # запуск оценки качества промптов
│
├── tests/
│   ├── test_rag.py
│   ├── test_tools.py
│   └── test_prompts.py         # eval тесты с метриками
│
└── logs/
    └── evaluation_log.jsonl    # лог оценок LLM-судьи
```

---

## Детали реализации

### 1. RAG Tool
- База знаний: 3 markdown-файла (РКО, эквайринг, кредиты) — реалистичные описания продуктов условного банка
- Chunking: 500 токенов, overlap 50
- Retrieval: top-3 чанков по cosine similarity
- После retrieval — синтез ответа отдельным промптом (rag_prompt.md)

### 2. Tariff Tool (function calling)
```python
# Принимает: тип бизнеса, оборот в месяц, нужные услуги
# Возвращает: подходящий тариф + стоимость + что входит
calculate_tariff(business_type: str, monthly_turnover: int, services: list[str]) -> TariffResult
```

### 3. Lead Tool (function calling)
```python
# Создаёт запись в SQLite, возвращает номер заявки
create_lead(name: str, phone: str, business_type: str, product: str) -> LeadResult
```

### 4. Memory
- `ConversationBufferMemory` с window=10 последних сообщений
- Профиль клиента накапливается в session state: `{business_type, turnover, needs, name, phone}`

### 5. Structured Output + LLM Judge
Каждый ответ агента оценивается по 3 метрикам (0-10):
```python
class EvaluationResult(BaseModel):
    relevance: int        # насколько ответ релевантен вопросу
    groundedness: int     # подкреплён ли ответ данными из базы знаний
    sales_effectiveness: int  # продвигает ли диалог к целевому действию
    reasoning: str        # объяснение оценок
```
Результаты пишутся в `logs/evaluation_log.jsonl`.

---

## Промпты — ключевые принципы

### system_prompt.md
- Роль: опытный банковский менеджер, цель — понять потребность и предложить продукт
- Стратегия: сначала квалификация (3-4 вопроса), потом предложение
- Guardrails: не придумывать тарифы, использовать только данные из RAG
- Стиль: деловой, но дружелюбный, без банковского жаргона

### Версионирование промптов
Каждый промпт хранится в отдельном .md файле.
В коде промпты загружаются через `load_prompt(name)` — это позволяет менять промпты без изменения кода и отслеживать историю через git.

---

## Метрики качества (для README)
Запускать через `python scripts/run_evals.py`:
- Средний relevance score по тестовым диалогам
- Средний groundedness score
- Конверсия диалогов до создания лида (%)
- Latency p50/p95

---

## Переменные окружения (.env)
```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
CHROMA_PERSIST_DIR=./chroma_db
LEADS_DB_PATH=./leads.db
LOG_LEVEL=INFO
```

---

## Порядок реализации

1. Скелет FastAPI + структура папок + requirements.txt
2. База знаний (knowledge_base/*.md + tariffs.json)
3. RAG: ingest.py → ChromaDB, rag_tool.py
4. Tools: tariff_tool.py, lead_tool.py
5. Промпты: system_prompt.md, rag_prompt.md, judge_prompt.md
6. Агент: agent.py с LangChain + все tools + memory
7. Evaluator: evaluator.py + judge_prompt.md
8. Telegram: main.py с webhook
9. Тесты: tests/
10. Scripts: run_evals.py
11. README.md (финальная документация)

---

## Важно для качества кода
- Все промпты — в отдельных файлах, не в коде
- Типизация везде (Pydantic, type hints)
- Логирование каждого шага агента
- `.env.example` без реальных ключей
- `README.md` должен объяснять архитектуру, промпт-инжиниринг и метрики
