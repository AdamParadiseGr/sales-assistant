import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS leads (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT NOT NULL,
    business_type TEXT,
    product     TEXT,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new'
)
"""


def _init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def _insert_lead(
    db_path: str,
    lead_id: str,
    name: str,
    phone: str,
    business_type: str,
    product: str,
    created_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO leads (id, name, phone, business_type, product, created_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (lead_id, name, phone, business_type, product, created_at, "new"),
        )
        conn.commit()


class LeadInput(BaseModel):
    name: str = Field(description="Имя клиента")
    phone: str = Field(description="Номер телефона клиента")
    business_type: str = Field(description="Тип бизнеса: ИП, ООО и т.д.")
    product: str = Field(description="Интересующий продукт: РКО, эквайринг, кредит и т.д.")


def create_lead_tool(db_path: str) -> StructuredTool:
    _init_db(db_path)

    def create_lead(
        name: str,
        phone: str,
        business_type: str,
        product: str,
    ) -> str:
        lead_id = str(uuid.uuid4())[:8].upper()
        created_at = datetime.now().isoformat(timespec="seconds")

        try:
            _insert_lead(db_path, lead_id, name, phone, business_type, product, created_at)
        except Exception as exc:
            logger.error("Failed to create lead: %s", exc)
            return f"Ошибка при создании заявки. Попробуйте ещё раз или запишите данные вручную."

        logger.info("Lead created: id=%s name=%s product=%s", lead_id, name, product)

        return (
            f"Заявка успешно создана!\n"
            f"  Номер заявки: {lead_id}\n"
            f"  Клиент: {name}\n"
            f"  Телефон: {phone}\n"
            f"  Продукт: {product}\n\n"
            f"Менеджер свяжется с вами в течение 30 минут в рабочее время (9:00–21:00 МСК)."
        )

    return StructuredTool.from_function(
        func=create_lead,
        name="create_lead",
        description=(
            "Создаёт заявку клиента в CRM. "
            "Используй только когда клиент явно выразил готовность и назвал имя и номер телефона. "
            "Возвращает номер заявки и подтверждение."
        ),
        args_schema=LeadInput,
    )


def get_all_leads(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM leads ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]
