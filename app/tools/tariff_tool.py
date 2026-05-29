import json
import logging
from pathlib import Path
from typing import List, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

TARIFFS_PATH = Path(__file__).parent.parent / "data" / "tariffs.json"

_ACQUIRING_SERVICE_MAP = {
    "acquiring_pos": "pos",
    "acquiring_internet": "internet",
    "acquiring_mpos": "mpos",
    "acquiring_sbp": "sbp",
    "эквайринг": "pos",
    "pos": "pos",
    "internet": "internet",
    "sbp": "sbp",
    "qr": "sbp",
}

_SERVICE_ALIASES = {
    "рко": "rko",
    "счёт": "rko",
    "счет": "rko",
    "расчётный счёт": "rko",
    "кредит": "credit",
    "овердрафт": "credit",
}


def _load_tariffs() -> dict:
    with open(TARIFFS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _find_rko_plan(tariffs: dict, monthly_turnover: int) -> dict | None:
    for plan in tariffs["rko"]["plans"]:
        lo = plan["min_turnover"]
        hi = plan["max_turnover"] if plan["max_turnover"] is not None else float("inf")
        if lo <= monthly_turnover <= hi:
            return plan
    return tariffs["rko"]["plans"][-1]


def _calc_acquiring_rate(rates: list, monthly_turnover: int) -> dict | None:
    for r in rates:
        hi = r["max_turnover"] if r["max_turnover"] is not None else float("inf")
        if r["min_turnover"] <= monthly_turnover <= hi:
            return r
    return rates[-1]


def _format_rub(amount: float) -> str:
    return f"{amount:,.0f} ₽".replace(",", " ")


def _calculate(
    business_type: str,
    monthly_turnover: int,
    services: List[str],
) -> str:
    tariffs = _load_tariffs()
    result_parts: list[str] = []

    # Normalize service names
    normalized: list[str] = []
    for s in services:
        s_lower = s.lower().strip()
        if s_lower in _SERVICE_ALIASES:
            normalized.append(_SERVICE_ALIASES[s_lower])
        else:
            normalized.append(s_lower)

    # Default to RKO if no services specified
    if not normalized:
        normalized = ["rko"]

    # --- RKO ---
    if "rko" in normalized:
        plan = _find_rko_plan(tariffs, monthly_turnover)
        fee = plan["monthly_fee"]
        fee_str = f"{fee} ₽/мес" if fee > 0 else "0 ₽/мес"
        cond = plan.get("monthly_fee_condition")
        free_pmts = plan["features"].get("free_online_payments_per_month")
        pmt_fee = plan["features"].get("online_payment_fee", 0)

        lines = [f"РКО — тариф «{plan['name']}»"]
        lines.append(f"  Обслуживание: {fee_str}" + (f" ({cond})" if cond else ""))
        if free_pmts:
            lines.append(f"  Платежи в другие банки: первые {free_pmts} бесплатно, далее {pmt_fee} ₽/шт")
        else:
            lines.append(f"  Платежи в другие банки: {pmt_fee} ₽/шт")
        withdraw_free = plan["features"].get("free_cash_withdrawal_per_month", 0)
        if withdraw_free:
            lines.append(f"  Снятие наличных: первые {_format_rub(withdraw_free)} бесплатно в месяц")
        result_parts.append("\n".join(lines))

    # --- Acquiring ---
    for svc in normalized:
        acq_key = _ACQUIRING_SERVICE_MAP.get(svc)
        if acq_key is None:
            continue
        acq = tariffs["acquiring"]["types"].get(acq_key)
        if acq is None:
            continue

        lines = [f"{acq['name']}"]
        if "rates" in acq:
            r = _calc_acquiring_rate(acq["rates"], monthly_turnover)
            rate_pct = r["rate"] * 100
            monthly_cost = monthly_turnover * r["rate"]
            lines.append(f"  Ставка: {rate_pct:.1f}% от оборота")
            lines.append(f"  При обороте {_format_rub(monthly_turnover)}/мес → ~{_format_rub(monthly_cost)}/мес")
            note = r.get("note")
            if note:
                lines.append(f"  ({note})")
        elif "rate" in acq:
            rate_pct = acq["rate"] * 100
            monthly_cost = monthly_turnover * acq["rate"]
            lines.append(f"  Ставка: {rate_pct:.1f}% от оборота")
            lines.append(f"  При обороте {_format_rub(monthly_turnover)}/мес → ~{_format_rub(monthly_cost)}/мес")

        setup = acq.get("setup_fee", 0)
        if setup == 0:
            lines.append("  Подключение: бесплатно")
        result_parts.append("\n".join(lines))

    # --- Credit hint ---
    if "credit" in normalized:
        result_parts.append(
            "Кредит\n"
            "  Оборотный кредит: от 300 000 до 10 000 000 ₽, ставка от 18% годовых\n"
            "  Кредит на развитие: от 1 000 000 до 50 000 000 ₽, ставка от 16% годовых\n"
            "  Для точного расчёта условий нужно уточнить параметры бизнеса у менеджера."
        )

    if not result_parts:
        return (
            "Не удалось рассчитать тариф. Пожалуйста, уточните нужные услуги: "
            "rko, acquiring_pos, acquiring_internet, acquiring_sbp, acquiring_mpos, credit."
        )

    header = (
        f"Расчёт для {business_type}, оборот {_format_rub(monthly_turnover)}/мес:\n"
    )
    return header + "\n\n".join(result_parts)


class TariffInput(BaseModel):
    business_type: str = Field(description="Тип бизнеса: ИП, ООО и т.д.")
    monthly_turnover: Union[int, str] = Field(
        description=(
            "Ежемесячный оборот в рублях. Передавай целое число без пробелов "
            "и символов валюты, например 300000 (не '300 000 ₽', не '300,000')."
        )
    )
    services: List[str] = Field(
        description=(
            "Список нужных услуг. Допустимые значения: "
            "rko, acquiring_pos, acquiring_internet, acquiring_sbp, acquiring_mpos, credit"
        )
    )

    @field_validator("monthly_turnover", mode="before")
    @classmethod
    def coerce_turnover(cls, v) -> int:
        return int(str(v).replace(" ", "").replace(",", ""))


def create_tariff_tool() -> StructuredTool:
    def calculate_tariff(
        business_type: str,
        monthly_turnover: Union[int, str],
        services: List[str],
    ) -> str:
        turnover = int(str(monthly_turnover).replace(" ", "").replace(",", ""))
        logger.info(
            "calculate_tariff: %s, turnover=%d, services=%s",
            business_type,
            turnover,
            services,
        )
        return _calculate(business_type, turnover, services)

    return StructuredTool.from_function(
        func=calculate_tariff,
        name="calculate_tariff",
        description=(
            "Рассчитывает стоимость тарифа по параметрам бизнеса. "
            "Используй когда знаешь оборот клиента и нужные услуги. "
            "Параметр services: список строк из допустимых значений: "
            "rko, acquiring_pos, acquiring_internet, acquiring_sbp, acquiring_mpos, credit."
        ),
        args_schema=TariffInput,
    )
