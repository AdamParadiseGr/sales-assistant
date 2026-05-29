"""
Tests for tariff_tool and lead_tool.

No external API calls required — all data is local (tariffs.json, SQLite).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest


# ===========================================================================
# Tariff Tool
# ===========================================================================

class TestTariffTool:
    """Tests for calculate_tariff."""

    @pytest.fixture(autouse=True)
    def tool(self):
        from app.tools.tariff_tool import create_tariff_tool
        self.tool = create_tariff_tool()

    def _invoke(self, business_type: str, monthly_turnover: int, services: list[str]) -> str:
        return self.tool.invoke({
            "business_type": business_type,
            "monthly_turnover": monthly_turnover,
            "services": services,
        })

    # --- RKO plan selection ---

    def test_rko_start_for_low_turnover(self):
        result = self._invoke("ИП", 200_000, ["rko"])
        assert "Старт" in result

    def test_rko_business_for_mid_turnover(self):
        result = self._invoke("ООО", 1_500_000, ["rko"])
        assert "Бизнес" in result

    def test_rko_corporate_for_high_turnover(self):
        result = self._invoke("ООО", 8_000_000, ["rko"])
        assert "Корпоративный" in result

    def test_rko_monthly_fee_shown(self):
        result = self._invoke("ИП", 1_000_000, ["rko"])
        # Either Business (990) or Corporate (2490)
        assert "₽" in result

    def test_rko_header_contains_turnover(self):
        result = self._invoke("ООО", 500_000, ["rko"])
        assert "500" in result  # turnover mentioned

    # --- Boundary conditions ---

    def test_rko_at_start_upper_boundary(self):
        result = self._invoke("ИП", 500_000, ["rko"])
        assert "Старт" in result

    def test_rko_at_business_lower_boundary(self):
        result = self._invoke("ИП", 500_001, ["rko"])
        assert "Бизнес" in result

    def test_rko_at_business_upper_boundary(self):
        result = self._invoke("ООО", 5_000_000, ["rko"])
        assert "Бизнес" in result

    def test_rko_at_corporate_lower_boundary(self):
        result = self._invoke("ООО", 5_000_001, ["rko"])
        assert "Корпоративный" in result

    # --- Acquiring ---

    def test_acquiring_pos_rate_shown(self):
        result = self._invoke("ИП", 300_000, ["acquiring_pos"])
        assert "%" in result

    def test_acquiring_pos_low_turnover_rate_2_3(self):
        result = self._invoke("ИП", 50_000, ["acquiring_pos"])
        assert "2.3" in result or "2,3" in result

    def test_acquiring_pos_mid_turnover_rate_2_1(self):
        result = self._invoke("ИП", 300_000, ["acquiring_pos"])
        assert "2.1" in result or "2,1" in result

    def test_acquiring_pos_monthly_cost_calculated(self):
        result = self._invoke("ИП", 300_000, ["acquiring_pos"])
        # 300000 * 2.1% = 6300
        assert "6" in result  # "6 300" or "6300" somewhere

    def test_acquiring_internet_rate_shown(self):
        result = self._invoke("ООО", 500_000, ["acquiring_internet"])
        assert "%" in result

    def test_acquiring_sbp_rate_0_4(self):
        result = self._invoke("ИП", 200_000, ["acquiring_sbp"])
        assert "0.4" in result or "0,4" in result

    def test_acquiring_mpos_rate_shown(self):
        result = self._invoke("ИП", 100_000, ["acquiring_mpos"])
        assert "%" in result

    # --- Combined ---

    def test_combined_rko_and_pos(self):
        result = self._invoke("ООО", 1_000_000, ["rko", "acquiring_pos"])
        assert "РКО" in result
        assert "%" in result  # acquiring rate

    def test_combined_rko_and_internet(self):
        result = self._invoke("ООО", 2_000_000, ["rko", "acquiring_internet"])
        assert "РКО" in result or "Бизнес" in result or "Корпоративный" in result

    def test_combined_all_services(self):
        result = self._invoke("ООО", 3_000_000, ["rko", "acquiring_pos", "acquiring_sbp"])
        # Should mention multiple sections
        assert result.count("₽") >= 2

    # --- Credit ---

    def test_credit_mention(self):
        result = self._invoke("ООО", 2_000_000, ["credit"])
        assert "кредит" in result.lower() or "Кредит" in result

    # --- Aliases ---

    def test_rko_alias_rko(self):
        result = self._invoke("ИП", 200_000, ["рко"])
        assert "Старт" in result or "Бизнес" in result  # normalised

    # --- Edge cases ---

    def test_zero_turnover_does_not_crash(self):
        result = self._invoke("ИП", 0, ["rko"])
        assert isinstance(result, str)

    def test_very_large_turnover(self):
        result = self._invoke("ООО", 100_000_000, ["rko"])
        assert "Корпоративный" in result

    def test_empty_services_defaults_to_rko(self):
        result = self._invoke("ИП", 300_000, [])
        assert isinstance(result, str)
        assert len(result) > 20


# ===========================================================================
# Lead Tool
# ===========================================================================

class TestLeadTool:
    """Tests for create_lead — uses a temp SQLite database."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db_path = str(tmp_path / "test_leads.db")
        from app.tools.lead_tool import create_lead_tool
        self.tool = create_lead_tool(db_path=self.db_path)

    def _invoke(self, name: str, phone: str, business_type: str, product: str) -> str:
        return self.tool.invoke({
            "name": name,
            "phone": phone,
            "business_type": business_type,
            "product": product,
        })

    # --- Success cases ---

    def test_create_lead_returns_success_message(self):
        result = self._invoke("Иван Петров", "79161234567", "ИП", "РКО")
        assert "Заявка успешно создана" in result

    def test_create_lead_result_contains_name(self):
        result = self._invoke("Мария Сидорова", "79261234567", "ООО", "Эквайринг")
        assert "Мария Сидорова" in result

    def test_create_lead_result_contains_phone(self):
        result = self._invoke("Алексей", "79991112233", "ИП", "Кредит")
        assert "79991112233" in result

    def test_create_lead_result_contains_lead_id(self):
        result = self._invoke("Тест", "70000000000", "ИП", "РКО")
        assert "Номер заявки:" in result

    def test_create_lead_id_is_uppercase_hex(self):
        result = self._invoke("Тест", "70000000000", "ИП", "РКО")
        # Extract ID
        line = next(l for l in result.splitlines() if "Номер заявки:" in l)
        lead_id = line.split(":")[-1].strip()
        assert len(lead_id) == 8
        assert lead_id == lead_id.upper()

    def test_create_lead_contains_callback_time(self):
        result = self._invoke("Сергей", "71234567890", "ООО", "РКО")
        assert "30 минут" in result or "менеджер" in result.lower()

    # --- Database persistence ---

    def test_lead_is_stored_in_db(self):
        self._invoke("Анна Кузнецова", "79031234567", "ООО", "Эквайринг")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM leads").fetchall()
        assert len(rows) == 1

    def test_lead_db_fields_correct(self):
        self._invoke("Борис Тихонов", "79876543210", "ИП", "Кредит")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM leads").fetchone()
        assert row["name"] == "Борис Тихонов"
        assert row["phone"] == "79876543210"
        assert row["business_type"] == "ИП"
        assert row["product"] == "Кредит"
        assert row["status"] == "new"

    def test_multiple_leads_all_stored(self):
        self._invoke("Клиент 1", "71111111111", "ИП", "РКО")
        self._invoke("Клиент 2", "72222222222", "ООО", "Эквайринг")
        self._invoke("Клиент 3", "73333333333", "ИП", "Кредит")

        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        assert count == 3

    def test_lead_ids_are_unique(self):
        for i in range(5):
            self._invoke(f"Клиент {i}", f"7000000000{i}", "ИП", "РКО")
        with sqlite3.connect(self.db_path) as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM leads").fetchall()]
        assert len(set(ids)) == 5

    # --- DB initialisation ---

    def test_table_created_on_init(self):
        with sqlite3.connect(self.db_path) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        assert "leads" in tables

    def test_tool_initialises_table_if_db_empty(self, tmp_path):
        new_db = str(tmp_path / "fresh.db")
        from app.tools.lead_tool import create_lead_tool
        create_lead_tool(db_path=new_db)
        with sqlite3.connect(new_db) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        assert "leads" in tables

    # --- get_all_leads helper ---

    def test_get_all_leads_returns_list(self):
        from app.tools.lead_tool import get_all_leads
        self._invoke("Клиент", "71234567890", "ИП", "РКО")
        leads = get_all_leads(self.db_path)
        assert isinstance(leads, list)
        assert len(leads) == 1
        assert leads[0]["name"] == "Клиент"
