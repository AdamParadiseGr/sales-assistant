from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_WINDOW = 10  # keep last 10 exchanges (20 messages)


class ClientProfile:
    """Accumulates structured client data across the conversation."""

    def __init__(self) -> None:
        self.business_type: Optional[str] = None
        self.turnover: Optional[int] = None
        self.needs: list[str] = []
        self.name: Optional[str] = None
        self.phone: Optional[str] = None

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                logger.warning("ClientProfile: unknown field '%s'", key)
                continue
            if key == "needs" and isinstance(value, str):
                if value not in self.needs:
                    self.needs.append(value)
            else:
                setattr(self, key, value)

    def is_qualified(self) -> bool:
        return bool(self.business_type and self.turnover)

    def can_create_lead(self) -> bool:
        return bool(self.name and self.phone)

    def to_dict(self) -> dict:
        return {
            "business_type": self.business_type,
            "turnover": self.turnover,
            "needs": self.needs,
            "name": self.name,
            "phone": self.phone,
        }

    def __repr__(self) -> str:
        return f"ClientProfile({self.to_dict()})"


class SessionMemory:
    """Per-session container: message history + client profile."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._messages: deque[BaseMessage] = deque(maxlen=_WINDOW * 2)
        self.client_profile = ClientProfile()

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def update_profile(self, **kwargs) -> None:
        self.client_profile.update(**kwargs)

    def get_history_text(self, last_n: int = 6) -> str:
        """Return recent messages as plain text for the evaluator."""
        messages = list(self._messages)
        lines: list[str] = []
        for msg in messages[-last_n:]:
            role = "Клиент" if isinstance(msg, HumanMessage) else "Ассистент"
            lines.append(f"{role}: {msg.content[:300]}")
        return "\n".join(lines)


class MemoryManager:
    """Registry of all active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id)
            logger.info("New session created: %s", session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)
