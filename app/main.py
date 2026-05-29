from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# These are initialised in lifespan so the embedding model loads once.
_sales_agent = None
_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sales_agent, _bot

    from app.agent import SalesAgent
    _sales_agent = SalesAgent()
    logger.info("SalesAgent initialised")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        from telegram import Bot
        _bot = Bot(token=token)
        logger.info("Telegram bot initialised")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="AI Sales Assistant — Точка Банк",
    description="RAG + function-calling агент для продажи банковских продуктов МСБ",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class ResetRequest(BaseModel):
    session_id: str


class WebhookSetupRequest(BaseModel):
    base_url: str  # e.g. "https://example.com"


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": _sales_agent is not None,
        "active_sessions": _sales_agent.active_sessions if _sales_agent else 0,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _sales_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not ready")
    response = await _sales_agent.chat(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, response=response)


@app.post("/reset")
async def reset_session(req: ResetRequest):
    if _sales_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not ready")
    _sales_agent.reset_session(req.session_id)
    return {"ok": True, "session_id": req.session_id}


@app.get("/profile/{session_id}")
async def get_profile(session_id: str):
    if _sales_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not ready")
    return _sales_agent.get_client_profile(session_id)


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    expected = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    if _bot is None or _sales_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not ready")

    from telegram import Update
    data = await request.json()

    try:
        update = Update.de_json(data, _bot)
    except Exception as exc:
        logger.warning("Failed to parse Telegram update: %s", exc)
        return {"ok": True}

    if not (update.message and update.message.text):
        return {"ok": True}

    chat_id = update.effective_chat.id
    session_id = str(chat_id)
    user_text = update.message.text.strip()

    if not user_text:
        return {"ok": True}

    logger.info("Telegram [%s]: %s", session_id, user_text[:80])

    try:
        await _bot.send_chat_action(chat_id=chat_id, action="typing")
        response = await _sales_agent.chat(session_id, user_text)
        await _bot.send_message(chat_id=chat_id, text=response)
    except Exception as exc:
        logger.exception("Error handling Telegram message: %s", exc)
        await _bot.send_message(
            chat_id=chat_id,
            text="Произошла техническая ошибка. Попробуйте ещё раз.",
        )

    return {"ok": True}


@app.post("/webhook/setup")
async def setup_webhook(req: WebhookSetupRequest):
    if _bot is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bot not initialised")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    webhook_url = f"{req.base_url.rstrip('/')}/webhook/{token}"
    await _bot.set_webhook(url=webhook_url)
    logger.info("Webhook set to: %s", webhook_url)
    return {"ok": True, "webhook_url": webhook_url}
