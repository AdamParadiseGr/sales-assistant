"""Console demo — interactive chat with the Sales Agent."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    from app.agent import SalesAgent

    agent = SalesAgent()
    model_name = agent._main_llm.model_name
    print(f"AI Sales Assistant | Groq / {model_name}")
    print("Введите 'exit' или 'выход' для завершения.\n")
    session_id = "demo"

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "выход"):
            print("До свидания!")
            break

        response = await agent.chat(session_id, user_input)
        print(f"\nАссистент: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
