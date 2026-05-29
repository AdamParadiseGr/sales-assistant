from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class AllKeysExhaustedError(Exception):
    """Raised when every configured Groq API key has hit its rate limit."""


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True if *exc* looks like a Groq / HTTP-429 rate-limit error."""
    try:
        from groq import RateLimitError as _GroqRL  # type: ignore[import]
        if isinstance(exc, _GroqRL):
            return True
    except ImportError:
        pass
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate_limit" in name or "429" in msg:
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_rate_limit(cause)
    return False


class GroqKeyManager:
    """
    Manages a pool of Groq API keys and rotates through them when a rate
    limit (HTTP 429) is encountered.

    Keys are read from numbered env vars:
        GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, …

    Falls back to the legacy GROQ_API_KEY if no numbered keys are set.

    You can also pass *keys* directly (handy for tests):
        GroqKeyManager(keys=["gsk_a", "gsk_b"])
    """

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys: list[str] = keys if keys is not None else self._load_keys()
        self._index: int = 0
        logger.info("GroqKeyManager initialised with %d key(s)", len(self._keys))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_keys() -> list[str]:
        keys: list[str] = []
        for i in range(1, 20):
            val = os.environ.get(f"GROQ_API_KEY_{i}")
            if val:
                keys.append(val)
            else:
                break
        if not keys:
            fallback = os.environ.get("GROQ_API_KEY")
            if fallback:
                keys.append(fallback)
        if not keys:
            raise ValueError(
                "No Groq API keys found in environment. "
                "Set GROQ_API_KEY or GROQ_API_KEY_1 / _2 / _3 in .env"
            )
        return keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_key(self) -> str:
        """Return the currently active API key."""
        return self._keys[self._index]

    def rotate(self) -> str:
        """
        Advance to the next key and return it.

        Raises AllKeysExhaustedError when there are no more keys to try.
        """
        prev = self._index
        self._index += 1
        if self._index >= len(self._keys):
            self._index = len(self._keys) - 1  # stay in bounds
            logger.error(
                "All %d Groq API key(s) exhausted — no more keys to rotate to",
                len(self._keys),
            )
            raise AllKeysExhaustedError(
                f"All {len(self._keys)} Groq API key(s) have hit their rate limits."
            )
        logger.warning(
            "Groq API key rotated: index %d → %d  (%d keys total)",
            prev,
            self._index,
            len(self._keys),
        )
        return self._keys[self._index]

    def reset(self) -> None:
        """Reset rotation back to the first key (call between requests if desired)."""
        self._index = 0
        logger.debug("GroqKeyManager reset to key index 0")

    @property
    def key_count(self) -> int:
        """Total number of configured keys."""
        return len(self._keys)
