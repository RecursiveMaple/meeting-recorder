"""LLM API client for sentence summarization."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM API client."""

    api_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.2"
    timeout: float = 5.0
    max_tokens: int = 100
    temperature: float = 0.3


class LLMClient:
    """Async LLM API client supporting OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(timeout=self.config.timeout + 1.0)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def summarize(self, text: str, system_prompt: str, user_prompt_template: str = "{{text}}", **kwargs) -> str:
        """Generate summary for the given text.

        Args:
            text: The text to summarize
            system_prompt: System prompt for the LLM
            user_prompt_template: Template with {{text}} placeholder
            **kwargs: Additional template variables

        Returns:
            Generated summary string

        Raises:
            asyncio.TimeoutError: If request times out
            httpx.HTTPError: If HTTP request fails
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.config.timeout + 1.0)

        # Build user prompt from template
        user_prompt = user_prompt_template.replace("{{text}}", text)
        for key, value in kwargs.items():
            user_prompt = user_prompt.replace(f"{{{{{key}}}}}", str(value))

        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        try:
            response = await asyncio.wait_for(
                self._client.post(f"{self.config.api_url}/chat/completions", headers=self._get_headers(), json=payload),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except asyncio.TimeoutError:
            logger.warning(f"LLM request timed out after {self.config.timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Default system prompts for different templates
DEFAULT_SYSTEM_PROMPTS = {
    "meeting_minutes": (
        "你是一个会议助手。请用简洁的语言总结以下会议内容，"
        "提取关键决策、行动项和重要信息。"
        "总结应该简短（1-2句话），便于快速浏览。"
    ),
    "interview": (
        "你是一个面试记录助手。请总结以下面试对话，"
        "提取候选人的关键回答要点和技能亮点。"
        "总结应该简短（1-2句话），便于快速回顾。"
    ),
    "general": ("请用简洁的语言总结以下内容，提取关键信息。总结应该简短（1-2句话）。"),
}

DEFAULT_USER_TEMPLATES = {"meeting_minutes": "{{text}}", "interview": "{{text}}", "general": "{{text}}"}
