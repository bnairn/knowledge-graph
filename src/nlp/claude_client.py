"""Claude API client with rate limiting and error handling."""

import time
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeClient:
    """Claude API client with rate limiting and structured output support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        requests_per_minute: int = 50,
    ):
        """Initialize the client.

        Args:
            api_key: Anthropic API key
            model: Model to use
            max_tokens: Maximum tokens per response
            requests_per_minute: Rate limit
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.requests_per_minute = requests_per_minute
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Dynamic rate limiter
        self._last_request_time = 0.0
        self._min_interval = 60.0 / requests_per_minute

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send completion request with rate limiting.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Max tokens for this request

        Returns:
            Response text
        """
        self._rate_limit()

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system or "",
                messages=messages,
            )

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            logger.debug(
                "claude_completion",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            # Extract text from response
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text

            return text_content

        except anthropic.RateLimitError as e:
            logger.warning("claude_rate_limited", error=str(e))
            raise
        except anthropic.APIError as e:
            logger.error("claude_api_error", error=str(e))
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def extract_structured(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
    ) -> T:
        """Extract structured data using tool use.

        Args:
            prompt: User prompt
            schema: Pydantic model class for the response
            system: System prompt

        Returns:
            Parsed Pydantic model instance
        """
        self._rate_limit()

        # Convert Pydantic schema to tool definition
        tool_name = schema.__name__.lower()
        tool = {
            "name": tool_name,
            "description": f"Extract {schema.__name__} from the text",
            "input_schema": schema.model_json_schema(),
        }

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or "",
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
            )

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            # Extract tool use result
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    return schema.model_validate(block.input)

            raise ValueError(f"No tool use found in response for {tool_name}")

        except anthropic.RateLimitError as e:
            logger.warning("claude_rate_limited", error=str(e))
            raise
        except anthropic.APIError as e:
            logger.error("claude_api_error", error=str(e))
            raise

    def get_usage_stats(self) -> dict[str, int]:
        """Get token usage statistics.

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def reset_usage_stats(self) -> None:
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
