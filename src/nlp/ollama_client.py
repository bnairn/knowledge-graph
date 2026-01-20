"""Ollama client for local LLM inference."""

import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel

from config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """Ollama client for local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 300.0,
    ):
        """Initialize the client.

        Args:
            base_url: Ollama API base URL
            model: Model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send completion request.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Max tokens (ignored for Ollama, uses model default)

        Returns:
            Response text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            logger.debug("ollama_completion", model=self.model, response_length=len(content))
            return content

        except httpx.HTTPError as e:
            logger.error("ollama_error", error=str(e))
            raise

    def extract_structured(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
    ) -> T:
        """Extract structured data by prompting for JSON.

        Args:
            prompt: User prompt
            schema: Pydantic model class for the response
            system: System prompt

        Returns:
            Parsed Pydantic model instance
        """
        # Build a prompt that asks for JSON output
        schema_json = schema.model_json_schema()

        json_prompt = f"""{prompt}

Respond with valid JSON that matches this schema:
{json.dumps(schema_json, indent=2)}

Return ONLY the JSON object, no other text."""

        full_system = system or ""
        full_system += "\nYou must respond with valid JSON only. No markdown, no explanation, just the JSON object."

        response = self.complete(json_prompt, system=full_system)

        # Try to extract JSON from response
        json_str = self._extract_json(response)

        try:
            data = json.loads(json_str)
            return schema.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("ollama_json_parse_error", error=str(e), response=response[:500])
            # Return empty result
            return schema.model_validate({schema.model_fields.keys().__iter__().__next__(): []})

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain other content.

        Args:
            text: Raw response text

        Returns:
            Extracted JSON string
        """
        # Try to find JSON object or array
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're code fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group()

        # Find JSON array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return match.group()

        return text

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics (minimal for local inference).

        Returns:
            Dictionary with stats
        """
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "note": "Local inference - no token tracking",
        }

    def reset_usage_stats(self) -> None:
        """Reset usage stats (no-op for Ollama)."""
        pass

    def check_connection(self) -> bool:
        """Check if Ollama is running and model is available.

        Returns:
            True if connection successful
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if our model is available
            if not any(self.model in name for name in model_names):
                logger.warning(
                    "ollama_model_not_found",
                    model=self.model,
                    available=model_names,
                )
                return False

            logger.info("ollama_connected", model=self.model)
            return True
        except Exception as e:
            logger.error("ollama_connection_failed", error=str(e))
            return False
