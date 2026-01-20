"""Secure token storage and lifecycle management."""

import json
from pathlib import Path

from google.oauth2.credentials import Credentials

from config.logging_config import get_logger

logger = get_logger(__name__)


class TokenManager:
    """Manages OAuth token storage and retrieval."""

    def __init__(self, token_path: Path):
        """Initialize token manager.

        Args:
            token_path: Path to store the token file
        """
        self.token_path = token_path

    def save_token(self, credentials: Credentials) -> None:
        """Persist credentials to storage.

        Args:
            credentials: Google OAuth credentials to save
        """
        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else [],
        }

        self.token_path.write_text(json.dumps(token_data, indent=2))
        logger.info("token_saved", path=str(self.token_path))

    def load_token(self) -> Credentials | None:
        """Load credentials from storage.

        Returns:
            Credentials if token file exists and is valid, None otherwise
        """
        if not self.token_path.exists():
            logger.debug("token_not_found", path=str(self.token_path))
            return None

        try:
            token_data = json.loads(self.token_path.read_text())
            credentials = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes"),
            )
            logger.info("token_loaded", path=str(self.token_path))
            return credentials
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("token_load_failed", error=str(e))
            return None

    def delete_token(self) -> None:
        """Delete stored token."""
        if self.token_path.exists():
            self.token_path.unlink()
            logger.info("token_deleted", path=str(self.token_path))

    def is_token_valid(self, credentials: Credentials | None) -> bool:
        """Check if credentials are valid.

        Args:
            credentials: Credentials to check

        Returns:
            True if credentials exist and are not expired
        """
        if credentials is None:
            return False
        return credentials.valid or credentials.refresh_token is not None
